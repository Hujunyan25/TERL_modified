import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from utils import logger as logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TERLAddTemporalConfig:
    """Configuration class for TERL policy network"""
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    action_size: int = 9
    num_quantiles: int = 32
    num_cosine_features: int = 64
    device: str = 'cpu'
    seed: int = 0
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    n_history: int = 5


def encoder(input_dimension: int, output_dimension: int) -> nn.Sequential:
    """Create encoder model"""
    return nn.Sequential(
        nn.Linear(input_dimension, output_dimension),
        nn.LayerNorm(output_dimension),
        nn.ReLU()
    )


class TargetSelectionModule(nn.Module):
    """Attention module specifically for target selection of evaders"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.self_transform = nn.Linear(2 * hidden_dim, hidden_dim)

        # Linear layer for transforming query
        self.query_transform = nn.Linear(hidden_dim, hidden_dim)

        # Linear layers for transforming key and value
        self.key_transform = nn.Linear(hidden_dim, hidden_dim)
        self.value_transform = nn.Linear(hidden_dim, hidden_dim)

        # Layer Norm for feature fusion
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Scaling factor
        self.scale = math.sqrt(hidden_dim)

    def forward(self, self_feature: torch.Tensor, evader_features: torch.Tensor,
                evader_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            self_feature: Self features [batch_size, 2 * hidden_dim]
            evader_features: Evader features [batch_size, num_evaders, hidden_dim]
            evader_mask: Evader mask [batch_size, num_evaders]

        Returns:
            enhanced_feature: Enhanced features [batch_size, hidden_dim]
            attention_weights: Attention weights [batch_size, num_evaders]
        """
        batch_size = self_feature.shape[0]  # [B, 2H]
        hidden_dim = self.hidden_dim
        num_evaders = evader_features.shape[1]

        # Add shape checking
        assert self_feature.shape == (batch_size, 2 * hidden_dim), f"Unexpected self_feature shape: {self_feature.shape}"
        assert evader_features.shape == (
            batch_size, num_evaders, hidden_dim), f"Unexpected evader_features shape: {evader_features.shape}"
        assert evader_mask.shape == (batch_size, num_evaders), f"Unexpected evader_mask shape: {evader_mask.shape}"

        self_feature = self.self_transform(self_feature)  # [B, H] # Transform self features

        # Transform query (self features)
        query = self.query_transform(self_feature).unsqueeze(1)  # [B, 1, H]

        # Transform key and value (evader features)
        keys = self.key_transform(evader_features)  # [B, N, H]
        values = self.value_transform(evader_features)  # [B, N, H]

        # Calculate attention scores
        scores = torch.matmul(query, keys.transpose(-2, -1)) / self.scale  # [B, 1, N]

        # Validate scores shape
        assert scores.shape == (batch_size, 1, num_evaders), f"Unexpected scores shape: {scores.shape}"

        # Apply mask
        if evader_mask is not None:
            scores = scores.masked_fill((~evader_mask.bool()).unsqueeze(1), float('-inf'))

        # Get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [B, 1, N]

        # Get weighted features
        weighted_features = torch.matmul(attention_weights, values)  # [B, 1, H]
        weighted_features = weighted_features.squeeze(1)  # [B, H]

        # Feature fusion and normalization
        enhanced_feature = self.layer_norm(self_feature + weighted_features)

        # Final shape checking
        assert enhanced_feature.shape == (batch_size, hidden_dim)
        assert attention_weights.squeeze(1).shape == (batch_size, num_evaders)

        return enhanced_feature, attention_weights.squeeze(1)


class TERL_add_temporal(nn.Module):
    """
    TERL policy network implementation
    Uses Transformer architecture to process multi-entity inputs
    """

    def __init__(self, config: Optional[TERLAddTemporalConfig] = None, **kwargs):
        """
        Initialize TERL policy network

        Args:
            config: TERL configuration object
            **kwargs: Optional configuration parameters, will override defaults in config
        """
        super().__init__()

        # Initialize parameters using config class or kwargs
        if config is None:
            config = TERLAddTemporalConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.hidden_dim = config.hidden_dim
        self.device = config.device
        torch.manual_seed(config.seed)

        # Define feature dimensions for various entity types
        self.feature_dims = {
            'self': 4,  # [vx, vy, min_obs_dis, pursuing_signal]
            'pursuers': 7,  # [px, py, vx, vy, dist, angle, pursuing_signal]
            'evaders': 7,  # [px, py, vx, vy, dist, pos_angle, head_angle]
            'obstacles': 5  # [px, py, radius, dist, angle]
        }

        # 新增：拼接融合线性层（2D→D），搭配LayerNorm
        self.concat_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True)  # 可选：增加非线性，提升融合能力
        )  

        self.final_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * (TERLAddTemporalConfig.n_history - 1), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True)  # 可选：增加非线性，提升融合能力
        )  

        # IQN parameters
        self.K = config.num_quantiles
        self.n = config.num_cosine_features
        # Precompute π values for cosine features
        self.register_buffer('pis',
                             torch.FloatTensor([np.pi * i for i in range(self.n)]).view(1, 1, self.n))

        # Define feature encoders
        self.entity_encoders = nn.ModuleDict({
            name: encoder(dim, self.hidden_dim)
            for name, dim in self.feature_dims.items()
        })

        # Type embedding
        self.type_embedding = nn.Embedding(4, self.hidden_dim)

        # 新增：运动补偿模块：单步位移->hidden_dim,实现历史特征时空对齐
        #位移编码器：2D位移（X/Y偏移）-> hidden_dim
        self.displacement_encoder = nn.Sequential(
            nn.Linear(2, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        ).to(self.device)

        #新增2，时间编码模块：
        self.time_emb_curr = nn.Parameter(torch.randn(self.hidden_dim, device=self.device)) #当前帧时间编码
        self.time_emb_his = nn.Parameter(torch.randn(self.hidden_dim, device=self.device)) #历史帧时间编码

        #新增3.时序融合模块：拼接当前特征+补偿后的历史特征，压缩回到hidden_dim
        self.temporal_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim)
        ).to(self.device)

        # 历史缓存：列表式存储最近3帧，先进先出 [feat, pos, mask]
        # feat: [B, M, D], pos: [B, 2], mask: [B, M]
        self.history_cache = {
            'transformed_feat': [],  # Transformer编码后的实体特征
            'self_original_pos': [],  # Self的原始2D位置（用于计算位移）
            'global_mask': []         # 全局实体掩码（[B, M]）
        }

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,  # Use Pre-LN structure to improve stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False
        )

        #新增4.历史缓存机制：缓存Transformer输入之前的最终融合的特征+上一时刻self位置
        #形状：his_cache_feat [batch_size, 实体总数，hidden_dim]
        #形状：his_cahce_pos [Batch_size, 2](批次，self的2D原始位置)
        self.his_cache_feat = None
        self.his_cache_pos = None

        # Add target selection module
        self.target_selection = TargetSelectionModule(self.hidden_dim)

        # IQN related layers
        self.cos_embedding = nn.Linear(self.n, self.hidden_dim)

        # Output layer
        self.hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, config.action_size)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        # Initialize weights
        self._init_weights()

        # Move model to specified device
        self.to(self.device)

        # For storing the last attention weights
        self._last_target_weights = None
        self._last_transformer_weights = None

    def _init_weights(self):
        """Initialize network weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    # Embedding layer initialized with standard normal distribution
                    nn.init.normal_(param.data, mean=0.0, std=0.02)
                elif 'layer_norm' in name:
                    # LayerNorm layer weights initialized to 1
                    nn.init.constant_(param.data, 1.0)
                elif 'output_layer' in name:
                    # Output layer uses a smaller initialization range to avoid overly large initial values
                    nn.init.uniform_(param.data, -0.003, 0.003)
                elif param.dim() > 1:
                    # For 2D and higher weights, use orthogonal initialization with gain=1/sqrt(2) for better initial scaling
                    nn.init.orthogonal_(param.data, gain=1 / math.sqrt(2))
                else:
                    # 1D weights use uniform distribution
                    bound = 1 / math.sqrt(param.size(0))
                    nn.init.uniform_(param.data, -bound, bound)
            elif 'bias' in name:
                # Bias terms initialized to 0
                nn.init.constant_(param.data, 0)


    def reset_history_cache(self):
        '''重置历史缓存，适配新的任务/新的episode'''
        self.history_cache['transformed_feat'].clear()
        self.history_cache['self_original_pos'].clear()
        self.history_cache['global_mask'].clear()


    def _update_history_cache(self, embedded_feature: torch.Tensor, self_ori_pos: torch.Tensor, global_mask: torch.Tensor):
        """
        更新历史缓存，先进先出，保留最近n_history帧
        Args:
            transformed_feat: Transformer编码后的特征 [B, M, D]
            self_ori_pos: Self原始2D位置 [B, 2]
            global_mask: 全局掩码 [B, M]
        """
        # 存入当前帧数据
        embedded_feature_detach = embedded_feature.detach().unsqueeze(0)
        self_ori_pos_detach = self_ori_pos.detach().unsqueeze(0)
        global_mask_detach = global_mask.detach().unsqueeze(0)
        self.history_cache['transformed_feat'].append(embedded_feature_detach)
        self.history_cache['self_original_pos'].append(self_ori_pos_detach)
        self.history_cache['global_mask'].append(global_mask_detach)
        # 超出缓存帧数则弹出最旧帧
        if len(self.history_cache['transformed_feat']) > TERLAddTemporalConfig.n_history:
            self.history_cache['transformed_feat'].pop(0)
            self.history_cache['self_original_pos'].pop(0)
            self.history_cache['global_mask'].pop(0)

    def _multi_step_motion_compensation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多步运动补偿：基于5帧历史缓存，逐实体单步掩码交集，时间衰减加权融合多帧补偿特征
        Returns:
            fused_his_feat: 融合后的历史特征 [B, M, D]
            final_fusion_mask: 最终融合掩码 [B, M]
        """
        n_cached = len(self.history_cache['transformed_feat'])
        # 缓存不足5帧，无历史可补偿，返回全零
        if n_cached < 5:
            B = self.history_cache['transformed_feat'][-1].shape[1]
            M = self.history_cache['transformed_feat'][-1].shape[2]
            return torch.zeros(B, M, self.hidden_dim, device=self.device), self.history_cache['global_mask'][-1].squeeze(0)

        compensated_history_feats = []
        weight_list = []
        fusion_mask_list = []
        # 遍历所有连续历史帧对（i, i+1），逐对做单步补偿
        for i in range(n_cached - 1):
            weight_list.clear()
            # 取第i帧（历史）和第i+1帧（次历史）
            his_feat_i = self.history_cache['transformed_feat'][i].squeeze(0)
            his_pos_i = self.history_cache['self_original_pos'][i].squeeze(0)
            his_mask_i = self.history_cache['global_mask'][i].squeeze(0)

            his_feat_j = self.history_cache['transformed_feat'][i+1].squeeze(0)
            his_pos_j = self.history_cache['self_original_pos'][i+1].squeeze(0)
            his_mask_j = self.history_cache['global_mask'][i+1].squeeze(0)

            B, M, D = his_feat_i.shape
            # 1. 逐实体单步掩码交集：仅连续两帧有效才参与补偿
            step_fusion_mask = his_mask_i * his_mask_j  # [B, M]
            # 2. 计算单步位移：Self位置差 [B, 2]
            delta_pos = his_pos_j - his_pos_i  # [B, 2]
            # 3. 位移编码为特征 [B, D]
            delta_feat = self.displacement_encoder(delta_pos)  # [B, D]
            delta_feat = delta_feat.unsqueeze(1)  # [B, 1, D]
            delta_feat = delta_feat.expand(-1, M, -1)  # [B, M, D]
            # 4. 历史特征运动补偿：广播位移到所有实体 [B, M, D]
            compensated_feat = his_feat_i + delta_feat # [B, M, D]

            # 将后一帧和当前帧进行融合
            fused_feat = self.concat_fusion(torch.cat([his_feat_j, delta_feat], dim=-1))  # [B, M, D]
            # 5. 掩码屏蔽无效实体(fused_feat的形状为[B, M, D]， step_fusion_mask的形状为[B, M])
            compensated_feat = fused_feat * step_fusion_mask.unsqueeze(-1)  # [B, M, D]
            fusion_mask_list.append(step_fusion_mask)
            # # 6. 提取Self特征（第0位，与原始逻辑一致）
            # compensated_self_feat = compensated_feat[:, 0, :]  # [B, D]
            # compensated_his_feats.append(compensated_self_feat)
            # compensated_his_feats.append(compensated_feat)

            # 时间衰减权重：越近的帧权重越高（i越大，权重越高）
            weight = (i + 1) / sum(range(1, n_cached))
            weight_tensor = torch.tensor(weight, device=self.device, dtype=torch.float32)
            weight_list.append(weight_tensor)
            weight_tensored = torch.stack(weight_list).unsqueeze(-1).unsqueeze(-1).to(self.device)  # [1,1,1]
            weight_tensored = weight_tensored.permute(1, 2, 0).expand(B, M, -1) # 维度：[1, M, 1]
            weighted_compensated_feat = weight_tensored * compensated_feat # 维度：[B, M, D]
            compensated_history_feats.append(weighted_compensated_feat)

        # 多帧补偿特征加权融合
        # weights = weights.permute(1, 2, 0, 3).flatten(start_dim= -2).expand(-1, M, -1) # 维度：[1, 1, D*(n-1)]
        compensated_his_feats = torch.stack(compensated_history_feats)  # [n-1, B, M, D]
        compensated_his_feats = compensated_his_feats.permute(1, 2, 0, 3).flatten(start_dim= -2) #维度：[B, M, D*(n-1)]
        print("compensated_his_feats shape:", compensated_his_feats.shape)

        final_fusion_feat = self.final_fusion(compensated_his_feats) #维度：[B, M, D]
        fusions = torch.stack(fusion_mask_list)  # [n-1, B, M]
        print("fusion_mask_list shape:", fusions.shape)
        # 最终融合掩码：最后一对帧的单步掩码
        final_fusion_mask = torch.prod(fusions, dim=0)  # [B, M]

        return final_fusion_feat, final_fusion_mask
    # --------------------------------------------------------------------------------
    
    def _validate_input(self, obs: Dict[str, torch.Tensor]) -> None:
        """Validate the format and dimensions of input data"""
        if not isinstance(obs, dict):
            raise ValueError("obs must be a dictionary")

        required_keys = {'self', 'types', 'masks'}
        if not all(key in obs for key in required_keys):
            raise ValueError(f"obs must contain keys: {required_keys}")

        if obs['self'].dim() != 2:
            raise ValueError("self features must be 2-dimensional [batch_size, feature_dim]")

        if obs['self'].shape[1] != self.feature_dims['self']:
            raise ValueError(f"self features must have dimension {self.feature_dims['self']}")

    def encode_entities(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode entity features and process through Transformer

        Args:
            obs: Dictionary containing features of various entities

        Returns:
            torch.Tensor: Encoded features [batch_size, hidden_dim]
        """
        B = obs['self'].shape[0]
        encoded_features = []

        # Encode various entities
        for entity_type, encoder in self.entity_encoders.items():
            features = obs[entity_type]
            if entity_type == 'self':
                encoded = encoder(features).unsqueeze(1)
                self_original_position = features[:, :2]  # 提取self的原始2D位置
            else:
                encoded = encoder(features)
            encoded_features.append(encoded)

        # Concatenate all encoded features
        entity_embed = torch.cat(encoded_features, dim=1)

        # Add type embedding
        type_embed = self.type_embedding(obs['types'].long())
        tokens = entity_embed + type_embed
        global_mask = obs['masks'] #全局掩码

        #2.更新历史缓存并执行多步运动补偿
        self._update_history_cache(tokens, self_original_position, global_mask)
        fused_history_feat, fusion_mask = self._multi_step_motion_compensation()

        # 3. 时序融合：当前增强特征 + 融合后的历史特征（仅Self维度）
        # 屏蔽无效特征，保证与掩码一致
        print("fused_history_feat shape:", fused_history_feat.shape)
        print("fusion_mask shape:", fusion_mask.shape)
        if (fused_history_feat == 0).all():
            attention_mask = ~global_mask.bool()
            transformed = self.transformer_encoder(tokens, src_key_padding_mask=attention_mask)
        # 拼接融合
        else:
            fused_history_feat = fused_history_feat * fusion_mask.unsqueeze(-1)      # [B, D]
            temporal_fused_feature = self.temporal_fusion(torch.cat([tokens, fused_history_feat], dim=-1))  
            fused_mask_history_and_current = global_mask * fusion_mask  # 融合后的掩码

            # Transformer processing
            attention_mask = ~fused_mask_history_and_current.bool()
            transformed = self.transformer_encoder(temporal_fused_feature, src_key_padding_mask=attention_mask)

        # Get self feature
        self_feature = transformed[:, 0]  # [B, H]
        global_feature = torch.max(transformed, dim=1).values  # [B, H]

        enhanced_feature = torch.cat([self_feature, global_feature], dim=-1)  # [B, 2H]

        # Extract evader features and mask - uniformly handle batch and single samples
        evader_indices = (obs['types'] == 2)  # [B, N]

        # Get the number of positions with type==2 per batch
        num_evaders_per_batch = evader_indices.sum(dim=1)  # [B]
        max_evaders = num_evaders_per_batch.max().item()

        # Use masked_select and reshape to handle irregular selections
        flat_mask = obs['masks'].masked_select(evader_indices)
        flat_features = transformed.masked_select(
            evader_indices.unsqueeze(-1).expand(-1, -1, transformed.size(-1))
        )

        # Reshape into regular shape
        evader_mask = flat_mask.reshape(B, max_evaders)  # [B, max_evaders]
        evader_features = flat_features.reshape(B, max_evaders, -1)  # [B, max_evaders, H]

        # Apply target selection module
        enhanced_features, attention_weights = self.target_selection(
            enhanced_feature,
            evader_features,
            evader_mask
        )

        # Store attention weights for visualization
        self._last_target_weights = attention_weights


        return enhanced_features

    def calc_cos(self, batch_size: int, num_tau: int = 8, cvar: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate cosine values

        Args:
            batch_size: Batch size
            num_tau: Number of tau samples
            cvar: CVaR parameter

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine values and tau values
        """
        if batch_size <= 0 or num_tau <= 0 or cvar <= 0:
            raise ValueError("batch_size, num_tau and cvar must be positive")

        taus = torch.rand(batch_size, num_tau).to(self.device).unsqueeze(-1)
        taus = torch.pow(taus, cvar).clamp(0, 1)
        cos = torch.cos(taus * self.pis)

        return cos, taus

    def forward(self,
                obs: Dict[str, torch.Tensor],
                num_tau: int = 8,
                cvar: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation

        Args:
            obs: Observation data dictionary
            num_tau: Number of tau samples
            cvar: CVaR parameter

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - quantiles: [batch_size, num_tau, action_size]
                - taus: [batch_size, num_tau, 1]
        """
        self._validate_input(obs)
        batch_size = obs['self'].shape[0]

        # Transformer encoding of observations
        #1.实体编码：返回增强特征+Transformer特征+全局掩码+Self原始位置
        current_feature = self.encode_entities(obs)


        # IQN processing

        cos, taus = self.calc_cos(batch_size, num_tau, cvar)
        cos_features = F.relu(self.cos_embedding(cos))

        # Feature combination
        features = (current_feature.unsqueeze(1) * cos_features).view(batch_size * num_tau, -1)

        # Output layer
        features = F.relu(self.hidden_layer(features))
        features = self.layer_norm(features)
        quantiles = self.output_layer(features)

        return quantiles.view(batch_size, num_tau, -1), taus

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get the attention weights from the last forward pass"""
        return {
            'target_selection': self._last_target_weights,
            'transformer': self._last_transformer_weights
        }

    def count_parameters(self) -> int:
        """Count the number of model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, directory: str) -> None:
        """
        Save the model, using version numbers to avoid overwriting existing files

        Args:
            directory: Save directory
        """
        try:
            os.makedirs(directory, exist_ok=True)

            # Find existing version numbers in the directory
            existing_versions = []
            for filename in os.listdir(directory):
                if filename.startswith("network_params_v") and filename.endswith(".pth"):
                    try:
                        version = int(filename[len("network_params_v"):-4])
                        existing_versions.append(version)
                    except ValueError:
                        continue

            # Determine the new version number
            new_version = max(existing_versions, default=0) + 1

            # Save model parameters with the new version number
            params_path = os.path.join(directory, f"network_params_v{new_version}.pth")
            torch.save(self.state_dict(), params_path)

            # Save the corresponding version configuration file
            config_path = os.path.join(directory, f"config_v{new_version}.json")
            with open(config_path, 'w') as f:
                json.dump(vars(self.config), f)

            logger.info(f"Model saved as version {new_version} in {directory}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load(cls,
             directory: str,
             device: str = 'cpu',
             version: Optional[int] = None,
             **kwargs) -> 'TERLPolicy':
        """
        Load the model

        Args:
            directory: Model directory
            version: Specific version number to load, if None then load the latest version
            device: Device
            **kwargs: Other parameters

        Returns:
            TERLPolicy: Loaded model

        Raises:
            FileNotFoundError: When the specified version's model file does not exist
            ValueError: When there are no valid model files in the directory
        """
        try:
            # Get all version numbers
            versions = []
            for filename in os.listdir(directory):
                if filename.startswith("network_params_v") and filename.endswith(".pth"):
                    try:
                        ver = int(filename[len("network_params_v"):-4])
                        versions.append(ver)
                    except ValueError:
                        continue

            if not versions:
                raise ValueError(f"No valid model files found in {directory}")

            # Determine the version to load
            if version is None:
                version = max(versions)  # Load the latest version
            elif version not in versions:
                raise FileNotFoundError(f"Version {version} not found in {directory}")

            params_path = os.path.join(directory, f"network_params_v{version}.pth")
            config_path = os.path.join(directory, f"config_v{version}.json")

            # Load configuration
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = TERLAddTemporalConfig(**config_dict)
            else:
                logger.warning(f"Config file for version {version} not found, using default configuration")
                config = TERLAddTemporalConfig()

            # Update device and other parameters
            config.device = device
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # Create model and load parameters
            model = cls(config)
            model.load_state_dict(torch.load(params_path, map_location=device))
            model.to(device)

            logger.info(f"Successfully loaded model version {version} from {directory}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
