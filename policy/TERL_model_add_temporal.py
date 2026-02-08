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


class TemporalFusionEncoder(nn.Module):
    def __init__(self, d_single, n_heads=4, d_model=256, num_layers=4, dropout=0.1):
        """
        仅对前五帧做时序融合的Transformer编码器模块
        :param d_single: 单目标基础维度（如7，ego/team/evader/obstacle最大值）
        :param n_heads: 多头注意力头数，要求d_model % n_heads == 0
        :param d_model: 编码器模型维度（推荐32/64）
        :param num_layers: TransformerEncoderLayer堆叠层数（推荐2）
        :param dropout: dropout率
        """
        super().__init__()
        self.d_single = d_single
        self.d_model = d_model

        # 1. 特征编码：基础特征→d_model，融合目标类型
        self.type_emb = nn.Embedding(4, d_model)  # 0=ego,1=team,2=evader,3=obstacle

        # 2. 时序位置编码：为前五帧（5帧）添加位置信息
        self.pos_emb = nn.Embedding(5, d_model)  # 仅5帧，精准适配

        # 3. Transformer编码器核心
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,  # 输入[B, T, d_model]
            norm_first=True,   # 先归一化后计算，提升训练稳定性
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 时序聚合+投影：还原为单目标基础维度
        self.proj = nn.Linear(d_model, d_single)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features_5former, masks_5former, types_5former):
        """
        前向传播：前五帧→单帧历史融合特征
        :param features_5former: 前五帧特征 [B,5,N,D]
        :param masks_5former: 前五帧掩码 [B * N,5]
        :param types_5former: 前五帧类型 [B * N,5]
        :return: hist_fusion: 历史融合特征 [B,1,N,D]
        """
        B, N, T, D = features_5former.shape  # T=5,B代表批次，T代表时序数量，N代表节点数量，D代表维度
        device = features_5former.device

        # 维度重排：[B*N,5,D]，每个目标独立做时序融合
        feature_reshaped = features_5former.reshape(B*N, T, D)
        # masks_reshaped = masks_5former.permute(0,2,1).reshape(B*N, T)
        # types_5former_unsqueeze = types_5former.unsqueeze(0).reshape(B, T, N)
        # types_reshaped = types_5former_unsqueeze.permute(0,2,1).reshape(B*N, T)
        # 特征编码+类型+位置嵌入
        x_type = self.type_emb(types_5former.long()) # shape：[B*N, T = 5, d_model]
        pos_idx = torch.arange(T, device=device).expand(B*N, T)
        x_pos = self.pos_emb(pos_idx) #shape: [B*N, T = 5, d_model]
        print("feature_reshaped.shape:", feature_reshaped.shape)
        print("x_type.shape:", x_type.shape)
        print("x_pos.shape:", x_pos.shape)
        x_enc = self.dropout(feature_reshaped + x_type + x_pos)
        x_enc = self.norm(x_enc)

        # 生成注意力掩码：-1e为padding，需要屏蔽，0为有效值
        # masks_5former = masks_5former.permute(1, 0)  # [B*N,5]
        # masks_5former_unsqueeze = masks_5former.unsqueeze(0).reshape(B, T, N)
        # masks_5former_unsqueeze = masks_5former_unsqueeze.permute(0,2,1).reshape(B*N, T)
        # masks_5former_after = masks_5former_unsqueeze
        if masks_5former.dtype == torch.bool:
            attn_mask = ~masks_5former.bool()
        else:
            attn_mask = (1 - masks_5former) * -1e9

        # Transformer编码器时序融合
        x_temporal = self.encoder(x_enc, src_key_padding_mask=attn_mask)  # [B*N,5,d_model]

        # 时序聚合（取最后一帧/均值，推荐取最后一帧）
        x_agg = x_temporal[:, -1, :]  # [B*N,d_model]

        # 投影还原维度+恢复形状
        history_fusion = self.proj(x_agg).reshape(B, N, D)  # [B,N,D]
        history_mask = masks_5former[:, -1].unsqueeze(0).reshape(B, -1)  # [B,N]
        history_type = types_5former[:, -1].unsqueeze(0).reshape(B, -1)  # [B,N]

        return history_fusion, history_mask, history_type


class FormerCurrentFusionAttention(nn.Module):
    def __init__(self, d_single, d_model=256, n_heads=4, dropout=0.1):
        """
        历史融合特征+当前帧 空间自注意力融合模块
        :param d_single: 单目标基础维度
        :param d_model: 空间融合模型维度
        :param n_heads: 多头注意力头数
        :param dropout: dropout率
        """
        super().__init__()
        self.d_single = d_single
        self.d_model = d_model

        # 1. 特征重编码：适配空间融合
        self.feat_linear = nn.Linear(d_single, d_model)
        self.type_emb = nn.Embedding(4, d_model)  # 再次融合类型信息

        # 2. 空间自注意力层
        self.space_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. 前馈网络+残差归一化
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 4. 输出投影：多目标/全局可选
        self.proj_multi = nn.Linear(d_model, d_single)  # 多目标输出（保留结构）
        self.proj_global = nn.Linear(d_model * 2, d_model)  # 全局输出（历史+当前拼接）

    def forward(self, x_concat, masks_concat, types_concat, return_global=False):
        """
        前向传播：历史+当前帧拼接特征→空间融合特征
        :param x_concat: 拼接特征 [B,2*N,D]（2=历史1帧+当前1帧）
        :param masks_concat: 拼接掩码 [B,2*N]
        :param types_concat: 拼接类型 [B,2*N]
        :param return_global: 是否返回全局单向量特征
        :return: 多目标空间融合特征 [B,2,N,D] / 全局特征 [B,d_model]
        """
        B, _ , D = x_concat.shape
        x_concat = x_concat.reshape(B, -1, 2, D)
        B, N, T_cat, D = x_concat.shape  # T_cat=2
        # 维度压缩：[B, N*T_cat, D]，将「历史+当前」作为不同目标参与空间交互
        x_squeeze = x_concat.permute(0,2,1,3).reshape(B * N, T_cat, D)

        # masks_concat张量变换
        masks_concat_unsqueeze = masks_concat.unsqueeze(0).reshape(B, N, T_cat)
        masks_concat_changed = masks_concat_unsqueeze.reshape(B * N, T_cat)
        # types_concat张量变换
        types_concat_unsqueeze = types_concat.unsqueeze(0).reshape(B, N, T_cat)
        types_concat_changed = types_concat_unsqueeze.reshape(B * N, T_cat)

        # 特征编码+类型嵌入
        x_feat = self.feat_linear(x_squeeze)  # [B*N, 2, d_model]
        x_type = self.type_emb(types_concat_changed.long())
        x_enc = self.dropout(x_feat + x_type)

        # 空间自注意力（屏蔽补0特征）
        if masks_concat_changed.dtype == torch.bool:
            attn_mask = ~masks_concat_changed.bool()
        else:
            attn_mask = (1 - masks_concat_changed) * -1e9
        x_attn, _ = self.space_attn(x_enc, x_enc, x_enc, key_padding_mask=attn_mask)
        x_attn = self.norm1(x_enc + self.dropout(x_attn))

        # 前馈网络+残差
        x_ffn = self.ffn(x_attn)
        x_ffn = self.norm2(x_attn + self.dropout(x_ffn))  # [B, N*2, d_model]

        # 输出分支
        if return_global:
            # 全局融合：单向量聚合所有信息
            global_feat = self.proj_global(x_ffn.reshape(B, -1))  # [B,d_model]
            return global_feat
        else:
            # 多目标融合：还原为[B,2,N,D]结构
            x_proj = self.proj_multi(x_ffn)  # [B, N*2, D]
            space_fusion = x_proj.reshape(B, N, T_cat, D).permute(0,2,1,3)  # [B,2,N,D]
            space_fusion = space_fusion.mean(dim = 1, keepdim = False)
            return space_fusion


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

        self.former_entity_encoders = nn.ModuleDict({
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

        #初始化历史帧时序融合网络
        self.temporal_fusion_encoder = TemporalFusionEncoder(d_single= self.hidden_dim)

        #初始化空间融合网络
        self.former_current_fusion_encoder = FormerCurrentFusionAttention(d_single=self.hidden_dim)



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


    def _motion_compensation_align(self, cached_observation: Dict[str, Dict]):
        """
        :param cached_observation: 历史帧的信息,key是[1,2,3,4,5], value是对应帧的信息字典
        多帧特征运动补偿+时序对齐：将t-5~t-1帧的邻居特征对齐到当前帧t的局部坐标系
        返回：对齐后的时序特征张量 + 统一有效掩码
        """
        # 解包缓存：时序有序 [t-5, t-4, t-3, t-2, t-1, t]
        N_pursuer = cached_observation[1]["pursuers"].shape[1]  # 邻居数量（以当前帧为准，保证多帧一致）
        N_evader = cached_observation[1]["evaders"].shape[1]  # 邻居数量（以当前帧为准，保证多帧一致）
        N_obstacle = cached_observation[1]["obstacles"].shape[1]  # 邻居数量（以当前帧为准，保证多帧一致）
        n_hist = TERLAddTemporalConfig.n_history  # 历史帧数：5帧
        
        # 步骤1：初始化对齐后的特征数组（存储6帧对齐后的特征）
        # 每帧邻居特征：拼接pos+vel → (N, 4)，6帧后为(6, N, 4)
        aligned_observations = cached_observation
        
        # 步骤3：历史帧（t-1~t-5）逐帧做运动补偿+对齐
        # 预提取所有帧的自身速度 (6, 2)
        self_velocities = []
        for _, observation in cached_observation.items():
            self_vel = observation["self"][:, :2] #提取前1～5帧的所有速度信息，shape：[Batch_size, 2]
            self_velocities.append(self_vel) #这里的self_vel存的是从前1帧到前5帧的所有自己速度信息，信息由近到远排序

        for i in range(n_hist):#每一帧中提取追捕者，逃避者和障碍物的信息
            #idx代表前idx帧
            idx = i + 1
            observation_former_idx = cached_observation[idx]
            # 提取τ帧的追捕者原始特征
            pursuers_position_former_idx = observation_former_idx["pursuers"][:, :, :2]  # (B,N,2)
            pursuers_velocity_former_idx = observation_former_idx["pursuers"][:, :, 2:4]  # (B,N,2)
            #提取τ帧的逃避者原始特征
            evader_position_former_idx = observation_former_idx["evaders"][:, :, :2] #(B,N,2)
            evader_velocity_former_idx = observation_former_idx["evaders"][:, :, 2:4] #(B,N,2)

            #提取τ帧的障碍物原始特征
            obstacle_former_idx = observation_former_idx["obstacles"][:, :, :2] #(B,N,2)
            
            # 计算从前几帧到t帧的累计位置补偿量 Δp_{t←τ}
            # 1. 提取前idx帧的自身速度 (前几帧, 2)
            self_velocities_t_to_current = self_velocities[:idx] #shape:[batch_size, 2]
            self_velocities_t_to_current_stack = torch.stack(self_velocities_t_to_current, dim=0) #shape:[帧数，batch_size, 2]
            # 2. pursuer和evader速度v_tau广播，与自身速度维度匹配
            pursuer_t_broadcast = torch.tile(pursuers_velocity_former_idx.unsqueeze(0), (len(self_velocities_t_to_current), 1, 1, 1)) #shape:[帧数，batch_size, N, 2]
            evader_t_broadcast = torch.tile(evader_velocity_former_idx.unsqueeze(0), (len(self_velocities_t_to_current), 1, 1, 1)) #shape:[帧数，batch_size, N, 2]
            
            # 3. 自身速度广播
            self_velocities_broadcast_pursuer = torch.tile(self_velocities_t_to_current_stack.unsqueeze(2), (1, 1, N_pursuer, 1)) #shape：[帧数，batch_size, N, 2]
            self_velocities_broadcast_evader = torch.tile(self_velocities_t_to_current_stack.unsqueeze(2), (1, 1, N_evader, 1)) #shape：[帧数，batch_size, N, 2]
            self_velocities_broadcast_obstacle = torch.tile(self_velocities_t_to_current_stack.unsqueeze(2), (1, 1, N_obstacle, 1)) #shape：[帧数，batch_size, N, 2]
            # 4. 计算每一步补偿量并累加 (N,2)
            delta_position_step_pursuer = (self_velocities_broadcast_pursuer - pursuer_t_broadcast) * 0.05  # (帧数,batch_size, N,2)
            delta_position_step_evader = (self_velocities_broadcast_evader - evader_t_broadcast) * 0.05  # (帧数,batch_size, N,2)
            delta_position_step_obstacles = self_velocities_broadcast_obstacle * 0.05 # (帧数,batch_size, N,2)
            delta_pursuer_total = torch.sum(delta_position_step_pursuer, dim=0) # (batch_size, N,2) 累计补偿量
            delta_evader_total = torch.sum(delta_position_step_evader, dim=0)  # (batch_size, N,2) 累计补偿量
            delta_obstacles_total = torch.sum(delta_position_step_obstacles, dim=0) # (batch_size, N,2) 累计补偿量
            

            # 5. 位置补偿：τ帧位置对齐到t帧坐标系
            pursuer_t_aligned = pursuers_position_former_idx + delta_pursuer_total  # (batch_size, N_pursuer,2)
            evader_t_aligned = evader_position_former_idx + delta_evader_total  # (batch_size, N_evader,2)
            obstacles_t_aligned = obstacle_former_idx + delta_obstacles_total  # (batch_size, N_obstacles,2)
            # 6. 速度无需补偿，保留原始值（反映邻居主动运动趋势）
            
            # 7. 赋值到对齐特征数组
            aligned_observations[idx]["pursuers"][:, :, :2] = pursuer_t_aligned  # (bach_size, N,2)
            aligned_observations[idx]["evaders"][:, :, :2] = evader_t_aligned #(batch_size, N, 2)
            aligned_observations[idx]["obstacles"][:, :, :2] = obstacles_t_aligned #(batch_size, N, 2)
        
        return aligned_observations 

    # --------------------------------------------------------------------------------
    
    def _validate_input(self, obs: Dict[str, Dict]) -> None:
        """Validate the format and dimensions of input data"""
        if not isinstance(obs, dict):
            raise ValueError("obs must be a dictionary")

        required_keys = {"current_observation", "observation_cache"}
        required_keys_in_current_observation = {'self', 'types', 'masks'}
        required_keys_in_cache_observation = {1, 2, 3, 4, 5}
        if not all(key in obs for key in required_keys):
            raise ValueError(f"obs must contain keys: {required_keys}")
        if not all(key in obs["current_observation"] for key in required_keys_in_current_observation):
            raise ValueError(f"current observation must contain keys: {required_keys_in_current_observation}")

        if not all(key in obs["observation_cache"] for key in required_keys_in_cache_observation):
            raise ValueError(f"cache observation must contain keys: {required_keys_in_cache_observation}")

        if obs["current_observation"]['self'].dim() != 2:
            raise ValueError("self features must be 2-dimensional [batch_size, feature_dim]")

        if obs["current_observation"]['self'].shape[1] != self.feature_dims['self']:
            raise ValueError(f"self features must have dimension {self.feature_dims['self']}")

    
    def encode_entities(self, obs: Dict[str, Dict]) -> torch.Tensor:
        """
        Encode entity features and process through Transformer

        Args:
            obs: Dictionary containing features of various entities

        Returns:
            torch.Tensor: Encoded features [batch_size, hidden_dim]
        """

        batch_size = obs["current_observation"]['self'].shape[0]
        current_observation = obs['current_observation'] #是当前观测到的信息，是一个字典
        current_encoded_features = []
        # Encode various entities,编码current_observation的那个信息
        for entity_type, encoder in self.entity_encoders.items():
            feature = current_observation[entity_type]
            if entity_type == 'self':
                encoded = encoder(feature).unsqueeze(1)
            else:
                encoded = encoder(feature)
            current_encoded_features.append(encoded)
        current_cat_feature = torch.cat(current_encoded_features, dim=1)

        #先进行运动补偿
        observation_after_compension = self._motion_compensation_align(obs["observation_cache"])
        # 生成初始的信息，以供输入
        cached_feature_dict = {}
        for key, observation in observation_after_compension.items(): #解析缓存中每一帧的内容, observation_after_compension: Dict[str, Dict]
            each_key_feature  = []
            for entity_type, encoder in self.former_entity_encoders.items():
                feature = observation[entity_type]
                if entity_type == 'self':
                    encoded = encoder(feature).unsqueeze(1)
                else:
                    encoded = encoder(feature)
                each_key_feature.append(encoded) #在列表中加入了每一帧的信息
            cached_feature_dict[key] = torch.cat(each_key_feature, dim = 1)
            #增添mask和type
            if "masks" not in cached_feature_dict:
                cached_feature_dict["masks"] = observation["masks"] #shape:[B, N]
            else:
                cached_feature_dict["masks"] = torch.cat([cached_feature_dict["masks"], observation["masks"]], dim = 0) #shape:[B* (帧数）, N]

            if "types" not in cached_feature_dict:
                cached_feature_dict["types"] = observation["types"] #shape:[B, N]
            else:
                cached_feature_dict["types"] = torch.cat([cached_feature_dict["types"], observation["types"]], dim = 0) #shape:[B* (帧数）, N ]

        cached_feature_dict["masks"] = cached_feature_dict["masks"].reshape(batch_size, 5, -1).permute(0, 2, 1).reshape(-1, 5)
        cached_feature_dict["types"] = cached_feature_dict["types"].reshape(batch_size, 5, -1).permute(0, 2, 1).reshape(-1, 5)
        #时序输入处理生成，得到cached_feature_dict字典，键有：1，2，3，4，5，masks，types
        #接下来把键为1，2，3，4，5的都整合到一起
        temporal_features = []
        for idx, features in cached_feature_dict.items():
            if idx in [1, 2, 3, 4, 5]:
                temporal_features.append(features)

        temporal_features = torch.cat(temporal_features, dim = 1).reshape(batch_size, -1, 5, self.hidden_dim)  #shape:[B, N * 5, D]

        # 对temporal_features进行时序融合
        fused_history_feat, history_mask, history_type = self.temporal_fusion_encoder(temporal_features, cached_feature_dict["masks"], cached_feature_dict["types"])

        concat_history_current_feature = torch.cat([fused_history_feat, current_cat_feature], dim = 1) #shape:[B, N * 2, D]
        concat_history_current_mask = torch.cat([history_mask, current_observation["masks"]], dim = 1) #shape:[B, N * 2]
        concat_history_current_type = torch.cat([history_type, current_observation["types"]], dim = 1) #shape:[B, N * 2]
        # 对融合后的历史特征和当前特征进行空间融合
        transformed = self.former_current_fusion_encoder(concat_history_current_feature, concat_history_current_mask, concat_history_current_type)
        #transformed的形状是
        # Get self feature
        self_feature = transformed[:, 0]  # [B, H]
        global_feature = torch.max(transformed, dim=1).values  # [B, H]

        enhanced_feature = torch.cat([self_feature, global_feature], dim=-1)  # [B, 2H]

        # Extract evader features and mask - uniformly handle batch and single samples
        evader_indices = (obs['current_observation']['types'] == 2)  # [B, N]

        # Get the number of positions with type==2 per batch
        num_evaders_per_batch = evader_indices.sum(dim=1)  # [B]
        max_evaders = num_evaders_per_batch.max().item()

        # Use masked_select and reshape to handle irregular selections
        flat_mask = obs['current_observation']['masks'].masked_select(evader_indices)
        flat_features = transformed.masked_select(
            evader_indices.unsqueeze(-1).expand(-1, -1, transformed.size(-1))
        )

        # Reshape into regular shape
        evader_mask = flat_mask.reshape(batch_size, max_evaders)  # [B, max_evaders]
        evader_features = flat_features.reshape(batch_size, max_evaders, -1)  # [B, max_evaders, H]

        # Apply target selection module
        enhanced_features, attention_weights = self.target_selection(
            enhanced_feature,
            evader_features,
            evader_mask
        )


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
                obs: Dict[str, Dict],
                num_tau: int = 8,
                cvar: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation

        Args:
            obs: Observation data dictionary（包含当前的current_observation和缓存信息cache_observation）
            num_tau: Number of tau samples
            cvar: CVaR parameter

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - quantiles: [batch_size, num_tau, action_size]
                - taus: [batch_size, num_tau, 1]
        """
        self._validate_input(obs)
        batch_size = obs["current_observation"]['self'].shape[0]


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
