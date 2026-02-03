import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.optim as optim
import wandb
from torch.nn import functional as F

from policy.DQN import DqnPolicy, DqnConfig
from policy.TERL_model import TERLPolicy, TERLConfig
from policy.TERL_model_add_temporal import TERL_add_temporal, TERLAddTemporalConfig
from policy.ablation_mlp_with_target_selection import MlpWithTargetSelectPolicy, MlpWithTargetSelectConfig
from policy.ablation_transformer_without_target_selection import TransformerWithoutTargetSelectPolicy, TransformerWithoutTargetSelectConfig
from policy.IQN import IQNPolicy, IQNConfig
from policy.MEAN import MEANPolicy, MEANConfig
from policy.replay_buffer import ReplayBuffer


class Agent:
    """
    Reinforcement learning agent class, integrated with wandb monitoring.

    Attributes:
        device (str): The device to use ('cpu' or 'cuda')
        LR (float): Learning rate
        TAU (float): Soft update parameter
        GAMMA (float): Discount factor
        BUFFER_SIZE (int): Experience replay buffer size
        BATCH_SIZE (int): Experience replay batch size
        training (bool): Whether in training mode
        action_size (int): Size of the action space
        use_iqn (bool): Whether to use IQN policy network
        policy_local (nn.Module): Local policy network
        policy_target (nn.Module): Target policy network
        optimizer (optim.Optimizer): Optimizer
        memory (ReplayBuffer): Experience replay buffer
        wandb_project (str): wandb project name
        rewards_window (deque): Sliding window for rewards
        losses_window (deque): Sliding window for losses
        training_step (int): Training step counter
    """

    def __init__(self,
                 hidden_dimension=256,
                 num_heads=8,
                 num_layers=4,
                 action_size=9,
                 BATCH_SIZE=128,
                 BUFFER_SIZE=1_000_000,
                 LR=1e-4,
                 TAU=1.0,
                 GAMMA=0.99,
                 device="cpu",
                 seed=0,
                 training=True,
                 use_iqn=True,
                 model_name=None,
                 wandb_project="multi-agent-rl-iqn", ):
        """
        Initialize Agent instance.

        Args:
            hidden_dimension (int): Dimension of the hidden layer.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of layers in the network.
            action_size (int): Size of the action space.
            BATCH_SIZE (int): Batch size.
            BUFFER_SIZE (int): Buffer size.
            LR (float): Learning rate.
            TAU (float): Soft update parameter.
            GAMMA (float): Discount factor.
            device (str): The device to use ('cpu' or 'cuda').
            seed (int): Random seed.
            training (bool): Whether in training mode.
            use_iqn (bool): Whether to use IQN policy network.
            model_name (str): Name of the model to use.
            wandb_project (str): wandb project name.
        """
        self.device = device
        self.LR = LR
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.training = training
        self.action_size = action_size
        self.use_iqn = use_iqn
        self.model_name = model_name

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if training:
            # wandb related initialization
            self.wandb_project = wandb_project
            self.rewards_window = deque(maxlen=100)
            self.losses_window = deque(maxlen=100)
            self.training_step = 0

            assert self.model_name is not None, "model_name must be specified!"
            if use_iqn:
                # Policy network initialization
                if self.model_name == "TERL":
                    config = TERLConfig(
                        hidden_dim=hidden_dimension,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        action_size=action_size,
                        device=device,
                        seed=seed
                    )
                    self.policy_local = TERLPolicy(config=config)
                    self.policy_target = TERLPolicy(config=config)
                    self.policy_target.load_state_dict(self.policy_local.state_dict())
                elif self.model_name == "TERL_add_temporal":
                    config = TERLAddTemporalConfig(
                        hidden_dim=hidden_dimension,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        action_size=action_size,
                        device=device,
                        seed=seed
                    )
                    self.policy_local = TERL_add_temporal(config=config)
                    self.policy_target = TERL_add_temporal(config=config)
                    self.policy_target.load_state_dict(self.policy_local.state_dict())
                elif self.model_name == "MlpWithTargetSelect":
                    config = MlpWithTargetSelectConfig(
                        hidden_dim=hidden_dimension,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        action_size=action_size,
                        device=device,
                        seed=seed
                    )
                    self.policy_local = MlpWithTargetSelectPolicy(config=config)
                    self.policy_target = MlpWithTargetSelectPolicy(config=config)
                    self.policy_target.load_state_dict(self.policy_local.state_dict())
                elif self.model_name == "TransformerWithoutTargetSelect":
                    config = TransformerWithoutTargetSelectConfig(
                        hidden_dim=hidden_dimension,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        action_size=action_size,
                        device=device,
                        seed=seed
                    )
                    self.policy_local = TransformerWithoutTargetSelectPolicy(config=config)
                    self.policy_target = TransformerWithoutTargetSelectPolicy(config=config)
                    self.policy_target.load_state_dict(self.policy_local.state_dict())
                elif self.model_name == "IQN":
                    config = IQNConfig(
                        hidden_dim=hidden_dimension,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        action_size=action_size,
                        device=device,
                        seed=seed
                    )
                    self.policy_local = IQNPolicy(config=config)
                    self.policy_target = IQNPolicy(config=config)
                    self.policy_target.load_state_dict(self.policy_local.state_dict())
                elif self.model_name == "MEAN":
                    config = MEANConfig(
                        hidden_dim=hidden_dimension,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        action_size=action_size,
                        device=device,
                        seed=seed
                    )
                    self.policy_local = MEANPolicy(config=config)
                    self.policy_target = MEANPolicy(config=config)
                    self.policy_target.load_state_dict(self.policy_local.state_dict())
            else:
                # DQN policy network initialization
                if self.model_name == "DQN":
                    config = DqnConfig(
                        hidden_dim=hidden_dimension,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        action_size=action_size,
                        device=device,
                        seed=seed
                    )
                    self.policy_local = DqnPolicy(config=config)
                    self.policy_target = DqnPolicy(config=config)
                    self.policy_target.load_state_dict(self.policy_local.state_dict())

            self.optimizer = optim.Adam(self.policy_local.parameters(), lr=self.LR)

            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, 5, 8, 5)

    def act_dqn(self, state, eps=0.0, use_eval=True):
        """
        Select action based on DQN policy.

        Args:
            state (array_like): Current state.
            eps (float): Epsilon value for epsilon-greedy strategy.
            use_eval (bool): Whether to use evaluation mode.

        Returns:
            int: Selected action.
            np.ndarray: Action values.
        """
        state = self.convert_to_tensor(state, self.device)

        if use_eval:
            self.policy_local.eval()
        else:
            self.policy_local.train()

        with torch.no_grad():
            action_values = self.policy_local(state)
        self.policy_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        return action, action_values.cpu().data.numpy()

    def convert_to_tensor(self, data, device):
        """Recursively convert np.array in dictionary to torch.Tensor"""
        if isinstance(data, dict):
            # If it's a dictionary, recursively process each key-value pair
            return {key: self.convert_to_tensor(value, device) for key, value in data.items()}
        elif isinstance(data, np.ndarray):
            # If it's an np.array, convert to torch.Tensor and move to device
            return torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            # Other types remain unchanged
            return data

    def act(self, state, eps=0.0, cvar=1.0, use_eval=True):
        """
        Select action based on IQN policy.

        Args:
            state (array_like): Current state.
            eps (float): Epsilon value for epsilon-greedy strategy.
            cvar (float): CVaR parameter.
            use_eval (bool): Whether to use evaluation mode.

        Returns:
            int: Selected action.
            np.ndarray: Quantiles.
            np.ndarray: Tau values.
        """
        state = self.convert_to_tensor(state, self.device)

        if use_eval:
            self.policy_local.eval()
        else:
            self.policy_local.train()

        with torch.no_grad():
            quantiles, taus = self.policy_local(state, self.policy_local.K, cvar)
            action_values = quantiles.mean(dim=1)

        self.policy_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        return action, quantiles.cpu().data.numpy(), taus.cpu().data.numpy()

    def act_adaptive(self, state, eps=0.0):
        """
        Select action with adaptive CVaR value.

        Args:
            state (array_like): Current state.
            eps (float): Epsilon value for epsilon-greedy strategy.

        Returns:
            int: Selected action.
            np.ndarray: Quantiles.
            np.ndarray: Tau values.
            float: CVaR value.
        """
        cvar = self.adjust_cvar(state)
        action, quantiles, taus = self.act(state, eps, cvar)
        return action, quantiles, taus, cvar

    def adjust_cvar(self, state):
        """Adjust CVaR value based on state"""
        obstacles = state["obstacles"]
        pursuers = state["pursuers"]
        evaders = state["evaders"]

        closest_d = np.inf

        for entity in obstacles:
            if any(np.abs(entity[:2])) < 1e-3:
                # padding
                continue
            dist = np.linalg.norm(entity[:2]) - entity[2] - 0.8
            closest_d = min(closest_d, dist)

        for entity in pursuers:
            if any(np.abs(entity[:2])) < 1e-3:
                # padding
                continue
            dist = np.linalg.norm(entity[:2]) - 1.6
            closest_d = min(closest_d, dist)

        for entity in evaders:
            if any(np.abs(entity[:2])) < 1e-3:
                # padding
                continue
            dist = np.linalg.norm(entity[:2]) - 1.6
            closest_d = min(closest_d, dist)

        cvar = 1.0
        if closest_d < 10.0:
            cvar = closest_d / 10.0

        return cvar

    def train(self):
        """
        Select training method based on policy.

        Returns:
            float: Training loss.
        """
        if self.use_iqn:
            return self.train_IQN()
        else:
            return self.train_DQN()

    def train_IQN(self):
        """
        Train IQN model.

        Returns:
            float: Training loss.
        """
        states, actions, rewards, next_states, dones = self.memory.sample().values()
        actions = actions.unsqueeze(-1).long()
        rewards = rewards.unsqueeze(-1).float()
        dones = dones.unsqueeze(-1).float()

        self.optimizer.zero_grad()
        # Get max predicted Q values (for next states) from target model
        self.policy_target.reset_history_cache()
        Q_targets_next, _ = self.policy_target(next_states)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1)  # (batch_size, 1, N)

        # Compute Q targets for current states
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA * Q_targets_next * (1. - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        self.policy_local.reset_history_cache()
        Q_expected, taus = self.policy_local(states)
        Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, 8, 8), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1)  # keepdim=True if per weights get multiple
        loss = loss.mean()

        # Minimize the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 0.5)
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def train_DQN(self):
        """
        Train DQN model.

        Returns:
            float: Training loss.
        """
        states, actions, rewards, next_states, dones = self.memory.sample().values()
        actions = actions.unsqueeze(-1).long()
        rewards = rewards.unsqueeze(-1).float()
        dones = dones.unsqueeze(-1).float()

        self.optimizer.zero_grad()

        # Compute target values
        Q_targets_next = self.policy_target(next_states)
        Q_targets_next, _ = Q_targets_next.max(dim=1, keepdim=True)
        Q_targets = rewards + (1 - dones) * self.GAMMA * Q_targets_next

        # Compute expected values
        Q_expected = self.policy_local(states)
        Q_expected = Q_expected.gather(1, actions)

        # Compute Huber loss
        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # Minimize the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 0.5)
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def soft_update(self):
        """
        Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, local_param in zip(self.policy_target.parameters(), self.policy_local.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

    def save_latest_model(self, directory):
        """
        Save the latest model.

        Args:
            directory (Path): Directory to save the model.
        """
        self.policy_local.save(directory)

    def load_model(self, path, device="cpu"):
        """
        Load the model.

        Args:
            path (str): Path to the model file.
            device (str): Device type ('cpu' or 'cuda').
        """
        if self.use_iqn:
            if self.model_name == "TERL":
                self.policy_local = TERLPolicy.load(path, device)
            elif self.model_name == "MlpWithTargetSelect":
                self.policy_local = MlpWithTargetSelectPolicy.load(path, device)
            elif self.model_name == "TransformerWithoutTargetSelect":
                self.policy_local = TransformerWithoutTargetSelectPolicy.load(path, device)
            elif self.model_name == "IQN":
                self.policy_local = IQNPolicy.load(path, device)
            elif self.model_name == "MEAN":
                self.policy_local = MEANPolicy.load(path, device)
            else:
                raise ValueError(f"Unknown model name: {self.model_name}")
        else:
            if self.model_name == "DQN":
                self.policy_local = DqnPolicy.load(path, device)
            else:
                raise ValueError(f"Unknown model name: {self.model_name}")


@torch.jit.script
def calculate_huber_loss(td_errors: torch.Tensor, k: float = 1.0):
    """
    Calculate Huber loss.

    Args:
        td_errors (torch.Tensor): Temporal difference errors.
        k (float): Threshold parameter in Huber loss.

    Returns:
        torch.Tensor: Huber loss.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss
