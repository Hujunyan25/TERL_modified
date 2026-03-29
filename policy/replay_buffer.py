import torch
import numpy as np

class ReplayBuffer:
    def __init__(
        self, 
        buffer_size, 
        batch_size, 
        device, 
        max_pursuers, 
        max_evaders, 
        max_obstacles,
        swap_step=10000,  # 【关键】多少步后切换为随机采样
        alpha=0.6,        # PER优先级系数
        beta=0.4          # PER重要性采样系数
    ):
        """
        自适应经验回放池：前期优先采样，后期随机采样
        Args:
            swap_step: 训练多少步后，从优先采样切换为随机采样（默认5万步）
        """
        # ===================== 【原代码完全保留】 =====================
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.size = 0
        self.ptr = 0

        self.self_dim = 4
        self.pursuer_dim = 7
        self.evader_dim = 7
        self.obstacle_dim = 5

        self.self_state = torch.zeros((buffer_size, self.self_dim), device=device)
        self.next_self_state = torch.zeros((buffer_size, self.self_dim), device=device)

        self.pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.next_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)

        self.evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.next_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)

        self.obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.next_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)

        self.masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.bool, device=device)
        self.next_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.bool, device=device)

        self.types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.long, device=device)
        self.next_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.long, device=device)

        self.actions = torch.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)

        # ===================== 【新增：自适应 + 优先采样配置】 =====================
        self.swap_step = swap_step      # 切换步数阈值
        self.current_step = 0           # 当前训练步数
        self.alpha = alpha              # PER 优先级指数
        self.beta = beta                # PER 重要性权重
        self.use_priority = True        # 是否启用优先采样（自动切换）
        self.max_priority = 1.0         # 最大优先级
        
        # 优先级存储（和buffer同设备、同大小）
        self.priorities = torch.ones(buffer_size, device=device)

    def step(self):
        """【新增】每训练一步调用一次，自动切换采样模式"""
        self.current_step += 1
        if self.current_step >= self.swap_step:
            self.use_priority = False  # 关闭优先采样 → 随机采样

    def add(self, obs, action, reward, next_obs, done):
        """【原代码+少量修改】添加经验，自动赋予最大优先级"""
        idx = self.ptr

        # 原存储逻辑完全不变
        self.self_state[idx] = obs['self']
        self.pursuers[idx] = obs['pursuers']
        self.evaders[idx] = obs['evaders']
        self.obstacles[idx] = obs['obstacles']
        self.masks[idx] = obs['masks']
        self.types[idx] = obs['types']

        self.next_self_state[idx] = next_obs['self']
        self.next_pursuers[idx] = next_obs['pursuers']
        self.next_evaders[idx] = next_obs['evaders']
        self.next_obstacles[idx] = next_obs['obstacles']
        self.next_masks[idx] = next_obs['masks']
        self.next_types[idx] = next_obs['types']

        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        # 【新增】新经验默认赋予最大优先级
        self.priorities[idx] = self.max_priority

        # 指针更新
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        """【核心修改】自适应采样：前期优先，后期随机"""
        if self.size < self.batch_size:
            raise ValueError(f"Insufficient data in buffer, current size is {self.size}, need at least {self.batch_size} samples")

        # ===================== 自适应采样逻辑 =====================
        if self.use_priority:
            # 模式1：前期 → 优先经验采样
            priors = self.priorities[:self.size] ** self.alpha
            probs = priors / priors.sum()
            
            # torch 按概率采样
            indices = torch.multinomial(probs, self.batch_size, replacement=True)
            
            # 重要性权重（稳定训练）
            weights = (self.size * probs[indices]) ** (-self.beta)
            weights = weights / weights.max()
        else:
            # 模式2：后期 → 完全随机采样（和你原代码一致）
            indices = torch.randint(0, self.size, (self.batch_size,), device=self.device)
            weights = torch.ones_like(indices, device=self.device)  # 权重全1

        # ===================== 原batch返回逻辑（完全保留） =====================
        batch = {
            'observations': {
                'self': self.self_state.index_select(0, indices),
                'pursuers': self.pursuers.index_select(0, indices),
                'evaders': self.evaders.index_select(0, indices),
                'obstacles': self.obstacles.index_select(0, indices),
                'masks': self.masks.index_select(0, indices),
                'types': self.types.index_select(0, indices)
            },
            'actions': self.actions.index_select(0, indices),
            'rewards': self.rewards.index_select(0, indices),
            'next_observations': {
                'self': self.next_self_state.index_select(0, indices),
                'pursuers': self.next_pursuers.index_select(0, indices),
                'evaders': self.next_evaders.index_select(0, indices),
                'obstacles': self.next_obstacles.index_select(0, indices),
                'masks': self.next_masks.index_select(0, indices),
                'types': self.next_types.index_select(0, indices)
            },
            'dones': self.dones.index_select(0, indices),
            'indices': indices,      # 【新增】用于更新优先级
            'weights': weights       # 【新增】PER训练权重
        }
        return batch

    def update_priority(self, indices, td_errors):
        """【新增】更新优先级（训练后必须调用）"""
        if not self.use_priority:
            return
        # 用TD误差更新优先级
        td_errors = torch.abs(td_errors) + 1e-5
        self.priorities = self.priorities.clone()  # 新建副本，断开旧计算图
        self.priorities[indices] = td_errors       # 现在修改的是新副本，安全！
        # 更新最大优先级
        self.max_priority = max(self.max_priority, td_errors.max().item())

    def __len__(self):
        return self.size