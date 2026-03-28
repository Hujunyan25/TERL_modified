import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, max_pursuers, max_evaders, max_obstacles,
                 alpha = 0.7, beta = 0.4, beta_increment = 0.002, epsilon = 1e-6):
        """
        Initialize the replay buffer
        Args:
            buffer_size (int): Maximum capacity of the buffer
            batch_size (int): Batch size for each sampling
            device (str): 'cpu' or 'cuda'
        """
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.size = 0
        self.ptr = 0

        # Define feature dimensions for various entity types
        self.self_dim = 4
        self.pursuer_dim = 7
        self.evader_dim = 7
        self.obstacle_dim = 5

        # Initialize storage space
        self.self_state = torch.zeros((buffer_size, self.self_dim), device=device)
        self.next_self_state = torch.zeros((buffer_size, self.self_dim), device=device)

        self.pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.next_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)

        self.evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.next_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)

        self.obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.next_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)

        self.masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.bool,
                                 device=device)
        self.next_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.bool,
                                      device=device)

        self.types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.long,
                                 device=device)
        self.next_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.long,
                                      device=device)

        self.actions = torch.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)

        # =========PER新增：存储每个样本的优先级===========
        self.priorities = torch.zeros(buffer_size, device = device)

        #PER核心参数
        self.alpha = alpha #优先级影响因子
        self.beta = beta #重要性采样因子
        self.beta_increment = beta_increment
        self.epsilon = epsilon

    def add(self, obs, action, reward, next_obs, done, td_error = None):
        """Add data to the buffer"""
        idx = self.ptr

        # Store current observations
        self.self_state[idx] = obs['self']
        self.pursuers[idx] = obs['pursuers']
        self.evaders[idx] = obs['evaders']
        self.obstacles[idx] = obs['obstacles']
        self.masks[idx] = obs['masks']
        self.types[idx] = obs['types']

        # Store next observations
        self.next_self_state[idx] = next_obs['self']
        self.next_pursuers[idx] = next_obs['pursuers']
        self.next_evaders[idx] = next_obs['evaders']
        self.next_obstacles[idx] = next_obs['obstacles']
        self.next_masks[idx] = next_obs['masks']
        self.next_types[idx] = next_obs['types']

        # Store actions, rewards, and done flags
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        #PER新增
        if td_error is None:
            #没有td_error的时候，设置当前为最大优先级
            max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
            self.priorities[idx] = max_priority
        else:
            #用TD误差计算优先级：优先级 = (|TD误差| + ε)^α，其中ε是一个小常数，防止优先级为0，α控制优先级的程度
            print(f"TD误差为：{td_error}")
            self.priorities[idx] = (torch.abs(td_error) + self.epsilon) ** self.alpha
        
        # Update pointer and buffer size
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        """Sample a batch of data from the buffer"""
        if self.size < self.batch_size:
            raise ValueError(f"Insufficient data in buffer, current size is {self.size}, need at least {self.batch_size} samples")

        # ===仅仅取已填充部分的优先级
        priors = self.priorities[:self.size]
        #采样的概率
        probs = priors / priors.sum()

        #按照概率分布采样索引
        indices = torch.multinomial(probs, self.batch_size, replacement=True)
        
        # ===计算重要性采样权重=====
        self.beta = min(1.0, self.beta + self.beta_increment) 
        #计算权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max() #归一化权重
        weights = weights.to(self.device)

        # Sample only from the filled part
        # indices = torch.randint(0, self.size, (self.batch_size,), device=self.device)

        # Sample batch data
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
            'dones': self.dones.index_select(0, indices)
        }
        return batch, weights, indices
    

    def update_priorities(self, indices, td_errors):
        '''
        更新训练之后的优先级

        :param indices: 采样时返回的索引列表
        :param td_errors: 新计算的误差
        '''
        with torch.no_grad():
            priorities = (torch.abs(td_errors) + self.epsilon) ** self.alpha
            #重新计算优先级并更新
            self.priorities[indices] = priorities

    def __len__(self):
        return self.size