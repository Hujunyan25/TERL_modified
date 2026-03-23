import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, max_pursuers, max_evaders, max_obstacles):
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

    def add(self, obs, action, reward, next_obs, done):
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

        # Update pointer and buffer size
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        """Sample a batch of data from the buffer"""
        if self.size < self.batch_size:
            raise ValueError(f"Insufficient data in buffer, current size is {self.size}, need at least {self.batch_size} samples")

        # Sample only from the filled part
        indices = torch.randint(0, self.size, (self.batch_size,), device=self.device)

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
        return batch

    def __len__(self):
        return self.size
