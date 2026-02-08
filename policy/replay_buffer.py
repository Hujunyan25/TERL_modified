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
        self.former_self_one_state = torch.zeros((buffer_size, self.self_dim), device=device)
        self.former_self_two_state = torch.zeros((buffer_size, self.self_dim), device=device)
        self.former_self_three_state = torch.zeros((buffer_size, self.self_dim), device=device)
        self.former_self_four_state = torch.zeros((buffer_size, self.self_dim), device=device)
        self.former_self_five_state = torch.zeros((buffer_size, self.self_dim), device=device)

        self.next_self_state = torch.zeros((buffer_size, self.self_dim), device=device)
        self.next_former_self_one_state = torch.zeros((buffer_size, self.self_dim), device=device)
        self.next_former_self_two_state = torch.zeros((buffer_size, self.self_dim), device=device)
        self.next_former_self_three_state = torch.zeros((buffer_size, self.self_dim), device=device)
        self.next_former_self_four_state = torch.zeros((buffer_size, self.self_dim), device=device)
        self.next_former_self_five_state = torch.zeros((buffer_size, self.self_dim), device=device)

        self.pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.former_one_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.former_two_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.former_three_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.former_four_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.former_five_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)

        self.next_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.next_former_one_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.next_former_two_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.next_former_three_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.next_former_four_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)
        self.next_former_five_pursuers = torch.zeros((buffer_size, max_pursuers, self.pursuer_dim), device=device)

        self.evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.former_one_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.former_two_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.former_three_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.former_four_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.former_five_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)


        self.next_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.next_former_one_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.next_former_two_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.next_former_three_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.next_former_four_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)
        self.next_former_five_evaders = torch.zeros((buffer_size, max_evaders, self.evader_dim), device=device)

        self.obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.former_one_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.former_two_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.former_three_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.former_four_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.former_five_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)


        self.next_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.next_former_one_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.next_former_two_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.next_former_three_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.next_former_four_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)
        self.next_former_five_obstacles = torch.zeros((buffer_size, max_obstacles, self.obstacle_dim), device=device)


        self.masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.bool,
                                 device=device)
        self.former_one_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                            dtype=torch.bool, device=device)
        self.former_two_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                            dtype=torch.bool, device=device)
        self.former_three_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                              dtype=torch.bool, device=device)
        self.former_four_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                             dtype=torch.bool, device=device)   
        self.former_five_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                             dtype=torch.bool, device=device)
        

        self.next_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.bool,
                                      device=device)
        self.next_former_one_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                            dtype=torch.bool, device=device)
        self.next_former_two_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                            dtype=torch.bool, device=device)
        self.next_former_three_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                              dtype=torch.bool, device=device)
        self.next_former_four_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                             dtype=torch.bool, device=device)   
        self.next_former_five_masks = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                             dtype=torch.bool, device=device)
        

        self.types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.long,
                                 device=device)
        self.former_one_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                            dtype=torch.long, device=device)
        self.former_two_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                            dtype=torch.long, device=device)    
        self.former_three_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                              dtype=torch.long, device=device)
        self.former_four_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                             dtype=torch.long, device=device)
        self.former_five_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                             dtype=torch.long, device=device)
        

        self.next_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles), dtype=torch.long,
                                      device=device)
        self.next_former_one_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                            dtype=torch.long, device=device)
        self.next_former_two_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                            dtype=torch.long, device=device)    
        self.next_former_three_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                              dtype=torch.long, device=device)
        self.next_former_four_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                             dtype=torch.long, device=device)
        self.next_former_five_types = torch.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles),
                                             dtype=torch.long, device=device)

        self.actions = torch.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)

    def add(self, obs, action, reward, next_obs, done):
        """Add data to the buffer"""
        idx = self.ptr

        # Store current observations
        self.self_state[idx] = obs['current_observation']['self']
        self.pursuers[idx] = obs['current_observation']['pursuers']
        self.evaders[idx] = obs['current_observation']['evaders']
        self.obstacles[idx] = obs['current_observation']['obstacles']
        self.masks[idx] = obs['current_observation']['masks']
        self.types[idx] = obs['current_observation']['types']

        # Store former observations
        self.former_self_one_state[idx] = obs["observation_cache"][1]['self']
        self.former_one_pursuers[idx] = obs["observation_cache"][1]['pursuers']
        self.former_one_evaders[idx] = obs["observation_cache"][1]['evaders']
        self.former_one_obstacles[idx] = obs["observation_cache"][1]['obstacles']
        self.former_one_masks[idx] = obs["observation_cache"][1]['masks']
        self.former_one_types[idx] = obs["observation_cache"][1]['types']

        self.former_self_two_state[idx] = obs["observation_cache"][2]['self']
        self.former_two_pursuers[idx] = obs["observation_cache"][2]['pursuers']
        self.former_two_evaders[idx] = obs["observation_cache"][2]['evaders']
        self.former_two_obstacles[idx] = obs["observation_cache"][2]['obstacles']
        self.former_two_masks[idx] = obs["observation_cache"][2]['masks']
        self.former_two_types[idx] = obs["observation_cache"][2]['types']
        self.former_three_types[idx] = obs["observation_cache"][3]['types']

        self.former_self_three_state[idx] = obs["observation_cache"][3]['self']
        self.former_three_pursuers[idx] = obs["observation_cache"][3]['pursuers']
        self.former_three_evaders[idx] = obs["observation_cache"][3]['evaders']
        self.former_three_obstacles[idx] = obs["observation_cache"][3]['obstacles']
        self.former_three_masks[idx] = obs["observation_cache"][3]['masks']
        self.former_three_types[idx] = obs["observation_cache"][3]['types']

        self.former_self_four_state[idx] = obs["observation_cache"][4]['self']
        self.former_four_pursuers[idx] = obs["observation_cache"][4]['pursuers']
        self.former_four_evaders[idx] = obs["observation_cache"][4]['evaders']
        self.former_four_obstacles[idx] = obs["observation_cache"][4]['obstacles']
        self.former_four_masks[idx] = obs["observation_cache"][4]['masks']
        self.former_four_types[idx] = obs["observation_cache"][4]['types']

        self.former_self_five_state[idx] = obs["observation_cache"][5]['self']
        self.former_five_pursuers[idx] = obs["observation_cache"][5]['pursuers']
        self.former_five_evaders[idx] = obs["observation_cache"][5]['evaders']
        self.former_five_obstacles[idx] = obs["observation_cache"][5]['obstacles']
        self.former_five_masks[idx] = obs["observation_cache"][5]['masks']
        self.former_five_types[idx] = obs["observation_cache"][5]['types']



        # Store next observations

        # Store next observations
        self.next_self_state[idx] = obs['current_observation']['self']
        self.next_pursuers[idx] = obs['current_observation']['pursuers']
        self.next_evaders[idx] = obs['current_observation']['evaders']
        self.next_obstacles[idx] = obs['current_observation']['obstacles']
        self.next_masks[idx] = obs['current_observation']['masks']
        self.next_types[idx] = obs['current_observation']['types']

        # Store former observations
        self.next_former_self_one_state[idx] = obs["observation_cache"][1]['self']
        self.next_former_one_pursuers[idx] = obs["observation_cache"][1]['pursuers']
        self.next_former_one_evaders[idx] = obs["observation_cache"][1]['evaders']
        self.next_former_one_obstacles[idx] = obs["observation_cache"][1]['obstacles']
        self.next_former_one_masks[idx] = obs["observation_cache"][1]['masks']
        self.next_former_one_types[idx] = obs["observation_cache"][1]['types']

        self.next_former_self_two_state[idx] = obs["observation_cache"][2]['self']
        self.next_former_two_pursuers[idx] = obs["observation_cache"][2]['pursuers']
        self.next_former_two_evaders[idx] = obs["observation_cache"][2]['evaders']
        self.next_former_two_obstacles[idx] = obs["observation_cache"][2]['obstacles']
        self.next_former_two_masks[idx] = obs["observation_cache"][2]['masks']
        self.next_former_two_types[idx] = obs["observation_cache"][2]['types']

        self.next_former_self_three_state[idx] = obs["observation_cache"][3]['self']
        self.next_former_three_pursuers[idx] = obs["observation_cache"][3]['pursuers']
        self.next_former_three_evaders[idx] = obs["observation_cache"][3]['evaders']
        self.next_former_three_obstacles[idx] = obs["observation_cache"][3]['obstacles']
        self.next_former_three_masks[idx] = obs["observation_cache"][3]['masks']
        self.next_former_three_types[idx] = obs["observation_cache"][3]['types']

        self.next_former_self_four_state[idx] = obs["observation_cache"][4]['self']
        self.next_former_four_pursuers[idx] = obs["observation_cache"][4]['pursuers']
        self.next_former_four_evaders[idx] = obs["observation_cache"][4]['evaders']
        self.next_former_four_obstacles[idx] = obs["observation_cache"][4]['obstacles']
        self.next_former_four_masks[idx] = obs["observation_cache"][4]['masks']
        self.next_former_four_types[idx] = obs["observation_cache"][4]['types']

        self.next_former_self_five_state[idx] = obs["observation_cache"][5]['self']
        self.next_former_five_pursuers[idx] = obs["observation_cache"][5]['pursuers']
        self.next_former_five_evaders[idx] = obs["observation_cache"][5]['evaders']
        self.next_former_five_obstacles[idx] = obs["observation_cache"][5]['obstacles']
        self.next_former_five_masks[idx] = obs["observation_cache"][5]['masks']
        self.next_former_five_types[idx] = obs["observation_cache"][5]['types']


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
                "current_observation": {
                    'self': self.self_state.index_select(0, indices),
                    'pursuers': self.pursuers.index_select(0, indices),
                    'evaders': self.evaders.index_select(0, indices),
                    'obstacles': self.obstacles.index_select(0, indices),
                    'masks': self.masks.index_select(0, indices),
                    'types': self.types.index_select(0, indices)     
                },
                "observation_cache": {
                    1: {
                        'self': self.former_self_one_state.index_select(0, indices),
                        'pursuers': self.former_one_pursuers.index_select(0, indices),
                        'evaders': self.former_one_evaders.index_select(0, indices),
                        'obstacles': self.former_one_obstacles.index_select(0, indices),
                        'masks': self.former_one_masks.index_select(0, indices),
                        'types': self.former_one_types.index_select(0, indices)
                    },
                    2: {
                        'self': self.former_self_two_state.index_select(0, indices),
                        'pursuers': self.former_two_pursuers.index_select(0, indices),
                        'evaders': self.former_two_evaders.index_select(0, indices),
                        'obstacles': self.former_two_obstacles.index_select(0, indices),
                        'masks': self.former_two_masks.index_select(0, indices),
                        'types': self.former_two_types.index_select(0, indices)
                    },
                    3: {
                        'self': self.former_self_three_state.index_select(0, indices),
                        'pursuers': self.former_three_pursuers.index_select(0, indices),
                        'evaders': self.former_three_evaders.index_select(0, indices),
                        'obstacles': self.former_three_obstacles.index_select(0, indices),
                        'masks': self.former_three_masks.index_select(0, indices),
                        'types': self.former_three_types.index_select(0, indices)
                    },
                    4: {
                        'self': self.former_self_four_state.index_select(0, indices),
                        'pursuers': self.former_four_pursuers.index_select(0, indices),
                        'evaders': self.former_four_evaders.index_select(0, indices),
                        'obstacles': self.former_four_obstacles.index_select(0, indices),
                        'masks': self.former_four_masks.index_select(0, indices),
                        'types': self.former_four_types.index_select(0, indices)
                    },
                    5: {
                        'self': self.former_self_five_state.index_select(0, indices),
                        'pursuers': self.former_five_pursuers.index_select(0, indices),
                        'evaders': self.former_five_evaders.index_select(0, indices),   
                        'obstacles': self.former_five_obstacles.index_select(0, indices),
                        'masks': self.former_five_masks.index_select(0, indices),
                        'types': self.former_five_types.index_select(0, indices)
                    }
                }
            },
            'actions': self.actions.index_select(0, indices),
            'rewards': self.rewards.index_select(0, indices),
            'next_observations': {
                "current_observation": {
                    'self': self.next_self_state.index_select(0, indices),
                    'pursuers': self.next_pursuers.index_select(0, indices),
                    'evaders': self.next_evaders.index_select(0, indices),
                    'obstacles': self.next_obstacles.index_select(0, indices),
                    'masks': self.next_masks.index_select(0, indices),
                    'types': self.next_types.index_select(0, indices)
                },
                "observation_cache": {
                    1: {
                        'self': self.next_former_self_one_state.index_select(0, indices),
                        'pursuers': self.next_former_one_pursuers.index_select(0, indices),
                        'evaders': self.next_former_one_evaders.index_select(0, indices),
                        'obstacles': self.next_former_one_obstacles.index_select(0, indices),
                        'masks': self.next_former_one_masks.index_select(0, indices),
                        'types': self.next_former_one_types.index_select(0, indices)
                    },
                    2: {
                        'self': self.next_former_self_two_state.index_select(0, indices),
                        'pursuers': self.next_former_two_pursuers.index_select(0, indices),
                        'evaders': self.next_former_two_evaders.index_select(0, indices),
                        'obstacles': self.next_former_two_obstacles.index_select(0, indices),
                        'masks': self.next_former_two_masks.index_select(0, indices),
                        'types': self.next_former_two_types.index_select(0, indices)
                    },
                    3: {
                        'self': self.next_former_self_three_state.index_select(0, indices),
                        'pursuers': self.next_former_three_pursuers.index_select(0, indices),
                        'evaders': self.next_former_three_evaders.index_select(0, indices), 
                        'obstacles': self.next_former_three_obstacles.index_select(0, indices),
                        'masks': self.next_former_three_masks.index_select(0, indices),
                        'types': self.next_former_three_types.index_select(0, indices)
                    },
                    4: {
                        'self': self.next_former_self_four_state.index_select(0, indices),
                        'pursuers': self.next_former_four_pursuers.index_select(0, indices),
                        'evaders': self.next_former_four_evaders.index_select(0, indices),
                        'obstacles': self.next_former_four_obstacles.index_select(0, indices),
                        'masks': self.next_former_four_masks.index_select(0, indices),
                        'types': self.next_former_four_types.index_select(0, indices)
                    },
                    5: {
                        'self': self.next_former_self_five_state.index_select(0, indices),
                        'pursuers': self.next_former_five_pursuers.index_select(0, indices),
                        'evaders': self.next_former_five_evaders.index_select(0, indices),
                        'obstacles': self.next_former_five_obstacles.index_select(0, indices),
                        'masks': self.next_former_five_masks.index_select(0, indices),
                        'types': self.next_former_five_types.index_select(0, indices)
                    }
                }
            },
            'dones': self.dones.index_select(0, indices)
        }
        return batch

    def __len__(self):
        return self.size
