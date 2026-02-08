import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
import plotly.graph_objects as go

from environment.env import MarineEnv
from policy.agent import Agent
from thirdparty.APF import ApfAgent
from utils import logger as logger
from config_manager import ConfigManager


class Trainer:
    """
    Trainer class for training and evaluating reinforcement learning agents in a specified environment.

    Attributes:
        train_env (MarineEnv): Training environment.
        eval_env (MarineEnv): Evaluation environment.
        pursuer_agent (Agent): Pursuer agent.
        evader_agent (ApfAgent): Evader agent.
        eval_config (list): List of evaluation configurations.
        UPDATE_EVERY (int): Number of steps between model updates.
        learning_starts (int): Number of steps before learning begins.
        target_update_interval (int): Interval for updating the target network.
        exploration_fraction (float): Fraction of total steps for exploration decay.
        initial_eps (float): Initial exploration rate.
        final_eps (float): Final exploration rate.
        episode_max_length (int): Maximum length of each episode.
        current_timestep (int): Current timestep.
        learning_timestep (int): Learning timestep counter.
        eval_timesteps (list): List of evaluation timesteps.
        eval_actions (list): List of evaluation actions.
        eval_trajectories (list): List of evaluation trajectories.
        eval_rewards (list): List of evaluation rewards.
        eval_successes (list): List of evaluation successes.
        eval_times (list): List of evaluation times.
        eval_energies (list): List of evaluation energies.
        eval_obs (list): List of evaluation observations.
        eval_pursuers (list): List of evaluation pursuers.
        eval_evaders (list): List of evaluation evaders.
    """

    def __init__(self,
                 train_env: MarineEnv,
                 eval_env: MarineEnv,
                 eval_schedule: dict,
                 pursuer_agent: Agent = None,
                 evader_agent: ApfAgent = None,
                 episode_max_length: int = 3000,
                 update_every: int = 4,
                 learning_starts: int = 3000,
                 target_update_interval: int = 10000,
                 exploration_fraction: float = 0.25,
                 initial_eps: float = 0.6,
                 final_eps: float = 0.05,
                 device: str = 'cpu'):
        """
        Initialize Trainer instance.

        Args:
            train_env (MarineEnv): Training environment.
            eval_env (MarineEnv): Evaluation environment.
            eval_schedule (dict): Evaluation schedule.
            pursuer_agent (Agent, optional): Pursuer agent. Defaults to None.
            evader_agent (ApfAgent, optional): Evader agent. Defaults to None.
            episode_max_length (int, optional): Maximum length of each episode. Defaults to 3000.
            update_every (int, optional): Number of steps between model updates. Defaults to 4.
            learning_starts (int, optional): Number of steps before learning begins. Defaults to 3000.
            target_update_interval (int, optional): Interval for updating the target network. Defaults to 10000.
            exploration_fraction (float, optional): Fraction of total steps for exploration decay. Defaults to 0.25.
            initial_eps (float, optional): Initial exploration rate. Defaults to 0.6.
            final_eps (float, optional): Final exploration rate. Defaults to 0.05.
            device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.device = device
        self.train_env = train_env
        self.eval_env = eval_env
        self.pursuer_agent = pursuer_agent
        self.evader_agent = evader_agent
        self.eval_config = []
        self.create_eval_configs(eval_schedule)

        self.trainer_config = ConfigManager().get_instance()
        self.UPDATE_EVERY = self.trainer_config.get("training.update_every", update_every)
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.episode_max_length = self.trainer_config.get("training.episode_max_length", episode_max_length)

        # Current timestep
        self.current_timestep = 0

        # Learning timestep (starts counting after learning_starts)
        self.learning_timestep = 0

        # Evaluation data
        self.eval_timesteps = []
        self.eval_actions = []
        # self.eval_trajectories = []
        self.eval_rewards = []
        self.eval_successes = []
        self.eval_times = []
        self.eval_energies = []
        self.eval_obs = []
        self.eval_pursuers = []
        self.eval_evaders = []

        self.trajectory_buffer = []

    def create_eval_configs(self, eval_schedule: dict):
        """
        Create evaluation configurations.

        Args:
            eval_schedule (dict): Evaluation schedule.
        """
        self.eval_config.clear()

        count = 0
        for i, num_episode in enumerate(eval_schedule["num_episodes"]):
            for _ in range(num_episode):
                self.eval_env.num_pursuers = eval_schedule["num_pursuers"][i]
                self.eval_env.num_evaders = eval_schedule["num_evaders"][i]
                self.eval_env.num_cores = eval_schedule["num_cores"][i]
                self.eval_env.num_obs = eval_schedule["num_obstacles"][i]
                self.eval_env.min_pursuer_evader_init_dis = eval_schedule["min_pursuer_evader_init_dis"][i]

                self.eval_env.reset()

                # Save evaluation configuration
                self.eval_config.append(self.eval_env.episode_data())
                count += 1
                # logger.info(f"Process {os.getpid()} Success in generating pursuer evader eval_configs "
                #             f"at num_episodes {i}: num_episode {_}🤗!")

    def save_eval_config(self, directory: Path):
        """
        Save evaluation configurations.

        Args:
            directory (Path): Directory to save configurations.
        """
        file = os.path.join(directory, "eval_configs.json")
        with open(file, "w+") as f:
            json.dump(self.eval_config, f)

    def convert_to_tensor(self, data, device):
        """
        Recursively convert np.array in a dictionary to torch.Tensor and move to the specified device.
        """
        if isinstance(data, dict):
            # If it's a dictionary, recursively process each key-value pair
            return {key: self.convert_to_tensor(value, device) for key, value in data.items()}
        elif isinstance(data, np.ndarray):
            # If it's an np.array, convert to torch.Tensor and move to device
            return torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            # Other types remain unchanged
            return data

    def learn(self,
              total_timesteps: int,
              eval_freq: int,
              eval_log_path: Path,
              verbose: bool = True):
        """
        Start the learning process.

        Args:
            total_timesteps (int): Total number of timesteps.
            eval_freq (int): Evaluation frequency.
            eval_log_path (Path): Path for evaluation logs.
            verbose (bool, optional): Whether to print detailed information. Defaults to True.
        """
        states_collision, _ = self.train_env.reset()
        states, _ = states_collision
        evader_states, _ = self.train_env.get_evaders_observation()

        # # Sample CVaR value from (0.0,1.0)
        # cvar = 1 - np.random.uniform(0.0, 1.0)

        # Current episode
        ep_rewards = np.zeros(len(self.train_env.pursuers))
        ep_deactivated_t = [-1] * len(self.train_env.pursuers)
        ep_length = 0
        ep_num = 0

        while self.current_timestep <= total_timesteps:
            eps = self.linear_eps(total_timesteps)

            states = [self.convert_to_tensor(state, self.device) for state in states]
            # Get actions for all robots
            actions = []
            for i, rob in enumerate(self.train_env.pursuers):
                if rob.deactivated:
                    actions.append(None)
                    continue

                if self.pursuer_agent.use_iqn:
                    action, _, _ = self.pursuer_agent.act(states[i], eps)
                else:
                    action, _ = self.pursuer_agent.act_dqn(states[i], eps)
                actions.append(action)

            evader_actions = []
            for j, evader in enumerate(self.train_env.evaders):
                if evader.deactivated:
                    evader_actions.append(None)
                    continue

                evader_action = self.evader_agent.act(evader_states[j])
                evader_actions.append(evader_action)

            # self.log_positions(ep_num)

            # Execute actions and get next states
            next_states, rewards, dones, infos = self.train_env.step((actions, evader_actions))
            next_evader_states, _ = self.train_env.get_evaders_observation()
            next_states = [self.convert_to_tensor(state, self.device) for state in next_states]

            # Save experience to replay buffer
            for i, pursuer in enumerate(self.train_env.pursuers):
                if pursuer.deactivated:
                    continue

                ep_rewards[i] += self.pursuer_agent.GAMMA ** ep_length * rewards[i]
                if self.pursuer_agent.training:
                    self.pursuer_agent.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

                if pursuer.collision:
                    pursuer.deactivated = True
                    ep_deactivated_t[i] = ep_length

            end_episode = (ep_length >= self.episode_max_length) or self.train_env.check_all_evader_is_captured() or \
                          len([pur for pur in self.train_env.pursuers if not pur.deactivated]) < 3  # Fewer than 3 active pursuers in the environment

            # Learn, update, and evaluate model
            if self.current_timestep >= self.learning_starts:
                for agent in [self.pursuer_agent]:
                    if agent is None or not agent.training:
                        continue

                    # Learn every UPDATE_EVERY steps
                    if self.current_timestep % self.UPDATE_EVERY == 0 and agent.memory.size > agent.BATCH_SIZE:
                        agent.train()

                    # Update target model every target_update_interval steps
                    if self.current_timestep % self.target_update_interval == 0:
                        agent.soft_update()

                # Evaluate every eval_freq steps
                if self.current_timestep == self.learning_starts or self.current_timestep % eval_freq == 0:
                    # self.evaluation()
                    # self.save_evaluation(eval_log_path)
                    for agent in [self.pursuer_agent]:
                        if agent is None or not agent.training:
                            continue
                        # Save the latest models
                        agent.save_latest_model(eval_log_path)

            if end_episode:
                ep_num += 1

                # Log training information
                self.log_training_episode(
                    ep_length=ep_length,
                    ep_num=ep_num,
                    eps=eps,
                    ep_rewards=ep_rewards,
                    infos=infos
                )

                # self.log_episode_trajectory(ep_num)

                if verbose:
                    logger.info(f"Process {os.getpid()} ======== Episode Info ========")
                    logger.info(f"Process {os.getpid()} - current ep_length: {ep_length}")
                    logger.info(f"Process {os.getpid()} - current ep_num: {ep_num}")
                    logger.info(f"Process {os.getpid()} - current exploration rate: {eps}")
                    logger.info(f"Process {os.getpid()} - current timesteps: {self.current_timestep}")
                    logger.info(f"Process {os.getpid()} - total timesteps: {total_timesteps}")
                    logger.info(f"Process {os.getpid()} ======== Episode Info ========")
                    logger.info(f"Process {os.getpid()} ======== Robots Info ========")
                    for i, rob in enumerate(self.train_env.pursuers):
                        info = infos[i]["state"]
                        if info == "deactivated after collision" or info == "deactivated after reaching goal":
                            logger.info(
                                f"Process {os.getpid()} - Robot {i} ep reward: {ep_rewards[i]:.2f}, {info} at step {ep_deactivated_t[i]}")
                        else:
                            logger.info(f"Process {os.getpid()} - Robot {i} ep reward: {ep_rewards[i]:.2f}, {info}")
                    logger.info(f"Process {os.getpid()} ======== Robots Info ========")

                states_collision, _ = self.train_env.reset()
                states, _ = states_collision
                evader_states, _ = self.train_env.get_evaders_observation()

                ep_rewards = np.zeros(len(self.train_env.pursuers))
                ep_deactivated_t = [-1] * len(self.train_env.pursuers)
                ep_length = 0
            else:
                states = next_states
                evader_states = next_evader_states
                ep_length += 1

            self.current_timestep += 1
            if self.current_timestep > total_timesteps:
                logger.info(f"Process {os.getpid()} Training completed!")

    def log_training_episode(self, ep_length, ep_num, eps, ep_rewards, infos):
        """Log information for each episode during training"""
        # Basic metrics
        metrics = {
            "episode/length": ep_length,
            "episode/number": ep_num,
            "episode/exploration_rate": eps,
            "episode/mean_reward": np.mean(ep_rewards),
            "episode/min_reward": np.min(ep_rewards),
            "episode/max_reward": np.max(ep_rewards),
            "episode/total_timesteps": self.current_timestep,
        }

        # Pursuer state information - Use dictionary list instead of Table
        for i, (pursuer, reward) in enumerate(zip(self.train_env.pursuers, ep_rewards)):
            # Add individual metrics for each pursuer
            metrics.update({
                f"pursuer_{i}/state": infos[i]["state"],
                f"pursuer_{i}/reward": reward,
                f"pursuer_{i}/deactivated": pursuer.deactivated,
                f"pursuer_{i}/collision": getattr(pursuer, 'collision', False),
                f"pursuer_{i}/position_x": float(getattr(pursuer, 'x', 0.0)),
                f"pursuer_{i}/position_y": float(getattr(pursuer, 'y', 0.0))
            })

        # Episode summary information
        metrics.update({
            "episode_summary/active_pursuers": len([p for p in self.train_env.pursuers if not p.deactivated]),
            "episode_summary/all_evaders_captured": self.train_env.check_all_evader_is_captured()
        })

        # Log all metrics at once
        wandb.log(metrics, step=self.current_timestep)

    def linear_eps(self, total_timesteps: int) -> float:
        """
        Linear exploration rate decay function.

        Args:
            total_timesteps (int): Total number of timesteps.

        Returns:
            float: Exploration rate at the current timestep.
        """
        progress = self.current_timestep / total_timesteps
        if progress < self.exploration_fraction:
            r = progress / self.exploration_fraction
            return self.initial_eps + r * (self.final_eps - self.initial_eps)
        else:
            return self.final_eps

    def evaluation(self):
        """
        Evaluate the agent's performance and record the time for each operation.
        """
        # Initialize data storage
        # actions_data = []
        # trajectories_data = []
        rewards_data = []
        successes_data = []
        times_data = []
        energies_data = []
        # obs_data = []
        # pursuers_data = []
        # evaders_data = []

        for idx, config in enumerate(self.eval_config):

            logger.info(f"Process {os.getpid()} - Evaluating episode {idx}")

            # Record environment reset time
            state, _ = self.eval_env.reset_with_eval_config(config)
            evader_states, _ = self.eval_env.get_evaders_observation()

            # obs = [[copy.deepcopy(rob.perception.observed_obstacles)] for rob in self.eval_env.pursuers]
            # pursuers = [[copy.deepcopy(rob.perception.observed_pursuers)] for rob in self.eval_env.pursuers]
            # evaders = [[copy.deepcopy(rob.perception.observed_evaders)] for rob in self.eval_env.pursuers]

            pursuer_num = len(self.eval_env.pursuers)

            rewards = [0.0] * pursuer_num
            times = [0.0] * pursuer_num
            energies = [0.0] * pursuer_num
            end_episode = False
            length = 0

            while not end_episode:

                action = []
                for i, rob in enumerate(self.eval_env.pursuers):
                    if rob.deactivated:
                        action.append(None)
                        continue

                    if self.pursuer_agent.use_iqn:
                        a, _, _ = self.pursuer_agent.act(state[i])
                    else:
                        a, _ = self.pursuer_agent.act_dqn(state[i])

                    action.append(a)

                # Record evader action selection time
                evader_action = []
                for j, evader in enumerate(self.eval_env.evaders):
                    if evader.deactivated:
                        evader_action.append(None)
                        continue

                    a = self.evader_agent.act(evader_states[j])
                    evader_action.append(a)

                # Execute actions
                state, reward, done, info = self.eval_env.step((action, evader_action))
                evader_states, _ = self.eval_env.get_evaders_observation()

                for i, rob in enumerate(self.eval_env.pursuers):
                    if rob.deactivated:
                        continue

                    rewards[i] += self.pursuer_agent.GAMMA ** length * reward[i]
                    times[i] += rob.dt * rob.N
                    energies[i] += rob.compute_action_energy_cost(action[i])
                    # obs[i].append(copy.deepcopy(rob.perception.observed_obstacles))
                    # pursuers[i].append(copy.deepcopy(rob.perception.observed_pursuers))
                    # evaders[i].append(copy.deepcopy(rob.perception.observed_evaders))

                    if rob.collision:
                        rob.deactivated = True

                end_episode = ((length >= self.episode_max_length / 2)
                               or len([puruer for puruer in self.eval_env.pursuers if not puruer.deactivated]) < 3
                               or self.eval_env.check_all_evader_is_captured())

                if end_episode:
                    logger.info(
                        f"Episode ended at length {length}, "
                        f"len(pursuers) = {len([puruer for puruer in self.eval_env.pursuers if not puruer.deactivated])}, "
                        f"check_all_evader_is_captured = {self.eval_env.check_all_evader_is_captured()}")
                length += 1

            actions = []
            trajectories = []
            for rob in (self.eval_env.pursuers + self.eval_env.evaders):
                actions.append(copy.deepcopy(rob.action_history))
                trajectories.append(copy.deepcopy(rob.trajectory))

            success = True if self.eval_env.check_all_evader_is_captured() else False

            # actions_data.append(actions)
            # trajectories_data.append(trajectories)
            rewards_data.append(np.mean(rewards))
            successes_data.append(success)
            times_data.append(np.mean(times))
            energies_data.append(np.mean(energies))
            # obs_data.append(obs)
            # pursuers_data.append(pursuers)
            # evaders_data.append(evaders)

        avg_r = np.mean(rewards_data)
        success_rate = np.sum(successes_data) / len(successes_data)
        idx = np.where(np.array(successes_data) == 1)[0]
        avg_t = None if np.shape(idx)[0] == 0 else np.mean(np.array(times_data)[idx])
        avg_e = None if np.shape(idx)[0] == 0 else np.mean(np.array(energies_data)[idx])

        logger.info(f"Process {os.getpid()} ++++++++ Evaluation Info ++++++++")
        logger.info(f"Process {os.getpid()} - Avg cumulative reward: {avg_r:.2f}")
        logger.info(f"Process {os.getpid()} - Success rate: {success_rate:.2f}")
        if avg_t is not None:
            logger.info(f"Process {os.getpid()} - Avg time: {avg_t:.2f}")
            logger.info(f"Process {os.getpid()} - Avg energy: {avg_e:.2f}")
        logger.info("++++++++ Evaluation Info ++++++++\n")

        self.eval_timesteps.append(self.current_timestep)
        # self.eval_actions.append(actions_data)
        # self.eval_trajectories.append(trajectories_data)
        self.eval_rewards.append(rewards_data)
        self.eval_successes.append(successes_data)
        self.eval_times.append(times_data)
        self.eval_energies.append(energies_data)
        # self.eval_obs.append(obs_data)
        # self.eval_pursuers.append(pursuers_data)
        # self.eval_evaders.append(evaders_data)
        eval_metrics = {
            "evaluation/avg_reward": avg_r,
            "evaluation/success_rate": success_rate,
            "evaluation/episode_count": len(self.eval_config),
            "evaluation/timestep": self.current_timestep,
        }

        if avg_t is not None:
            eval_metrics.update({
                "evaluation/avg_time": avg_t,
                "evaluation/avg_energy": avg_e,
            })

        wandb.log(eval_metrics)

    def save_evaluation(self, eval_log_path: Path):
        """
        Save evaluation results.

        Args:
            eval_log_path (Path): Path for evaluation logs.
        """
        filename = "evaluations.npz"

        # Convert irregular arrays to object arrays
        eval_data = {
            "timesteps": np.array(self.eval_timesteps, dtype=object),
            "actions": np.array(self.eval_actions, dtype=object),
            # "trajectories": np.array(self.eval_trajectories, dtype=object),
            "rewards": np.array(self.eval_rewards, dtype=object),
            "successes": np.array(self.eval_successes),
            "times": np.array(self.eval_times),
            "energies": np.array(self.eval_energies),
        }

        os.makedirs(eval_log_path, exist_ok=True)
        np.savez(os.path.join(eval_log_path, filename), **eval_data)
