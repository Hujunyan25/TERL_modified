import argparse
import copy
import json
import os
import sys
import time as t_module
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import wandb

from config_manager import ConfigManager
from utils import logger as logger
from environment.env import MarineEnv
from policy.agent import Agent
from thirdparty.APF import ApfAgent

sys.path.insert(0, "./thirdparty")


def evaluation(states, agent, evader_agent, eval_env: MarineEnv, use_rl=True, use_iqn=True, act_adaptive=True,
               save_episode=False):
    """Evaluate performance of the agent.
    """
    rob_num = len(eval_env.pursuers)

    rewards = [0.0] * rob_num
    times = [0.0] * rob_num
    energies = [0.0] * rob_num
    computation_times = []

    end_episode = False
    length = 0

    collision_id_set = set()
    experiment_min_distance = float("inf")

    pursuer_state, evader_state = states

    while not end_episode:
        # Gather actions for robots from agents
        action = []
        for i, rob in enumerate(eval_env.pursuers):
            if rob.deactivated:
                action.append(None)
                continue

            assert rob.robot_type == 'pursuer', "Every robot must be pursuer!"

            start = t_module.perf_counter()  # Use perf_counter for higher resolution timing
            if use_rl:
                if use_iqn:
                    if act_adaptive:
                        a, _, _, _ = agent.act_adaptive(pursuer_state[i])
                    else:
                        a, _, _ = agent.act(pursuer_state[i])
                else:
                    a, _ = agent.act_dqn(pursuer_state[i])
            else:
                a = agent.act(pursuer_state[i])
            end = t_module.perf_counter()
            computation_times.append(end - start)
            # logger.info(f"Agent {i} took {end - start:.4f} seconds to compute action.")

            action.append(a)

        evaders_action = []
        for j, evader in enumerate(eval_env.evaders):
            if evader.deactivated:
                evaders_action.append(None)
                continue

            assert evader.robot_type == 'evader', "Every robot must be evader!"

            a = evader_agent.act(evader_state[j])
            evaders_action.append(a)

        # Execute actions in the training environment
        pursuer_state, reward, done, info = eval_env.step((action, evaders_action))
        evader_state, _ = eval_env.get_evaders_observation()

        for i, rob in enumerate(eval_env.pursuers):
            if rob.deactivated:
                continue

            assert rob.robot_type == 'pursuer', "Every robot must be pursuer!"

            rewards[i] += agent.GAMMA ** length * reward[i]

            times[i] += rob.dt * rob.N
            energies[i] += rob.compute_action_energy_cost(action[i])

            if rob.collision:
                collision_id_set.add(rob.id)
                rob.deactivated = True

        min_distance = float("inf")

        # Calculate the minimum distance between active pursuers
        active_pursuers = [rob for rob in eval_env.pursuers if not rob.deactivated]
        active_evaders = [rob for rob in eval_env.evaders if not rob.deactivated]

        for i, rob1 in enumerate(active_pursuers):
            for j, rob2 in enumerate(active_pursuers):
                if rob1.id != rob2.id:
                    d = np.linalg.norm(np.array(rob1.x, rob1.y) - np.array(rob2.x, rob2.y))
                    min_distance = min(min_distance, d)

            # Calculate the minimum distance between pursuers and evaders
            for evader in active_evaders:
                d = np.linalg.norm(np.array(rob1.x, rob1.y) - np.array(evader.x, evader.y))
                min_distance = min(min_distance, d)

        # Record the minimum distance throughout the experiment
        experiment_min_distance = min(experiment_min_distance, min_distance)

        end_episode = (length >= 3000) or len([pursuer for pursuer in eval_env.pursuers if
                                               not pursuer.deactivated]) < 3 or eval_env.check_all_evader_is_captured()
        if end_episode:
            logger.info(
                f"Episode ended at length {length}, len(pursuers) = {len([pursuer for pursuer in eval_env.pursuers if not pursuer.deactivated])}, check_all_evader_is_captured = {eval_env.check_all_evader_is_captured()}")
        length += 1

    success = eval_env.check_all_evader_is_captured()

    # Modify time statistics logic
    # Directly take the maximum pursuit time
    max_pursuit_time = max(times)

    # Directly calculate the average pursuit time per evader
    avg_pursuit_time = max_pursuit_time / eval_env.num_evaders

    # Energy consumption statistics remain unchanged
    success_energies = [energies[i] for i, rob in enumerate(eval_env.pursuers) if not rob.deactivated]

    # Collision ratio
    collision_ratio = len(collision_id_set) / rob_num

    if save_episode:
        trajectories = [copy.deepcopy(rob.trajectory) for rob in eval_env.pursuers + eval_env.evaders]
        return success, rewards, computation_times, max_pursuit_time, avg_pursuit_time, success_energies, trajectories, collision_ratio, experiment_min_distance
    else:
        return success, rewards, computation_times, max_pursuit_time, avg_pursuit_time, success_energies, collision_ratio, experiment_min_distance


def exp_setup(envs, eval_schedule, i):
    """Set up the experiment environment for evaluation."""
    observations = []

    for test_env in envs:
        test_env.num_pursuers = eval_schedule["num_pursuers"][i]
        test_env.num_evaders = eval_schedule["num_evaders"][i]
        test_env.num_obs = eval_schedule["num_obstacles"][i]
        test_env.min_pursuer_evader_init_dis = eval_schedule["min_pursuer_evader_init_dis"][i]

        pursuer_state, _ = test_env.reset()
        state, _ = pursuer_state
        evader_state, _ = test_env.get_evaders_observation()
        states = [state, evader_state]
        observations.append(states)

    return observations


def dashboard(eval_schedule, indent=0):
    """Recursively print the configuration dictionary."""
    config = eval_schedule
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"Process {os.getpid()} -" + '  ' * indent + f"{key}:")
            logger.info(value, indent + 1)
        else:
            logger.info(f"Process {os.getpid()} -" + '  ' * indent + f"{key}: {value}")


def run_experiment(eval_schedules, index):
    """Run the experiment with the specified evaluation schedules."""
    agents = [Terl_agent, Dqn_agent]
    evader_agents = [evader_agent1, evader_agent2]
    names = model_name
    envs = [test_env_1, test_env_2]
    evaluations = [evaluation, evaluation]

    color_palette = ["#2E7BA6", "#B46FA2", "#2A8C66", "#C78A2A", "#4B61C6", "#E0BFE0", "#3D5A80", "#914E25"]

    colors = color_palette

    assert len(agents) == len(evader_agents) == len(envs) == len(evaluations) == len(names) <= len(
        colors), "Lengths of agents, evader_agents, envs, evaluations, and names must be the same! And the length of colors must be greater than or equal to the length of agents."

    save_trajectory = args.save_trajectory
    logger.info(f"Process {os.getpid()} - save_trajectory: {save_trajectory}")

    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")

    robot_nums = []
    # all_test_rob_exp = []
    all_successes_exp = []
    all_rewards_exp = []
    all_success_times_exp = []
    per_evader_success_times_exp = []
    all_success_energies_exp = []
    all_collision_ratios_exp = []
    all_experiment_min_distance_exp = []
    all_trajectories_exp = [] if save_trajectory else None
    all_eval_configs_exp = [] if save_trajectory else None

    for idx, count in enumerate(eval_schedules["num_episodes"]):
        dashboard(eval_schedules, idx)

        robot_nums.append(eval_schedules["num_pursuers"][idx])
        # all_test_rob = [0]*len(agents)
        all_successes = [[] for _ in agents]
        all_rewards = [[] for _ in agents]
        all_computation_times = [[] for _ in agents]
        all_max_success_times_with_success = [[] for _ in agents]
        all_avg_success_times_with_success = [[] for _ in agents]  # Average time per target
        all_success_energies = [[] for _ in agents]
        all_trajectories = [[] for _ in agents] if save_trajectory else None
        all_eval_configs = [[] for _ in agents] if save_trajectory else None
        all_collision_ratios = [[] for _ in agents]
        all_experiment_min_distance = [[] for _ in agents]

        for i in range(count):
            observations = exp_setup(envs, eval_schedules, idx)  # Include observations of all pursuers and evaders
            for j in range(len(agents)):
                logger.info(f"Evaluating all agents on schedule_{idx}_agent_{j}_ep_{i}")
                agent = agents[j]
                evader_agent = evader_agents[j]
                env = envs[j]
                eval_func = evaluations[j]
                name = names[j]

                if save_trajectory:
                    all_eval_configs[j].append(env.episode_data())

                # obs = env.reset()
                obs = observations[j]
                trajectories = None
                if save_trajectory:
                    if name == "TERL":
                        success, rewards, computation_times, max_pursuit_time, avg_pursuit_time, success_energies, trajectories, collision_ratio, experiment_min_distance = eval_func(
                            obs, agent, evader_agent, env, act_adaptive=False, save_episode=True)
                    elif name == "IQN":
                        success, rewards, computation_times, max_pursuit_time, avg_pursuit_time, success_energies, trajectories, collision_ratio, experiment_min_distance = eval_func(
                            obs, agent, evader_agent, env, act_adaptive=False, save_episode=True)
                    elif name == "MEAN":
                        success, rewards, computation_times, max_pursuit_time, avg_pursuit_time, success_energies, trajectories, collision_ratio, experiment_min_distance = eval_func(
                            obs, agent, evader_agent, env, act_adaptive=False, save_episode=True)
                    elif name == "DQN":
                        success, rewards, computation_times, max_pursuit_time, avg_pursuit_time, success_energies, trajectories, collision_ratio, experiment_min_distance = eval_func(
                            obs, agent, evader_agent, env, use_iqn=False, act_adaptive=False, save_episode=True)
                    else:
                        raise RuntimeError("Agent not implemented!")
                else:
                    if name == "TERL":
                        success, rewards, computation_times, max_pursuit_time, avg_pursuit_time, success_energies, collision_ratio, experiment_min_distance = eval_func(
                            obs, agent, evader_agent, env, act_adaptive=False, save_episode=False)
                    elif name == "IQN":
                        success, rewards, computation_times, max_pursuit_time, avg_pursuit_time, success_energies, collision_ratio, experiment_min_distance = eval_func(
                            obs, agent, evader_agent, env, act_adaptive=False, save_episode=False)
                    elif name == "MEAN":
                        success, rewards, computation_times, max_pursuit_time, avg_pursuit_time, success_energies, collision_ratio, experiment_min_distance = eval_func(
                            obs, agent, evader_agent, env, act_adaptive=False, save_episode=False)
                    elif name == "DQN":
                        success, rewards, computation_times, max_pursuit_time, avg_pursuit_time, success_energies, collision_ratio, experiment_min_distance = eval_func(
                            obs, agent, evader_agent, env, use_iqn=False, act_adaptive=False, save_episode=False)
                    else:
                        raise RuntimeError("Agent not implemented!")

                all_successes[j].append(success)
                # all_test_rob[j] += eval_schedules["num_cooperative"][idx]
                # all_successes[j] += success
                all_rewards[j] += rewards
                all_computation_times[j] += computation_times
                all_max_success_times_with_success[j].append(max_pursuit_time)
                all_avg_success_times_with_success[j].append(avg_pursuit_time)
                all_success_energies[j] += success_energies
                all_collision_ratios[j].append(collision_ratio)
                all_experiment_min_distance[j].append(experiment_min_distance)
                if save_trajectory:
                    all_trajectories[j].append(copy.deepcopy(trajectories))

        for k, name in enumerate(names):
            s_rate = np.sum(all_successes[k]) / len(all_successes[k])
            # s_rate = all_successes[k]/all_test_rob[k]
            avg_r = np.mean(all_rewards[k])
            avg_compute_t = np.mean(all_computation_times[k])
            avg_t = np.mean(all_max_success_times_with_success[k])
            avg_t_per_evader = np.mean(all_avg_success_times_with_success[k])
            avg_e = np.mean(all_success_energies[k])
            avg_collision_ratio = np.mean(all_collision_ratios[k])
            avg_experiment_min_distance = np.mean(all_experiment_min_distance[k])
            logger.info(f"{name} | success rate: {s_rate:.2f} | avg_reward: {avg_r:.2f} | avg_compute_t: {avg_compute_t} | \
                  avg_t: {avg_t:.2f} | avg_t_per_evader: {avg_t_per_evader:.2f}| avg_e: {avg_e:.2f} "
                        f"| avg_collision_ratio: {avg_collision_ratio:.2f} | avg_experiment_min_distance: {avg_experiment_min_distance:.2f}")

        logger.info("\n")

        # all_test_rob_exp.append(all_test_rob)
        all_successes_exp.append(all_successes)
        all_rewards_exp.append(all_rewards)
        all_success_times_exp.append(all_max_success_times_with_success)
        per_evader_success_times_exp.append(all_avg_success_times_with_success)
        all_success_energies_exp.append(all_success_energies)
        all_collision_ratios_exp.append(all_collision_ratios)
        all_experiment_min_distance_exp.append(all_experiment_min_distance)
        if save_trajectory:
            all_trajectories_exp.append(copy.deepcopy(all_trajectories))
            all_eval_configs_exp.append(copy.deepcopy(all_eval_configs))

    # Save data
    if save_trajectory:
        exp_data = dict(eval_schedules=eval_schedules,
                        names=names,
                        all_successes_exp=all_successes_exp,
                        all_rewards_exp=all_rewards_exp,
                        all_success_times_exp=all_success_times_exp,
                        all_success_energies_exp=all_success_energies_exp,
                        all_collision_ratios_exp=all_collision_ratios_exp,
                        all_experiment_min_distance_exp=all_experiment_min_distance_exp,
                        all_trajectories_exp=all_trajectories_exp,
                        all_eval_configs_exp=all_eval_configs_exp
                        )
    else:
        exp_data = dict(eval_schedules=eval_schedules,
                        names=names,
                        all_successes_exp=all_successes_exp,
                        all_rewards_exp=all_rewards_exp,
                        all_success_times_exp=all_success_times_exp,
                        all_success_energies_exp=all_success_energies_exp,
                        all_collision_ratios_exp=all_collision_ratios_exp,
                        all_experiment_min_distance_exp=all_experiment_min_distance_exp,
                        )

    exp_dir = f"experiment_data/baseline_exp_data_{index}_{timestamp}"
    os.makedirs(exp_dir)

    filename = os.path.join(exp_dir, "exp_results.json")
    with open(filename, "w") as file:
        json.dump(exp_data, file)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()

    bar_width = 0.25
    interval_scale = 1.25
    set_label = [True] * len(names)

    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    for i, robot_num in enumerate(robot_nums):
        all_successes = all_successes_exp[i]
        all_max_success_times_with_success = all_success_times_exp[i]
        all_avg_success_times_with_success = per_evader_success_times_exp[i]
        all_success_energies = all_success_energies_exp[i]
        all_collision_ratios = all_collision_ratios_exp[i]
        all_experiment_min_distance = all_experiment_min_distance_exp[i]

        for j, pos in zip(range(len(all_successes)), offsets):
            # Bar plot for success rate
            s_rate = np.sum(all_successes[j]) / len(all_successes[j])
            if set_label[j]:
                ax1.bar(interval_scale * i + pos, s_rate, 0.8 * bar_width, color=colors[j], label=names[j])
                set_label[j] = False
            else:
                ax1.bar(interval_scale * i + pos, s_rate, 0.8 * bar_width, color=colors[j])

            # Box plot for time
            box = ax2.boxplot(all_max_success_times_with_success[j], positions=[interval_scale * i + pos],
                              flierprops={'marker': '.', 'markersize': 1}, patch_artist=True)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[j])
            for line in box["medians"]:
                line.set_color("black")

            # Box plot for energy
            box = ax3.boxplot(all_success_energies[j], positions=[interval_scale * i + pos],
                              flierprops={'marker': '.', 'markersize': 1}, patch_artist=True)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[j])
            for line in box["medians"]:
                line.set_color("black")

            # Box plot for time per evader
            box = ax4.boxplot(all_avg_success_times_with_success[j], positions=[interval_scale * i + pos],
                              flierprops={'marker': '.', 'markersize': 1}, patch_artist=True)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[j])
            for line in box["medians"]:
                line.set_color("black")

            # Bar plot for collision ratio
            if set_label[j]:
                ax5.bar(interval_scale * i + pos, np.mean(all_collision_ratios[j]), 0.8 * bar_width, color=colors[j],
                        label=names[j])
                set_label[j] = False
            else:
                ax5.bar(interval_scale * i + pos, np.mean(all_collision_ratios[j]), 0.8 * bar_width, color=colors[j])

            # Line plot for experiment min distance
            ax6.plot(interval_scale * i + pos, np.mean(all_experiment_min_distance[j]), marker='o', color=colors[j],
                     label=names[j])

    # Ensure the number of ticks matches the number of labels
    xticks = interval_scale * np.arange(len(robot_nums))

    evader_nums = eval_schedules["num_evaders"]
    xticklabels = [f"{num1}/{num2}" for num1, num2 in zip(robot_nums, evader_nums)]

    # Set ticks and labels for ax1
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    ax1.set_title("Success Rate")
    ax1.legend()

    # Set ticks and labels for ax2
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)
    ax2.set_title("Time")

    # Set ticks and labels for ax3
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticklabels)
    ax3.set_title("Energy")

    ax4.set_xticks(xticks)
    ax4.set_xticklabels(xticklabels)
    ax4.set_title("Time per Evader")

    ax5.set_xticks(xticks)
    ax5.set_xticklabels(xticklabels)
    ax5.set_title("Collision Ratio")

    ax6.set_xticks(xticks)
    ax6.set_xticklabels(xticklabels)
    ax6.set_title("Experiment Min Distance")

    # Save figures
    fig1.savefig(os.path.join(exp_dir, "success_rate.png"))
    fig2.savefig(os.path.join(exp_dir, "time.png"))
    fig3.savefig(os.path.join(exp_dir, "energy.png"))
    fig4.savefig(os.path.join(exp_dir, "time_per_evader.png"))
    fig5.savefig(os.path.join(exp_dir, "collision_ratio.png"))
    fig6.savefig(os.path.join(exp_dir, "experiment_min_distance.png"))

    # plt.show()


def initialize_wandb(config):
    """Initialize Weights & Biases (wandb)."""
    try:
        wandb.init(
            project=args.project,
            group=args.group,
            name=f"Baseline Experiment",
            config=config,
        )
        return True
    except Exception as e:
        raise f"Failed to initialize wandb: {e}"


def initialize_config_manager(config_file):
    """Initialize and load configuration manager."""
    config_manager = ConfigManager()
    config_manager.load_config(config_file)
    return config_manager


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run experiment with different training data and model index.')

    parser.add_argument(
        "-D", "--device", type=str, required=True,
        help="Device to run all subprocesses (e.g., 'cuda:0', 'cpu')"
    )

    parser.add_argument(
        "-S", "--save_trajectory", action="store_true",
        help="Save the trajectory of the pursuers and evaders"
    )

    parser.add_argument(
        "-C", "--config", type=int, required=True,
        help="Path to the config file"
    )

    # Add wandb-related parameters
    parser.add_argument(
        "--project",
        type=str,
        default="baseline_experiment",
        help="wandb project name"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="wandb group name"
    )

    # Parse command-line arguments
    args = parser.parse_args()

    seed = 6  # PRNG seed for all testing envs

    exp_config_list = [
        "exp_config_small_scale.yaml",
        "exp_config_medium_scale.yaml",
        "exp_config_less_pursuer_more_evader.yaml",
        "exp_config_large_scale.yaml",
    ]

    # Model names
    model_name = [
        "TERL",
        "DQN",
    ]

    save_dir = f"TrainedModels/{model_name[0]}"

    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir =os.path.join(project_root, save_dir)
    config_file = os.path.join(project_root, "config", f"{exp_config_list[args.config]}")
    logger.info(f"config_file: {config_file}")

    config_manager = initialize_config_manager(config_file)
    device = args.device
    exp_config = config_manager.get_instance()
    exp_schedule = exp_config.get("exp_schedule")

    initialize_wandb(exp_schedule)

    test_env_1 = MarineEnv(seed)
    Terl_agent = Agent(device=device, model_name=model_name[0])
    Terl_agent.load_model(model_dir, device)
    evader_agent1 = ApfAgent(test_env_1.evaders[0].a, test_env_1.evaders[0].w)

    save_dir = f"TrainedModels/{model_name[1]}"
    model_dir = os.path.join(project_root,save_dir)

    test_env_2 = MarineEnv(seed)
    Dqn_agent = Agent(device=device, model_name=model_name[1], use_iqn=False)
    Dqn_agent.load_model(model_dir, device)
    evader_agent2 = ApfAgent(test_env_2.evaders[0].a, test_env_2.evaders[0].w)



    run_experiment(exp_schedule, args.config)
    wandb.finish()