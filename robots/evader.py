import copy
import numpy as np
from robots.perception import Perception
from robots.robot import Robot
from config_manager import ConfigManager


class Evader(Robot):
    """
    Evader class, inheriting from Robot class, representing an evader in a pursuit-evasion scenario.

    Attributes:
        robot_type (str): Robot type, set to 'evader'.
        max_speed (float): Maximum speed.
        perception (Perception): Perception object, set as evader type.
    """

    def __init__(self, index):
        """
        Initializes an evader instance.

        Args:
            index (int): Evader index.
        """
        super().__init__(index)
        # Evader-specific attributes (new attributes or overridden base class attributes)
        self.eva_config = ConfigManager.get_instance()
        self.robot_type = "evader"
        self.max_speed = self.eva_config.get("evader.max_speed", default=3.0)
        self.a = self.eva_config.get("evader.a", default=np.array([-0.4, 0.0, 0.4]))
        self.w = self.eva_config.get("evader.w", default=np.array(
            [-0.5235987755982988, -0.2617993877991494, 0.0, 0.2617993877991494, 0.5235987755982988]))
        self.perception = Perception(is_evader=True)  # Override perception in subclass

        # Precomputed values
        self.compute_actions()  # Calculate and update action list

    def perception_output(self, obstacles, pursuers, in_robot_frame=True):
        """
        Process the evader's perception of obstacles, pursuers, and other evaders in the environment.

        Generates and returns perception state arrays including self-state, static obstacles,
        pursuers, and evaders. Returns None and collision status if the evader is deactivated.

        Args:
            obstacles (list): List of static obstacles in the environment
            pursuers (list): List of pursuers in the environment
            evaders (list): List of other evaders in the environment
            in_robot_frame (bool, optional): Whether to perceive in robot's coordinate frame. Defaults to True.

        Returns:
            tuple:
                list: Perceived state arrays including self-state, static obstacles, pursuers, and evaders
                bool: Collision status indicating whether the robot has collided with others
        """
        if self.deactivated:
            return None, self.collision

        # Clear perception data
        self.perception.observation["self"].clear()
        self.perception.observation["statics"].clear()
        self.perception.observation["pursuers"].clear()

        # Clear recorded index lists
        self.perception.observed_pursuers.clear()
        self.perception.observed_evaders.clear()
        self.perception.observed_obstacles.clear()

        # Self-state perception
        if in_robot_frame:
            # Project velocity to robot's coordinate frame
            abs_velocity_r = self.project_to_robot_frame(self.velocity)
            ego_states = [self.x, self.y] + list(abs_velocity_r)
            self.perception.observation["self"] = list(ego_states)  # Self-state including global position affects subsequent perception indices
        else:
            self.perception.observation["self"] = [self.x, self.y, self.theta, self.speed, self.velocity[0],
                                                   self.velocity[1]]

        # Static obstacle perception
        for i, obstacle in enumerate(obstacles):

            # Record observed obstacle
            self.perception.observed_obstacles.append(i)

            if not self.collision:
                self.check_collision(obstacle.x, obstacle.y, obstacle.r)  # Check collision with obstacle

            if in_robot_frame:
                position_robot_frame = self.project_to_robot_frame(np.array([obstacle.x, obstacle.y]), is_vector=False)
                self.perception.observation["statics"].append(
                    [position_robot_frame[0], position_robot_frame[1], obstacle.r])
            else:
                self.perception.observation["statics"].append([obstacle.x, obstacle.y, obstacle.r])

        # Pursuer perception
        for j, pursuer in enumerate(pursuers):
            if pursuer.deactivated:
                continue  # Skip deactivated pursuers

            # Record observed pursuer by ID (different from obstacle indexing which uses position in list)
            self.perception.observed_pursuers.append(pursuer.id)

            if not self.collision:
                self.check_collision(pursuer.x, pursuer.y, pursuer.r)  # Check collision with pursuer

            if in_robot_frame:
                position_robot_frame = self.project_to_robot_frame(np.array([pursuer.x, pursuer.y]), is_vector=False)
                v_r = self.project_to_robot_frame(pursuer.velocity, is_vector=True)
                new_pursuer_observation = list(np.concatenate((position_robot_frame, v_r)))
                self.perception.observation["pursuers"].append(new_pursuer_observation)
            else:
                self.perception.observation["pursuers"].append(
                    [pursuer.x, pursuer.y, pursuer.velocity[0], pursuer.velocity[1]])


        # Process perception data
        self_state = copy.deepcopy(self.perception.observation["self"])
        static_observations = self.copy_sort(self.perception.max_obstacle_num, "statics", in_robot_frame)
        pursuer_observations = self.copy_sort(self.perception.max_pursuer_num, "pursuers", in_robot_frame)

        return self_state + static_observations + pursuer_observations, self.collision
