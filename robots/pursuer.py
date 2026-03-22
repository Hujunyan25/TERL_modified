import copy
from typing import List, Tuple, Optional
import time

import numpy as np
import wandb

import utils
from config_manager import ConfigManager
from robots.evader import Evader
from robots.perception import Perception
from robots.robot import Robot
from utils import logger as logger


class Pursuer(Robot):
    """
    Pursuer class, inheriting from the Robot class, representing a pursuer in a pursuit-evasion scenario.

    Attributes:
        robot_type (str): Robot type, set to 'pursuer'.
        max_speed (float): Maximum speed.
        perception (Perception): Perception object, set as a pursuer type.
        captured_evaderId_list (List[int]): List of evader IDs captured by this pursuer.
        is_current_target_captured (bool): Whether the current tracking target has been captured.
        distance_capture (float): Maximum distance allowed for a successful capture.
        angle_capture (float): Minimum required encirclement angle for capture.
    """

    def __init__(self, index):
        """
        Initializes a pursuer instance.

        Args:
            index (int): Pursuer index.
        """
        super().__init__(index)
        # Pursuer-specific attributes
        # Overriding base class attributes
        self.pur_config = ConfigManager.get_instance()
        self.robot_type = "pursuer"
        self.capture_event_count = 0

        self.max_speed = self.pur_config.get("pursuer.max_speed", default=3.0)
        self.a = self.pur_config.get("pursuer.a", default=np.array([-0.4, 0.0, 0.4]))
        self.w = self.pur_config.get("pursuer.w", default=np.array([-0.5235987755982988, 0.0, 0.5235987755982988]))
        self.perception = Perception(is_evader=False)

        self.captured_evaderId_list: List[int] = []  # List of captured evaders
        self.is_current_target_captured = False  # Whether the current target has been captured
        self.is_pursuing = False  # Whether the pursuer is actively pursuing

        self.distance_capture = self.pur_config.get("env.capture_distance", default=8.0)  # Capture range threshold
        self.angle_capture = np.pi  # Minimum encirclement angle required for capture

        # Precomputed values
        self.compute_actions()  # Compute and update action list

        self.num_self_state = 4
        self.num_static_state = 10
        self.num_pursuer_state = 21
        self.num_evader_state = 7 * self.perception.max_evader_num

        self.num_self_feature = 4
        self.num_static_feature = 5
        self.num_pursuer_feature = 7
        self.num_evader_feature = 7

    def find_related_evader(self, evaders: List[Evader]):
        """
        Find evaders within the encirclement distance.

        Args:
            evaders (List[Evader]): List of all evaders.

        Returns:
            List[Evader]: List of evaders within the capture range, sorted by distance.
        """

        self_position = np.array([self.x, self.y])

        related_evaders = sorted(
            [e for e in evaders if
             np.linalg.norm(np.array([e.x, e.y]) - self_position) < self.distance_capture and not e.deactivated],
            key=lambda e: np.linalg.norm(np.array([e.x, e.y]) - self_position)
        )
        return related_evaders

    def check_capture_current_target(self, pursuers, evaders: List[Evader]) -> Tuple[
        bool, List[float], int, Optional[int]]:
        """
        Check whether the current target is captured.

        Capture conditions:
        1. At least 3 pursuers must surround the evader.
        2. All pursuers must be within `self.distance_capture` of the evader.
        3. Pursuers must form a true encirclement, not just be on the same side (maximum angle gap ≤ 180 degrees).
        4. The distribution of angles between adjacent pursuers should be relatively even (maximum angle ≤ 3x minimum angle).

        Args:
            pursuers: List of all pursuers.
            evaders (List[Evader]): List of all evaders.

        Returns:
            Tuple[bool, List[float], int, Optional[int]]:
                - Whether the capture is successful.
                - List of angles between adjacent pursuers (in radians).
                - Number of pursuers involved in the capture.
                - Captured evader ID (if successful).
        """
        # Configuration parameters
        MAX_ANGLE_GAP = np.pi  # Maximum allowable angle gap (180 degrees)
        MAX_ANGLE_RATIO = 3.0  # Maximum angle should not exceed 3 times the minimum angle
        MIN_PURSUERS = 3  # Minimum number of pursuers required besides itself (at least 3 in total)

        self_position = np.array([self.x, self.y])

        # Check each evader
        for evader in evaders:

            evader_position = np.array([evader.x, evader.y])

            # Check distance from the evader
            if np.linalg.norm(evader_position - self_position) > self.distance_capture:
                continue

            active_pursuers = [p for p in self.perception.observed_pursuers if not p.deactivated]
            if len(active_pursuers) < MIN_PURSUERS:
                continue

            # Compute distances of all pursuers from the evader
            pursuer_positions = np.array([[p.x, p.y] for p in active_pursuers])
            distances_to_evader = np.linalg.norm(pursuer_positions - evader_position, axis=1)

            # Get all pursuers within capture distance
            nearby_pursuers = [
                p for i, p in enumerate(active_pursuers)
                if distances_to_evader[i] < self.distance_capture
            ]

            # Check if the number of pursuers meets the minimum requirement
            if len(nearby_pursuers) < MIN_PURSUERS:
                continue

            involved_pursuers = nearby_pursuers + [self]

            # 步骤2：计算极角并排序（0~360度）
            polar_angles = []
            for uav in involved_pursuers:
                vector = uav - evader_position
                angle_rad = np.arctan2(vector[1], vector[0])
                angle_rad = angle_rad % (2 * np.pi)
                polar_angles.append(angle_rad)
            
            sorted_indices = np.argsort(polar_angles)
            sorted_angles = [polar_angles[i] for i in sorted_indices]
            
            # 步骤3：计算相邻夹角（闭环）
            adjacent_angles = []
            ideal_angle = 2 * np.pi / len(involved_pursuers)  # 理想均匀夹角
            for i in range(len(involved_pursuers)):
                curr = sorted_angles[i]
                next_ang = sorted_angles[(i + 1) % len(involved_pursuers)]
                diff = next_ang - curr
                diff = diff % (2 * np.pi)
                adjacent_angles.append(diff)

            ideal_angle = 2 * np.pi / len(involved_pursuers)
            avg_adjacent = np.mean(adjacent_angles)
            avg_diff = abs(avg_adjacent - ideal_angle)
            avg_ok = avg_diff <= 5 * np.pi / 180
            #每个夹角与理想值的偏差
            single_diffs = [abs(ang - ideal_angle) for ang in adjacent_angles]
            max_single_diffs = max(single_diffs)
            single_ok = max_single_diffs <= np.pi / 18

            is_encircled = avg_ok and single_ok
            if not is_encircled:
                continue

            # # Compute angles between adjacent pursuers
            # adjacent_angles = [
            #     (pursuer_angles[(i + 1) % len(pursuer_angles)] - pursuer_angles[i]) % (2 * np.pi)
            #     for i in range(len(pursuer_angles))
            # ]

            # # Check angle distribution
            # min_angle = min(adjacent_angles)
            # max_angle = max(adjacent_angles)

            # # Validate encirclement:
            # # 1. Maximum gap ≤ 180 degrees
            # # 2. Maximum angle ≤ 3 times the minimum angle
            # if max_angle > MAX_ANGLE_GAP or max_angle > min_angle * MAX_ANGLE_RATIO:
            #     continue

            # Mark capture status and log the event
            self.is_current_target_captured = True
            self.captured_evaderId_list.append(evader.id)

            capture_info = (
                f"\n{'='*50}\n"
                f"🎯 Capture Event #{self.capture_event_count}\n"
                f"{'='*50}\n"
                f"🤖 Lead Pursuer:\n"
                f"   • ID: {self.id}\n"
                f"   • Position: ({self.x:.2f}, {self.y:.2f})\n\n"
                f"🎯 Target Captured:\n"
                f"   • Evader ID: {evader.id}\n"
                f"   • Total Captures: {len(self.captured_evaderId_list)}\n"
                f"   • Capture History: {self.captured_evaderId_list}\n\n"
                f"👥 Assisting Pursuers ({len(nearby_pursuers)}):\n"
                f"   " + "\n   ".join([
                    f"• Pursuer {p.id}: ({p.x:.2f}, {p.y:.2f})"
                    for p in nearby_pursuers
                ]) + "\n"
                f"{'='*50}"
            )
            logger.info(capture_info)

            # Log capture event using dictionary format
            capture_metrics = {
                f"capture_event_{self.capture_event_count}/pursuer_id": self.id,
                f"capture_event_{self.capture_event_count}/position_x": self.x,
                f"capture_event_{self.capture_event_count}/position_y": self.y,
                f"capture_event_{self.capture_event_count}/captured_evader": evader.id,
                f"capture_event_{self.capture_event_count}/num_nearby_pursuers": len(nearby_pursuers),
                f"capture_event_{self.capture_event_count}/timestamp": time.time(),
            }

            # Log positions of assisting pursuers
            for idx, p in enumerate(nearby_pursuers):
                capture_metrics.update({
                    f"capture_event_{self.capture_event_count}/helper_{idx}/id": p.id,
                    f"capture_event_{self.capture_event_count}/helper_{idx}/x": p.x,
                    f"capture_event_{self.capture_event_count}/helper_{idx}/y": p.y,
                })

            wandb.log(capture_metrics)
            self.capture_event_count += 1

            return True, adjacent_angles, len(nearby_pursuers), evader.id

        # No capturable evader found
        return False, [], 0, None

    def perception_output(self, obstacles, pursuers, evaders, in_robot_frame=True):
        """
        Process the robot's (pursuer's) perception output of obstacles, pursuers, and evaders in the environment.

        Generates and returns perception state arrays including self-state, static obstacles, pursuers, and evaders.
        Returns None and collision status if the robot is deactivated.

        Args:
            obstacles (list): List of static obstacles in the environment.
            pursuers (list): List of pursuers in the environment.
            evaders (list): List of evaders in the environment.
            in_robot_frame (bool, optional): Whether the perception is in the robot's coordinate frame. Defaults to True.

        Returns:
            tuple:
                list: Perceived state arrays including self-state, static obstacles, pursuers, and evaders.
                bool: Collision status indicating whether the robot has collided with others.
        """
        if self.deactivated:
            return None, self.collision

        self.perception.observation["self"].clear()
        self.perception.observation["statics"].clear()
        self.perception.observation["evaders"].clear()
        self.perception.observation["pursuers"].clear()
        self.perception.observation['masks'].clear()
        self.perception.observation['types'].clear()
        self.perception.observed_pursuers.clear()
        self.perception.observed_evaders.clear()
        self.perception.observed_obstacles.clear()

        # Self-state perception (including target information)
        if in_robot_frame:
            # Velocity relative to seabed in robot's frame
            abs_velocity_r = self.project_to_robot_frame(self.velocity, is_vector=True)

            if obstacles:
                # Extract obstacle coordinates as numpy array
                obstacle_positions = np.array([[obs.x, obs.y] for obs in obstacles])  # Shape: (num_obstacles, 2)
                self_position = np.array([self.x, self.y])  # Shape: (2,)

                # Compute Euclidean distances between all obstacles and robot
                distances = np.linalg.norm(obstacle_positions - self_position, axis=1)  # Shape: (num_obstacles,)

                # Find nearest obstacle distance, return perception range if beyond detection
                min_distance = np.min(distances)

            self.perception.observation["self"] = list(abs_velocity_r) + [min_distance] + [self.is_pursuing]
            self.perception.observation['masks'].append(True)
            assert len(self.perception.observation["self"]) == self.num_self_state

        # Perception of other pursuers (cooperative pursuit)
        for k, pursuer in enumerate(pursuers):
            if pursuer is self:
                continue
            if pursuer.deactivated:
                continue

            self.perception.observed_pursuers.append(pursuer)

            if not self.collision:
                self.check_collision(pursuer.x, pursuer.y, pursuer.r)

            if in_robot_frame:
                pos_r = self.project_to_robot_frame(np.array([pursuer.x, pursuer.y]), False)
                v_r = self.project_to_robot_frame(pursuer.velocity)
                dis = np.linalg.norm(pos_r)
                angle = np.arctan2(pos_r[1], pos_r[0])
                pursuing_signal = pursuer.is_pursuing
                new_obs = list(np.concatenate((pos_r, v_r, [dis, angle, pursuing_signal])))
                self.perception.observation["pursuers"].append(new_obs)

        # Handle pursuer masks
        pursuer_observed_count = len(self.perception.observation["pursuers"])
        pursuer_masks = [True] * min(pursuer_observed_count, self.perception.max_pursuer_num)
        if pursuer_observed_count < self.perception.max_pursuer_num:
            pursuer_masks.extend([False] * (self.perception.max_pursuer_num - pursuer_observed_count))
        self.perception.observation['masks'].extend(pursuer_masks[:self.perception.max_pursuer_num])

        # Evader perception
        for j, evader in enumerate(evaders):
            if evader.deactivated:
                continue

            self.perception.observed_evaders.append(evader.id)

            if not self.collision:
                self.check_collision(evader.x, evader.y, evader.r)

            if in_robot_frame:
                pos_r = self.project_to_robot_frame(np.array([evader.x, evader.y]), False)
                v_r = self.project_to_robot_frame(evader.velocity)
                dis = np.linalg.norm(pos_r)
                pos_angle = np.arctan2(pos_r[1], pos_r[0])
                heading_angle_error: list = utils.calculate_angle_from_components(float(v_r[0]), float(v_r[1]))
                new_obs = list(np.concatenate((pos_r, v_r,))) + [dis, pos_angle] + heading_angle_error
                self.perception.observation["evaders"].append(new_obs)

        evader_observed_count = len(self.perception.observation["evaders"])
        evader_masks = [True] * min(evader_observed_count, self.perception.max_evader_num)
        if evader_observed_count < self.perception.max_evader_num:
            evader_masks.extend([False] * (self.perception.max_evader_num - evader_observed_count))
        self.perception.observation['masks'].extend(evader_masks[:self.perception.max_evader_num])

        # Static obstacle perception
        for i, obs in enumerate(obstacles):

            self.perception.observed_obstacles.append(i)

            if not self.collision:
                self.check_collision(obs.x, obs.y, obs.r)

            if in_robot_frame:
                pos_r = self.project_to_robot_frame(np.array([obs.x, obs.y]), False)
                dis = np.linalg.norm(pos_r)
                pos_angle = np.arctan2(pos_r[1], pos_r[0])
                self.perception.observation["statics"].append([pos_r[0], pos_r[1], obs.r, dis, pos_angle])

        static_observed_count = len(self.perception.observation["statics"])
        static_masks = [True] * min(static_observed_count, self.perception.max_obstacle_num)
        if static_observed_count < self.perception.max_obstacle_num:
            static_masks.extend([False] * (self.perception.max_obstacle_num - static_observed_count))
        self.perception.observation['masks'].extend(static_masks[:self.perception.max_obstacle_num])

        assert (len(self.perception.observation['masks']) == self.perception.max_obstacle_num +
                self.perception.max_pursuer_num + self.perception.max_evader_num + 1), \
            (f"masks: {self.perception.observation['masks']}, max_obstacle_num: {self.perception.max_obstacle_num},"
             f" max_pursuer_num: {self.perception.max_pursuer_num}, max_evader_num: {self.perception.max_evader_num}")

        self.perception.observation['types'].extend([0] +
                                                    [1] * self.perception.max_pursuer_num +
                                                    [2] * self.perception.max_evader_num +
                                                    [3] * self.perception.max_obstacle_num
                                                    )
        assert (len(self.perception.observation['types']) == self.perception.max_obstacle_num +
                self.perception.max_pursuer_num + self.perception.max_evader_num + 1), f"types: {self.perception.observation['types']}"

        # Process perception data
        self_state = copy.deepcopy(self.perception.observation["self"])
        static_observations = self.copy_sort(self.perception.max_obstacle_num, "statics", in_robot_frame)
        evader_observations = self.copy_sort(self.perception.max_evader_num, "evaders", in_robot_frame)
        pursuer_observations = self.copy_sort(self.perception.max_pursuer_num, "pursuers", in_robot_frame)

        assert len(self_state) == self.num_self_state
        assert len(static_observations) == self.num_static_state
        assert len(evader_observations) == self.num_evader_state
        assert len(pursuer_observations) == self.num_pursuer_state

        # Organize observations into structured format
        observation_dict = {
            "self": np.array(self_state).reshape(self.num_self_feature),
            "pursuers": np.array(pursuer_observations).reshape(-1, self.num_pursuer_feature),
            "evaders": np.array(evader_observations).reshape(-1, self.num_evader_feature),
            "obstacles": np.array(static_observations).reshape(-1, self.num_static_feature),
            "masks": np.array(self.perception.observation['masks']),  # Boolean masks indicating valid observations
            "types": np.array(self.perception.observation['types'])  # Entity type identifiers
        }

        return observation_dict, self.collision
