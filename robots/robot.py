import copy
import heapq
import numpy as np


class Robot:
    """
    Base robot class defining common attributes and behaviors for all robot types.

    Attributes:
        id (int): Robot index identifier
        robot_type (str): Robot category (to be overridden in subclasses)
        dt (float): Discrete time step duration (seconds)
        N (int): Number of time steps per action
        length (float): Robot body length
        width (float): Robot body width
        obs_dis (float): Minimum safe distance threshold
        r (float): Collision radius
        detect_r (float): Detection range radius
        a (np.ndarray): Linear acceleration options (m/s²)
        w (np.ndarray): Angular velocity options (rad/s)
        action_list (list): Combined action space
        perception (Perception): Sensing module (to be implemented in subclasses)
        max_speed (float): Maximum movement speed (to be set in subclasses)
        x (float): World coordinate X position
        y (float): World coordinate Y position
        theta (float): Heading angle orientation
        speed (float): Forward movement speed
        velocity (np.ndarray): Velocity relative to seabed
        start (tuple): Initial position coordinates
        collision (bool): Collision status flag
        deactivated (bool): Operational status flag
        init_theta (float): Initial orientation angle
        init_speed (float): Initial movement speed
        action_history (list): Record of recent actions
        trajectory (list): Movement history coordinates
        coefficient_water_resistance (float): Hydrodynamic resistance factor

    Methods:
        compute_k(): Calculate water resistance coefficient
        compute_actions(): Generate action space combinations
        compute_actions_dimension(): Get action space size
        compute_dist_reward_scale(): Calculate distance reward scaling
        compute_penalty_matrix(): Generate action penalty matrix
        compute_action_energy_cost(): Calculate energy expenditure for actions
        reset_state(): Initialize robot state
        get_robot_transform(): Get coordinate transformation matrices
        get_steer_velocity(): Calculate directional velocity components
        update_velocity(): Update velocity vector
        update_state(): Advance robot state through time
        check_collision(): Detect entity collisions
        compute_distance(): Calculate entity separation
        check_detection(): Verify entity detectability
        project_to_robot_frame(): Coordinate system transformation
        get_padding_length(): Get observation padding size
        copy_sort(): Process sensor observations
        perception_output(): Generate perception data (abstract method)
    """

    def __init__(self, index):
        """
        Initialize base robot instance.

        Args:
            index (int): Unique identifier for the robot
        """
        # Core properties
        self.id = index
        self.robot_type = None  # To be overridden
        self.dt = 0.1  # Time step duration (s)
        self.N = 10  # Action steps

        # Physical dimensions
        self.length = 2.0
        self.width = 1.0

        # Spatial parameters
        self.obs_dis = 5.0  # Safety buffer distance
        self.detect_r = 0.5 * np.sqrt(self.length**2 + self.width**2)  # Detection range
        self.r = self.detect_r  # Collision boundary

        # Motion parameters (subclass-overridable)
        self.a = np.array([-0.4, 0.0, 0.4])  # Linear accelerations (m/s²)
        self.w = np.array([-np.pi/6, 0.0, np.pi/6])  # Angular velocities (rad/s)
        self.action_list = None

        # Subclass-defined properties
        self.perception = None  # To be implemented
        self.max_speed = None  # Subclass-specific

        # Dynamic state
        self.x = None  # World X
        self.y = None  # World Y
        self.theta = None  # Heading angle
        self.speed = None  # Forward speed
        self.velocity = None  # Seabed-relative velocity

        # Operational state
        self.start = None  # Spawn coordinates
        self.collision = False  # Collision flag
        self.deactivated = False  # Active status

        # Initial conditions
        self.init_theta = 0.0  # Starting orientation
        self.init_speed = 0.0  # Initial speed

        # Historical data
        self.action_history = []  # Action command log
        self.trajectory = []  # Movement path

        # Hydrodynamic property
        self.coefficient_water_resistance = None

    def compute_actions(self):
        """Generate Cartesian product of acceleration and angular velocity options."""
        self.action_list = [(acc, ang_v) for acc in self.a for ang_v in self.w]

    def compute_actions_dimension(self):
        """
        Get size of action space.

        Returns:
            int: Number of available actions
        """
        return len(self.action_list)

    def compute_dist_reward_scale(self):
        """
        Calculate distance reward normalization factor.

        Returns:
            float: Scaling factor for distance-based rewards
        """
        return 1 / (self.max_speed * self.N * self.dt)

    def compute_penalty_matrix(self):
        """
        Generate action penalty matrix for optimization.

        Returns:
            np.ndarray: 2x2 penalty matrix
        """
        scale_a = 1 / (np.max(self.a) * np.max(self.a))
        scale_w = 1 / (np.max(self.w) * np.max(self.w))
        p = -0.5 * np.array([[scale_a, 0.0], [0.0, scale_w]])
        return p

    def compute_action_energy_cost(self, action):
        """
        Calculate normalized energy cost for an action.

        Args:
            action (int): Action index from action_list

        Returns:
            float: Combined normalized energy expenditure
        """
        a, w = self.action_list[action]
        a /= np.max(self.a)
        w /= np.max(self.w)
        return np.abs(a) + np.abs(w)

    def reset_state(self, current_velocity=np.zeros(2)):
        """
        Reset robot to initial state with optional current velocity.

        Args:
            current_velocity (np.ndarray): Ocean current vector (default: [0,0])
        """
        self.action_history.clear()
        self.trajectory.clear()
        if self.start is None:
            raise ValueError("The initial position is not defined.")
        self.x = self.start[0]
        self.y = self.start[1]
        self.theta = self.init_theta
        self.speed = self.init_speed
        self.update_velocity(current_velocity)
        self.trajectory.append([self.x, self.y, self.theta, self.speed, self.velocity[0], self.velocity[1]])

    def get_robot_transform(self):
        """
        Get coordinate transformation matrices between world and robot frames.

        Returns:
            tuple: (rotation_matrix, translation_vector)
        """
        rotation_matrix_world_to_robot = np.array(
            [[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]]
        )
        translation_vector_world_to_robot = np.array([[self.x], [self.y]])
        return rotation_matrix_world_to_robot, translation_vector_world_to_robot

    def get_steer_velocity(self):
        """
        Calculate directional velocity components.

        Returns:
            np.ndarray: [vx, vy] velocity vector
        """
        return self.speed * np.array([np.cos(self.theta), np.sin(self.theta)])

    def update_velocity(self, current_velocity=np.zeros(2)):
        """
        Update velocity vector with current influence.

        Args:
            current_velocity (np.ndarray): Ocean current vector
        """
        steer_velocity = self.get_steer_velocity()
        self.velocity = steer_velocity + current_velocity

    def update_state(self, action, current_velocity):
        """
        Advance robot state through one time step.

        Args:
            action (int): Action index
            current_velocity (np.ndarray): Ocean current vector
        """
        self.update_velocity(current_velocity)
        dis = self.velocity * self.dt
        self.x += dis[0]
        self.y += dis[1]

        a, w = self.action_list[action]
        # Apply water resistance
        self.speed += a * self.dt
        self.speed = np.clip(self.speed, 0.0, self.max_speed)
        self.theta += w * self.dt
        self.theta = self.theta % (2 * np.pi)

    def check_collision(self, entities_x, entities_y, entities_r):
        """Check collision with circular entity."""
        distance = self.compute_distance(entities_x, entities_y, entities_r)
        if distance <= 0.0:
            self.collision = True

    def compute_distance(self, x, y, r, in_robot_frame=False):
        """
        Calculate distance to entity accounting for collision radii.

        Args:
            x (float): Entity X
            y (float): Entity Y
            r (float): Entity radius
            in_robot_frame (bool): Use robot coordinates

        Returns:
            float: Center-to-center distance minus radii
        """
        if in_robot_frame:
            distance = np.sqrt(x ** 2 + y ** 2) - r - self.r
        else:
            distance = np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2) - r - self.r
        return distance

    def project_to_robot_frame(self, array: np.ndarray, is_vector: bool = True) -> np.ndarray:
        """
        Transform coordinates/vectors to robot's frame.

        Args:
            array: Input coordinates/vector
            is_vector: Whether input is directional

        Returns:
            np.ndarray: Transformed coordinates
        """
        assert isinstance(array, np.ndarray), "The input needs to be a numpy array"
        assert np.shape(array) == (2,), "The input array must have shape (2,)"

        vector_world_frame = np.reshape(array, (2, 1))
        rotation_matrix_world_to_robot, translation_vector_world_to_robot = self.get_robot_transform()

        rotation_matrix_robot_to_world = np.transpose(rotation_matrix_world_to_robot)
        translation_vector_robot_to_world = -rotation_matrix_robot_to_world @ translation_vector_world_to_robot

        if is_vector:
            array_robot_frame = rotation_matrix_robot_to_world @ vector_world_frame
        else:
            array_robot_frame = rotation_matrix_robot_to_world @ vector_world_frame + translation_vector_robot_to_world

        array_robot_frame = array_robot_frame.reshape((2,))
        return array_robot_frame

    def get_padding_length(self, observation_type):
        """
        Get observation padding size based on robot type.

        Returns:
            int: Required padding length
        """
        padding_length_dict = {
            "evader": {"statics": 3, "pursuers": 4, "evaders": 4},
            "pursuer": {"statics": 5, "pursuers": 7, "evaders": 7}
        }
        return padding_length_dict[self.robot_type][observation_type]

    def copy_sort(self, max_num, observation_type, in_robot_frame=True):
        """
        Process sensor observations with sorting and padding.

        Returns:
            list: Padded observation vector
        """
        padding_length = self.get_padding_length(observation_type)
        if not self.perception.observation[observation_type]:
            return np.zeros(max_num * padding_length).tolist()

        observations_deepcopy_sorted = copy.deepcopy(heapq.nsmallest(
            max_num,
            self.perception.observation[observation_type],
            key=lambda entities: self.compute_distance(entities[0], entities[1], self.r, in_robot_frame)
        ))

        states = np.concatenate(observations_deepcopy_sorted)
        required_length = max_num * padding_length

        # 如果状态数组长度不足，进行填充
        if len(states) < required_length:
            padding = np.zeros(required_length - len(states))
            states = np.concatenate((states, padding))

        return states.tolist()

    def perception_output(self, obstacles, pursuers, evaders, in_robot_frame=True):
        """Abstract method for perception processing"""
        raise NotImplementedError("Must be implemented in subclass")