import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PysoroGym.World import World
from PysoroGym.Body import Body
from PysoroGym.physics import Material
from PysoroGym.visulisation import OpenGLRenderer
from PysoroGym.shapes import Box
from PysoroGym.soft_robot_body import SoftRobotBody
from PysoroGym.soft_robot_controller import SoftRobotController


class PushBoxEnv(gym.Env):
    """Environment where soft robot pushes a box from start to target position"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.steps = 0
        
        # Episode tracking
        self.episode_reward = 0
        self.reward_history = []
        self.reward_window_size = 50  # Check reward trend over last 50 steps
        self.early_stop_threshold = -0.5  # Stop if average reward is good
        self.stagnation_steps = 100  # Stop if no improvement for this many steps
        self.best_distance = float('inf')
        self.steps_since_improvement = 0
        
        # Two segments, 3 chambers each = 6 total actions
        self.n_chambers = 6
        
        # Define action space - 6 pressure values (2 segments × 3 chambers)
        self.action_space = spaces.Box(
            low=0.0, 
            high=1, 
            shape=(self.n_chambers,),
            dtype=np.float32
        )
        
        # Simplified observation space
        # Observation: [tip_pos(3), box_pos(3), target_pos(3)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),  # 3 + 3 + 3 = 9
            dtype=np.float32
        )
        
        # Create renderer once during initialization
        if self.render_mode == "human":
            self.renderer = OpenGLRenderer(width=1024, height=768)
        else:
            self.renderer = None
        
        # Initialize world and components
        self.world = None
        self.robot = None
        self.controller = None
        self.box = None
        self.ground = None
        self.target_zone = None
        
        # Environment parameters
        self.success_threshold = 0.1  # 10cm threshold
        self.box_size = 0.3  # 30cm cube
        self.box_height_offset = self.box_size / 2  # Center of box above ground
        
        # Initial positions
        self.robot_start_pos = np.array([0.0, 0.8, 0.0])
        self.box_start_pos = np.array([0.25, 0, 0.25])
        self.target_pos = np.array([0.4, 0.0, 0.0])  # Only x,z matter for success
        
        # Initialize environment
        self._setup_world()
        
    def _setup_world(self):
        """Initialize the simulation world"""
        # Create world with the existing renderer
        self.world = World(
            gravity=(0, -9.81, 0), 
            renderer=self.renderer,
            dt=1.0/240.0
        )
        
        # Create ground
        ground = Body(body_type=Body.STATIC)
        ground_material = Material(friction=1.0, elasticity=0.0)
        ground.add_shape(Box([100, 0.1, 100], color=(0.204, 0.275, 0.329)), material=ground_material)
        ground.position = np.array([0, -0.05, 0])
        self.world.add(ground)
        
        # Create box to push
        self.box = Body(mass=0.5, position=self.box_start_pos.copy())
        box_material = Material(friction=0.8, elasticity=0.1)
        self.box.linear_damping = 0.9  # High damping for stability
        self.box.angular_damping = 0.9
        box_shape = Box([self.box_size, self.box_size, self.box_size], color=(0.8, 0.3, 0.3))
        self.box.add_shape(box_shape, material=box_material)
        self.world.add(self.box)
        
        # # Create target zone (visual only)
        # self.target_zone = Body(body_type=Body.KINEMATIC)
        # target_shape = Box([0.12, 0.02, 0.12], color=(0.2, 0.8, 0.2, 0.5))
        # self.target_zone.add_shape(target_shape, material=Material(friction=0, elasticity=0))
        # self.target_zone.position = np.array([self.target_pos[0], 0.01, self.target_pos[2]])
        # self.world.add(self.target_zone)
        
        # Create robot
        self._create_robot()
        
    def _create_robot(self):
        """Create a two-segment soft robot"""
        segment_configs = [
            {
                'L': 0.05,  # 50mm length
                'rad_robot': 0.008,  # 8mm radius
                'pressures': [0.0, 0.0, 0.0],
                'tip_force': [0.0, 0.0, 0.0]
            },
            {
                'L': 0.05,  # 50mm length
                'rad_robot': 0.008,  # 8mm radius
                'pressures': [0.0, 0.0, 0.0],
                'tip_force': [0.0, 0.0, 0.0]
            }
        ]
        
        # Rotation matrix to point robot horizontally (along +x)
        angle = np.pi/2
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        
        self.robot = SoftRobotBody(
            position=tuple(self.robot_start_pos),
            orientation=rotation_matrix,
            n_segments_per_robot=10,
            n_sides=8,
            material=Material(friction=0.8, elasticity=0.2),
            scale=10.0,
            fixed_base=True,
            segment_configs=segment_configs
        )
        
        self.world.add(self.robot)
        
        # Create controller with slower response for stability
        self.controller = SoftRobotController(
            self.robot,
            interpolation_time=0.5  # 0.5 seconds to reach target pressure
        )
        
    def _get_observation(self):
        """Get simplified observation"""
        # Robot tip position
        tip_pos = self.robot.get_tip_position()
        
        # Box position (center of mass)
        box_pos = self.box.position
        
        # Target position (extend to 3D for consistency)
        target_3d = np.array([self.target_pos[0], box_pos[1], self.target_pos[2]])
        
        # Combine into observation
        obs = np.concatenate([
            tip_pos.flatten(),
            box_pos.flatten(),
            target_3d.flatten()
        ])
        
        return obs.astype(np.float32)
    
    def _get_reward(self):
        """Calculate reward for pushing task"""
        # Get positions
        tip_pos = self.robot.get_tip_position()
        box_pos = self.box.position
        
        # Distance from box to target (only x,z matter)
        box_to_target_2d = np.linalg.norm(box_pos[[0,2]] - self.target_pos[[0,2]])
        
        # Distance from robot tip to box
        robot_to_box = np.linalg.norm(tip_pos - box_pos)
        
        # Main reward: negative distance from box to target
        reward = -box_to_target_2d
        
        # Encourage robot to be close to box (but not too close)
        optimal_distance = 0.08  # Slightly more than box radius
        distance_error = abs(robot_to_box - optimal_distance)
        if distance_error < 0.05:
            reward += 0.2
        else:
            reward -= 0.1 * distance_error
        
        # Big bonus for success
        if box_to_target_2d < self.success_threshold:
            reward += 100.0
        
        # Small penalty for time
        reward -= 0.01
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # Reset tracking variables
        self.episode_reward = 0
        self.reward_history = []
        self.best_distance = float('inf')
        self.steps_since_improvement = 0
        
        # Clear world
        if self.world is not None:
            self.world.bodies.clear()
            if hasattr(self.world.physics_engine, 'clear_bodies'):
                self.world.physics_engine.clear_bodies()
        
        # Setup world
        self._setup_world()
        self.steps = 0
        
        # Reset box to start position (with small random offset)
        if seed is not None:
            offset = np.random.uniform(-0.02, 0.02, size=2)
            self.box.position = self.box_start_pos + np.array([offset[0], 0, offset[1]])
        
        observation = self._get_observation()
        info = {
            'box_position': self.box.position.copy(),
            'target_position': self.target_pos.copy()
        }
        
        return observation, info
    
    def step(self, action):
        """Perform one step in the environment"""
        # Reshape action for two segments
        pressure_command = action.reshape((2, 3))
        self.controller.set_pressure_command(pressure_command)
        
        # Step 1: Wait for pressure to reach target (interpolation phase)
        max_interpolation_steps = int(self.controller.interpolation_time / self.world.physics_engine.dt) + 10
        interpolation_steps = 0
        
        while self.controller.is_interpolating and interpolation_steps < max_interpolation_steps:
            # Update controller
            self.controller.update(self.world.physics_engine.dt, self.world.simulation_time)
            # Step physics
            self.world.step()
            interpolation_steps += 1
            
            # Render occasionally during interpolation
            if self.render_mode == "human" and interpolation_steps % 10 == 0:
                self.render()
        
        # Step 2: Let the robot move with the achieved pressure
        movement_steps = 60  # 0.25 seconds of movement
        for i in range(movement_steps):
            # Controller maintains pressure
            self.controller.update(self.world.physics_engine.dt, self.world.simulation_time)
            # Step physics
            self.world.step()
            
            # Render occasionally
            if self.render_mode == "human" and i % 10 == 0:
                self.render()
        
        self.steps += 1
        
        # Get observation and reward
        observation = self._get_observation()
        reward = self._get_reward()
        
        # Track rewards and performance
        self.episode_reward += reward
        self.reward_history.append(reward)
        
        # Check termination conditions
        box_to_target_2d = np.linalg.norm(self.box.position[[0,2]] - self.target_pos[[0,2]])
        
        # Track improvement
        if box_to_target_2d < self.best_distance:
            self.best_distance = box_to_target_2d
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
        
        # Success termination
        terminated = box_to_target_2d < self.success_threshold
        
        # Early termination conditions
        truncated = False
        early_stop_reason = None
        
        # 1. Maximum steps reached
        if self.steps >= self.max_steps:
            truncated = True
            early_stop_reason = "max_steps"
        
        # 2. Good performance - stop early if doing well
        elif len(self.reward_history) >= self.reward_window_size:
            avg_recent_reward = np.mean(self.reward_history[-self.reward_window_size:])
            if avg_recent_reward > self.early_stop_threshold and box_to_target_2d < 0.2:
                truncated = True
                early_stop_reason = "good_performance"
        
        # 3. Stagnation - no improvement for too long
        elif self.steps_since_improvement >= self.stagnation_steps:
            truncated = True
            early_stop_reason = "stagnation"
        
        # 4. Box fell off the platform or got too far
        elif abs(self.box.position[1]) > 1.0 or np.linalg.norm(self.box.position[[0,2]]) > 2.0:
            truncated = True
            early_stop_reason = "out_of_bounds"
            reward -= 50  # Penalty for losing the box
        
        info = {
            'distance_to_target': box_to_target_2d,
            'box_position': self.box.position.copy(),
            'success': terminated,
            'total_physics_steps': interpolation_steps + movement_steps,
            'episode_reward': self.episode_reward,
            'avg_recent_reward': np.mean(self.reward_history[-self.reward_window_size:]) if len(self.reward_history) >= self.reward_window_size else np.mean(self.reward_history),
            'steps_since_improvement': self.steps_since_improvement,
            'early_stop_reason': early_stop_reason
        }
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human" and self.renderer:
            import pygame
            
            # Process events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            
            # Render world
            self.world.render()
            
            # Small delay to make rendering visible
            pygame.time.wait(1)  # 1ms delay
            
        elif self.render_mode == "rgb_array" and self.renderer:
            self.world.render()
            return self.renderer.get_frame()
    
    def close(self):
        """Close the environment"""
        if self.world is not None:
            self.world.running = False
            self.world.bodies.clear()
            if hasattr(self.world.physics_engine, 'clear_bodies'):
                self.world.physics_engine.clear_bodies()
        
        if self.renderer is not None:
            import pygame
            pygame.quit()
            self.renderer = None


class CircularPushBoxEnv(PushBoxEnv):
    """Environment where soft robot pushes a box along a circular trajectory"""
    
    def __init__(self, render_mode=None, max_steps=2000):  # Increased max_steps
        # Set initial parameters before calling parent init
        self.circle_center = np.array([0.25, 0.0, 0.0])  # Adjusted center
        self.circle_radius = 0.1  # Smaller radius for easier task
        self.trajectory_points = 16  # Fewer waypoints initially
        self.current_waypoint = 0
        
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        
        # Override box start position to be on the circle
        angle = 2 * np.pi * 0 / self.trajectory_points  # Start at waypoint 0
        self.box_start_pos = np.array([
            self.circle_center[0] + self.circle_radius * np.cos(angle),
            0.08,  # Box height
            self.circle_center[2] + self.circle_radius * np.sin(angle)
        ])
        
        # No need to modify observation space - we'll replace target in parent's observation
        
    def _get_target_position(self, waypoint_idx):
        """Get position of a specific waypoint on the circle"""
        angle = 2 * np.pi * waypoint_idx / self.trajectory_points
        x = self.circle_center[0] + self.circle_radius * np.cos(angle)
        z = self.circle_center[2] + self.circle_radius * np.sin(angle)
        return np.array([x, 0.0, z])
    
    def _get_observation(self):
        """Get current observation including next waypoint"""
        # Get basic observation from parent class
        base_obs = super()._get_observation()
        
        # Get current waypoint position
        waypoint_pos = self._get_target_position(self.current_waypoint)
        
        # Replace the parent's static target (indices 9:12) with current waypoint
        base_obs[9:12] = waypoint_pos
        
        return base_obs.astype(np.float32)
    
    def _get_reward(self):
        """Calculate reward for circular pushing task"""
        # Get positions
        tip_pos = self.robot.get_tip_position()
        box_pos = self.box.position
        current_waypoint_pos = self._get_target_position(self.current_waypoint)
        next_waypoint_pos = self._get_target_position((self.current_waypoint + 1) % self.trajectory_points)
        
        # Distance from box to current waypoint
        box_to_waypoint = np.linalg.norm(box_pos[:2] - current_waypoint_pos[:2])
        
        # Distance from robot to box
        robot_to_box = np.linalg.norm(tip_pos - box_pos)
        
        # Base reward: negative distance to current waypoint
        reward = -box_to_waypoint * 2.0  # Scale up the importance
        
        # Bonus for being close to the box
        if robot_to_box < 0.12:
            reward += 0.2
        else:
            reward -= 0.2 * robot_to_box
        
        # Check if reached current waypoint
        if box_to_waypoint < self.success_threshold:
            reward += 20.0  # Big bonus for reaching waypoint
            self.current_waypoint = (self.current_waypoint + 1) % self.trajectory_points
            
            # Update visual target zone to new waypoint
            new_target = self._get_target_position(self.current_waypoint)
            self.target_zone.position = new_target + np.array([0, 0.01, 0])
            
            # Extra reward for completing full circle
            if self.current_waypoint == 0:
                reward += 100.0
        
        # Direction reward: encourage pushing towards next waypoint
        box_to_next = next_waypoint_pos[:2] - box_pos[:2]
        box_movement = self.box.linear_velocity[:2]
        if np.linalg.norm(box_to_next) > 0 and np.linalg.norm(box_movement) > 0:
            direction_alignment = np.dot(box_movement, box_to_next) / (np.linalg.norm(box_movement) * np.linalg.norm(box_to_next))
            reward += direction_alignment * 0.5
        
        # Energy penalty
        pressure_sum = np.sum([np.sum(p) for p in self.robot.get_all_pressures()])
        reward -= 0.01 * pressure_sum
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        # Reset waypoint to start
        self.current_waypoint = 0
        
        # Call parent reset
        observation, info = super().reset(seed=seed, options=options)
        
        # Override box position to start on the circle
        start_angle = 0  # Could randomize this: np.random.uniform(0, 2*np.pi)
        self.box.position = np.array([
            self.circle_center[0] + self.circle_radius * np.cos(start_angle),
            0.08,
            self.circle_center[2] + self.circle_radius * np.sin(start_angle)
        ])
        
        # Set target zone to first waypoint
        self.target_pos = self._get_target_position(self.current_waypoint)
        self.target_zone.position = self.target_pos + np.array([0, 0.01, 0])
        
        # Update observation with correct waypoint
        observation = self._get_observation()
        info['current_waypoint'] = self.current_waypoint
        info['target_position'] = self.target_pos.copy()
        
        return observation, info
    
    def _get_info(self):
        """Get diagnostic information for the current step"""
        info = super()._get_info()
        info['current_waypoint'] = self.current_waypoint
        info['waypoints_completed'] = self.current_waypoint  # Assumes starting from 0
        return info