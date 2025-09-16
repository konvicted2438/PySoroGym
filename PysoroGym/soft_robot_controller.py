import numpy as np
from typing import Optional


class SoftRobotController:
    """Controller for soft robot that handles pressure updates with smooth interpolation"""
    
    def __init__(self, soft_robot_body, interpolation_time=1.0, segment_index=None):
        """
        Initialize controller
        
        Parameters
        ----------
        soft_robot_body : SoftRobotBody
            The soft robot body to control
        interpolation_time : float
            Time in seconds to interpolate between pressure states
        segment_index : int or None
            If specified, control only this segment. If None, control all segments.
        """
        self.robot = soft_robot_body
        self.interpolation_time = interpolation_time
        self.segment_index = segment_index
        
        # Determine number of segments to control
        if segment_index is not None:
            self.n_segments = 1
        else:
            self.n_segments = len(soft_robot_body.segments)
        
        # Current and target pressures for each segment
        self.current_pressures = [np.zeros(3) for _ in range(self.n_segments)]
        self.target_pressures = [np.zeros(3) for _ in range(self.n_segments)]
        self.start_pressures = [np.zeros(3) for _ in range(self.n_segments)]
        
        # Interpolation tracking
        self.interpolation_start_time = 0.0
        self.is_interpolating = False
        
        # Command queue for sequential pressure commands
        self.command_queue = []
        self.current_command_index = 0
        
    def set_pressure_command(self, pressures, duration=None):
        """
        Set a single pressure command that will be interpolated to
        
        Parameters
        ----------
        pressures : array_like or list of array_like
            Target pressure values. If controlling single segment: [p1, p2, p3]
            If controlling all segments: [[p1, p2, p3], [p1, p2, p3], ...]
        duration : float or None
            Time to reach target (if None, uses default interpolation_time)
        """
        # Convert to list of arrays
        if self.segment_index is not None:
            # Single segment control
            self.target_pressures = [np.array(pressures)]
        else:
            # Multi-segment control
            if len(pressures) != self.n_segments:
                raise ValueError(f"Expected {self.n_segments} pressure arrays, got {len(pressures)}")
            self.target_pressures = [np.array(p) for p in pressures]
            
        self.start_pressures = [p.copy() for p in self.current_pressures]
        self.interpolation_start_time = 0.0  # Will be set on next update
        self.is_interpolating = True
        
        if duration is not None:
            self.interpolation_time = duration
            
    def set_pressure_sequence(self, pressure_commands):
        """
        Set a sequence of pressure commands to execute
        
        Parameters
        ----------
        pressure_commands : list of tuples
            Each tuple is (pressures, duration) where:
            - pressures: [p1, p2, p3] or [[p1,p2,p3], ...] depending on mode
            - duration: time to reach this state in seconds
        """
        self.command_queue = pressure_commands
        self.current_command_index = 0
        
        # Start with first command
        if len(self.command_queue) > 0:
            pressures, duration = self.command_queue[0]
            self.set_pressure_command(pressures, duration)
            
    def update(self, dt, current_time):
        """
        Update controller state and apply pressures to robot
        
        Parameters
        ----------
        dt : float
            Time step
        current_time : float
            Current simulation time
        """
        # Handle interpolation
        if self.is_interpolating:
            # Set start time if just beginning
            if self.interpolation_start_time == 0.0:
                self.interpolation_start_time = current_time
                
            # Calculate interpolation progress
            elapsed = current_time - self.interpolation_start_time
            t = min(elapsed / self.interpolation_time, 1.0)
            
            # Smooth interpolation using cosine
            smooth_t = 0.5 * (1 - np.cos(np.pi * t))
            
            # Interpolate pressures for each segment
            for i in range(self.n_segments):
                self.current_pressures[i] = (
                    self.start_pressures[i] * (1 - smooth_t) + 
                    self.target_pressures[i] * smooth_t
                )
            
            # Apply to robot
            self._apply_pressures()
            
            # Check if interpolation is complete
            if t >= 1.0:
                self.is_interpolating = False
                
                # Move to next command in queue if available
                if len(self.command_queue) > 0:
                    self.current_command_index += 1
                    if self.current_command_index < len(self.command_queue):
                        pressures, duration = self.command_queue[self.current_command_index]
                        self.set_pressure_command(pressures, duration)
                    else:
                        # Loop back to start
                        self.current_command_index = 0
                        pressures, duration = self.command_queue[0]
                        self.set_pressure_command(pressures, duration)
        else:
            # Maintain current pressures
            self._apply_pressures()
            
    def _apply_pressures(self):
        """Apply current pressures to the robot"""
        if self.segment_index is not None:
            # Control single segment
            self.robot.set_segment_pressures(self.segment_index, self.current_pressures[0])
        else:
            # Control all segments
            self.robot.set_all_pressures(self.current_pressures)
            
    def get_current_pressures(self):
        """Get current pressure values"""
        if self.segment_index is not None:
            return self.current_pressures[0].copy()
        else:
            return [p.copy() for p in self.current_pressures]
        
    def stop(self):
        """Stop all pressures (set to zero)"""
        if self.segment_index is not None:
            self.set_pressure_command([0, 0, 0], duration=0.5)
        else:
            zero_pressures = [[0, 0, 0] for _ in range(self.n_segments)]
            self.set_pressure_command(zero_pressures, duration=0.5)