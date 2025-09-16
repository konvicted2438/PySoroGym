import numpy as np
import matplotlib.pyplot as plt
from soft_robot_kinematics import SoftRobotKinematics
import time

def main():
    """Main function demonstrating forward and inverse kinematics"""
    
    # Initialize the soft robot
    robot = SoftRobotKinematics()
    
    print("Soft Robot Kinematics - Python Implementation")
    print("=" * 50)
    
    # Forward kinematics example (equivalent to MATLAB code)
    print("\n1. Forward Kinematics:")
    print("Input pressures: [2, 1, 0] bar")
    
    act_pressure = np.array([0, 0.3, 2])  # pressure values in bar
    start_time = time.time()
    try:
        solution, y_tip, y_all = robot.forward_kinematics(act_pressure)
        computation_time = time.time() - start_time

        print(f"Computed tip position: [{y_tip[0]:.6f}, {y_tip[1]:.6f}, {y_tip[2]:.6f}] m")
        print(f"Computation time: {computation_time} seconds")
        
        # Visualize the result with tube
        robot.visualize_robot(y_all, "Forward Kinematics - Pressure [2,1,0] bar", show_tube=True)
        
    except Exception as e:
        print(f"Forward kinematics failed: {e}")
    
    # Inverse kinematics example
    print("\n2. Inverse Kinematics:")
    desired_position = np.array([0.02, 0.02, 0.042])  # unit: m
    print(f"Desired position: [{desired_position[0]:.3f}, {desired_position[1]:.3f}, {desired_position[2]:.3f}] m")
    
    try:
        pressures, y_tip_achieved, y_all_inv = robot.inverse_kinematics(desired_position)
        
        print(f"Required pressures: [{pressures[0]:.3f}, {pressures[1]:.3f}, {pressures[2]:.3f}] bar")
        print(f"Achieved tip position: [{y_tip_achieved[0]:.6f}, {y_tip_achieved[1]:.6f}, {y_tip_achieved[2]:.6f}] m")
        print(f"Position error: {np.linalg.norm(y_tip_achieved - desired_position):.6f} m")
        #print(f"Computation time: {robot.compu_time[-1]:.4f} seconds")
        
        # Visualize the result with tube and desired position
        robot.visualize_robot(y_all_inv, "Inverse Kinematics - Target [0.02, 0.02, 0.042] m", 
                             desired_position=desired_position, show_tube=True)
        
    except Exception as e:
        print(f"Inverse kinematics failed: {e}")
    
    # Performance analysis
    if len(robot.compu_time) > 0:
        print(f"\nPerformance Summary:")
        print(f"Average computation time: {np.mean(robot.compu_time):.4f} seconds")
        print(f"Total computations: {len(robot.compu_time)}")
    
    # Show all plots
    plt.show()

def test_workspace():
    """Test the robot's workspace by trying multiple positions"""
    robot = SoftRobotKinematics()
    
    print("\n3. Workspace Analysis:")
    print("Testing multiple target positions...")
    
    # Test points within expected workspace
    test_points = [
        [0.01, 0.01, 0.03],
        [0.015, 0.005, 0.035],
        [0.005, 0.015, 0.04],
        [-0.01, 0.01, 0.03],
        [0.01, -0.01, 0.03]
    ]
    
    successful_points = []
    failed_points = []
    
    for i, point in enumerate(test_points):
        try:
            desired_pos = np.array(point)
            pressures, achieved_pos, _ = robot.inverse_kinematics(desired_pos)
            error = np.linalg.norm(achieved_pos - desired_pos)
            
            if error < 0.001:  # 1mm tolerance
                successful_points.append(point)
                print(f"Point {i+1}: SUCCESS - Error: {error*1000:.2f} mm")
            else:
                failed_points.append(point)
                print(f"Point {i+1}: FAILED - Error: {error*1000:.2f} mm")
                
        except Exception as e:
            failed_points.append(point)
            print(f"Point {i+1}: ERROR - {str(e)[:50]}...")
    
    print(f"\nWorkspace Analysis Results:")
    print(f"Successful points: {len(successful_points)}/{len(test_points)}")
    print(f"Success rate: {len(successful_points)/len(test_points)*100:.1f}%")



if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment these for additional analysis
    # test_workspace()
    # pressure_sweep()
