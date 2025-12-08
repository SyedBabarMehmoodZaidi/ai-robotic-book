---
sidebar_position: 6
---

# Module 1 Exercises: ROS 2 Fundamentals

## Learning Objectives

By completing these exercises, you will:
- Apply ROS 2 architecture concepts to create functional robot systems
- Implement Python nodes that communicate via topics, services, and actions
- Create and use URDF models for robot representation
- Organize and manage ROS 2 packages with proper launch files and parameters
- Successfully control a simulated robot using ROS 2

## Exercise 1: Basic ROS 2 Communication Patterns

### Objective
Create a simple robot system that demonstrates all four communication patterns: nodes, topics, services, and actions.

### Requirements
1. Create a sensor node that publishes simulated sensor data (e.g., temperature, distance) to a topic
2. Create a data processor node that subscribes to the sensor data and processes it
3. Create a service server that can reset the sensor data or change processing mode
4. Create an action server that simulates a long-running robot task (e.g., moving to a location)

### Steps
1. Create a new ROS 2 package: `robot_exercises`
2. Implement the four nodes described above
3. Create a launch file that starts all nodes together
4. Test the communication between nodes

### Expected Outcome
- Sensor data continuously published to a topic
- Data processor receiving and processing the data
- Service client able to request resets or mode changes
- Action client able to send goals to the long-running task

### Verification
- Verify all nodes can communicate properly
- Check that services respond correctly
- Confirm actions provide feedback and can be canceled

## Exercise 2: Python Robot Controller

### Objective
Build a Python-based robot controller that integrates multiple ROS 2 concepts.

### Requirements
1. Use rclpy to create a robot controller node
2. Subscribe to simulated sensor data topics
3. Publish velocity commands to control a simulated robot
4. Provide a service to change the robot's operating mode
5. Use parameters to configure controller behavior
6. Include proper error handling and logging

### Implementation Details
```python
# Example structure for the controller
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from example_interfaces.srv import SetBool

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        # Initialize publishers, subscribers, services, etc.
        pass

    def sensor_callback(self, msg):
        # Process sensor data and compute control commands
        pass

    def control_loop(self):
        # Implement control algorithm
        pass

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
```

### Steps
1. Create the robot controller node with the specified functionality
2. Implement a simple control algorithm (e.g., obstacle avoidance)
3. Add parameter configuration for control gains
4. Create a service to enable/disable the controller
5. Test with simulated sensor data

### Verification
- Controller should respond to sensor data appropriately
- Service should enable/disable control behavior
- Parameters should be adjustable at runtime
- Control commands should be published at the correct rate

## Exercise 3: URDF Robot Model Creation

### Objective
Create a URDF model for a simple wheeled robot and validate it.

### Requirements
1. Create a URDF file for a differential drive robot with:
   - A base/chassis link
   - Two wheel links
   - A caster wheel link
   - Proper joints connecting all parts
2. Include visual, collision, and inertial properties
3. Use Xacro to simplify the URDF with macros
4. Validate the URDF file and visualize it

### URDF Structure
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_robot">
  <!-- Define constants -->
  <xacro:property name="wheel_radius" value="0.05" />
  <xacro:property name="wheel_width" value="0.02" />
  <xacro:property name="base_length" value="0.3" />
  <xacro:property name="base_width" value="0.2" />
  <xacro:property name="base_height" value="0.1" />

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Define wheel macro -->
  <xacro:macro name="wheel" params="prefix x_pos y_pos z_pos">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${x_pos} ${y_pos} ${z_pos}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Create wheels using macro -->
  <xacro:wheel prefix="left" x_pos="0.1" y_pos="${base_width/2 + wheel_width/2}" z_pos="0"/>
  <xacro:wheel prefix="right" x_pos="0.1" y_pos="${-base_width/2 - wheel_width/2}" z_pos="0"/>

  <!-- Caster wheel -->
  <link name="caster_wheel">
    <visual>
      <geometry>
        <sphere radius="${wheel_radius/2}"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="${wheel_radius/2}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_wheel"/>
    <origin xyz="${-base_length/2} 0 ${-wheel_radius/2}" rpy="0 0 0"/>
  </joint>
</robot>
```

### Steps
1. Create the URDF file using the structure above
2. Save it as `simple_robot.urdf.xacro`
3. Validate the URDF using `check_urdf` command
4. Visualize the robot in RViz
5. Create a launch file to display the robot model

### Verification
- URDF should validate without errors
- Robot should display correctly in RViz
- All joints should be properly connected
- Physical properties should be realistic

## Exercise 4: Package Organization and Launch Files

### Objective
Organize the previous exercises into a proper ROS 2 package with launch files and parameter management.

### Requirements
1. Create a well-structured ROS 2 package containing:
   - The sensor node from Exercise 1
   - The robot controller from Exercise 2
   - The URDF model from Exercise 3
2. Create parameter files for different configurations
3. Create launch files for different scenarios
4. Include proper documentation and setup files

### Package Structure
```
robot_exercises/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── resource/robot_exercises
├── robot_exercises/
│   ├── __init__.py
│   ├── sensor_node.py
│   ├── robot_controller.py
│   └── utils.py
├── launch/
│   ├── basic_launch.py
│   ├── full_robot_launch.py
│   └── simulation_launch.py
├── config/
│   ├── basic_params.yaml
│   ├── advanced_params.yaml
│   └── simulation_params.yaml
├── urdf/
│   └── simple_robot.urdf.xacro
├── test/
│   └── test_nodes.py
└── README.md
```

### Steps
1. Create the package structure as shown above
2. Move your nodes into the appropriate directories
3. Create launch files for different use cases
4. Create parameter files for different configurations
5. Update package.xml and setup.py appropriately
6. Build and test the package

### Launch File Example
```python
# launch/full_robot_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            get_package_share_directory('robot_exercises'),
            'config',
            'advanced_params.yaml'
        ),
        description='Path to parameters file'
    )

    # Get launch configurations
    use_sim_time_config = LaunchConfiguration('use_sim_time')
    params_file_config = LaunchConfiguration('params_file')

    # Robot controller node
    robot_controller = Node(
        package='robot_exercises',
        executable='robot_controller',
        name='robot_controller',
        parameters=[
            params_file_config,
            {'use_sim_time': use_sim_time_config}
        ],
        output='screen'
    )

    # Sensor node
    sensor_node = Node(
        package='robot_exercises',
        executable='sensor_node',
        name='sensor_node',
        parameters=[{'use_sim_time': use_sim_time_config}],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time,
        params_file,
        robot_controller,
        sensor_node
    ])
```

### Verification
- Package should build successfully with `colcon build`
- Launch files should start all required nodes
- Parameters should be configurable via YAML files
- All components should work together as expected

## Exercise 5: Integrated Robot Control

### Objective
Combine all concepts learned in Module 1 to create a complete robot control system.

### Requirements
1. Use your URDF model to represent a robot in simulation
2. Implement a controller that uses sensor data to navigate
3. Use services to change navigation modes
4. Use actions for complex navigation tasks
5. Manage all parameters through configuration files
6. Organize everything in a proper package with launch files

### Implementation Steps
1. Extend your robot controller to implement basic navigation
2. Add obstacle avoidance behavior using sensor data
3. Implement a service to change navigation modes (e.g., wander, follow-wall, go-to-goal)
4. Create an action server for goal navigation
5. Use launch files to start the complete system
6. Test in simulation (you can use Gazebo or a simple simulation environment)

### Navigation Algorithm Example
```python
# Simplified navigation algorithm
def navigate_to_goal(self):
    # Simple proportional controller for navigation
    error_x = self.goal_x - self.current_x
    error_y = self.goal_y - self.current_y

    distance = (error_x**2 + error_y**2)**0.5

    if distance < self.tolerance:
        return True  # Goal reached

    # Calculate desired heading
    desired_theta = math.atan2(error_y, error_x)
    current_theta = self.current_orientation

    # Simple proportional control
    angular_velocity = self.kp_angle * (desired_theta - current_theta)
    linear_velocity = min(self.max_linear_vel * distance, self.max_linear_vel)

    # Create and publish velocity command
    cmd_vel = Twist()
    cmd_vel.linear.x = linear_velocity
    cmd_vel.angular.z = angular_velocity

    self.cmd_vel_publisher.publish(cmd_vel)

    return False  # Goal not reached yet
```

### Verification
- Robot should be able to navigate to specified goals
- Obstacle avoidance should work properly
- Service should allow changing navigation modes
- Action server should provide feedback during navigation
- All parameters should be configurable

## Exercise 6: Control a Simulated Robot

### Objective
Connect your ROS 2 nodes to control a simulated robot and complete a navigation task.

### Requirements
1. Use your robot controller to control a simulated robot
2. Navigate the robot through a simple course
3. Use sensor data to avoid obstacles
4. Document the process and results

### Steps
1. If you have Gazebo installed, create a simple world with obstacles
2. Spawn your robot model in the simulation
3. Use your controller to navigate through the course
4. Record the robot's path and performance metrics
5. Document any challenges and solutions

### Alternative Simulation Setup
If Gazebo is not available, you can create a simple simulation using RViz markers and tf2 transforms to visualize robot movement.

### Verification
- Robot should successfully navigate the course
- Control system should respond appropriately to sensor data
- Navigation should be stable and reliable
- Results should be documented

## Challenge Exercise: Advanced Robot Features

### Objective
Extend your robot system with additional advanced features.

### Options (Choose at least one):
1. **Path Planning**: Implement A* or Dijkstra's algorithm for path planning
2. **Mapping**: Create a simple occupancy grid map from sensor data
3. **Localization**: Implement basic localization using sensor data
4. **Multi-Robot**: Coordinate multiple robots with different roles

### Requirements
- Integrate the chosen feature with your existing system
- Use proper ROS 2 communication patterns
- Document the implementation and results
- Test the system thoroughly

## Submission and Evaluation

### What to Submit
1. Complete ROS 2 package with all implemented nodes
2. Launch files for different scenarios
3. Parameter configuration files
4. URDF model files
5. Documentation of your implementation process
6. Results and performance metrics from the exercises

### Evaluation Criteria
- **Functionality** (40%): Do all components work as expected?
- **Code Quality** (20%): Is the code well-structured and documented?
- **ROS 2 Best Practices** (20%): Are ROS 2 concepts properly applied?
- **Problem Solving** (20%): How well are challenges addressed?

### Success Metrics
- Successfully complete at least 5 of the 6 main exercises
- Demonstrate understanding of all Module 1 concepts
- Show ability to integrate multiple ROS 2 components
- Achieve the independent test criteria: control a simulated robot

## Tips for Success

1. **Start Simple**: Begin with basic functionality and gradually add complexity
2. **Test Incrementally**: Test each component before integrating
3. **Use ROS Tools**: Leverage `ros2 topic`, `ros2 service`, `ros2 action` for debugging
4. **Check Documentation**: Refer to ROS 2 documentation when facing issues
5. **Validate Early**: Use URDF validators and check launch files before running

## Troubleshooting Common Issues

### Node Communication Issues
- Verify that nodes are on the same ROS domain
- Check that topic names match between publishers and subscribers
- Ensure message types are compatible

### Parameter Issues
- Verify parameter names match between declaration and usage
- Check that parameter files are properly formatted YAML
- Ensure launch files point to correct parameter file paths

### URDF Problems
- Use `check_urdf` to validate URDF files
- Verify all joints have proper parent-child relationships
- Check that visual and collision geometries are properly defined

## Summary

These exercises provide hands-on experience with all the fundamental concepts of ROS 2 covered in Module 1. By completing them, you will have built a functional robot system that demonstrates proper use of nodes, topics, services, actions, parameter management, and URDF modeling. The exercises progress from basic concepts to integrated systems, allowing you to apply your knowledge in increasingly complex scenarios.

Successfully completing these exercises demonstrates that you can create a ROS 2 system capable of controlling a simulated robot, meeting the independent test criteria for Module 1.