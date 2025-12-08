---
sidebar_position: 5
---

# ROS 2 Packages, Launch Files, and Parameter Management

## Learning Objectives

By the end of this section, you will be able to:
- Create and organize ROS 2 packages effectively
- Understand package structure and dependencies
- Create and use launch files to start multiple nodes
- Manage parameters using YAML files and command-line tools
- Configure robot applications with proper parameter handling

## Introduction to ROS 2 Packages

A **ROS 2 package** is the fundamental unit for organizing and distributing ROS 2 software. It contains nodes, libraries, configuration files, and other resources needed for a specific functionality. Proper package management is essential for creating maintainable and reusable robot software.

### Package Structure

A typical ROS 2 package follows this structure:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata and dependencies
├── src/                    # Source code files
│   ├── node1.cpp
│   └── node2.cpp
├── include/                # Header files (C++)
├── scripts/                # Standalone scripts (Python, bash, etc.)
├── launch/                 # Launch files
├── config/                 # Configuration files
├── params/                 # Parameter files
├── test/                   # Test files
└── README.md               # Package documentation
```

## Creating ROS 2 Packages

### Using colcon to Create Packages

ROS 2 provides tools to create packages with the proper structure:

```bash
# Create a C++ package
ros2 pkg create --build-type ament_cmake my_cpp_package

# Create a Python package
ros2 pkg create --build-type ament_python my_python_package

# Create a package with dependencies
ros2 pkg create --build-type ament_python --dependencies rclpy std_msgs sensor_msgs my_python_package
```

### Package.xml File

The `package.xml` file contains metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Package for my robot functionality</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>ament_python</buildtool_depend>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### setup.py for Python Packages

For Python packages, you need a `setup.py` file:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='Package for my robot functionality',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = my_robot_package.my_node:main',
            'another_node = my_robot_package.another_node:main',
        ],
    },
)
```

## Launch Files

Launch files allow you to start multiple nodes with specific configurations simultaneously. They're essential for managing complex robot systems.

### Python Launch Files

ROS 2 uses Python for launch files, providing flexibility and powerful features:

```python
# launch/my_robot_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    # Get launch configuration
    use_sim_time_config = LaunchConfiguration('use_sim_time')

    # Define nodes
    robot_controller = Node(
        package='my_robot_package',
        executable='robot_controller',
        name='robot_controller',
        parameters=[
            {'use_sim_time': use_sim_time_config},
            {'robot_name': 'my_robot'},
            {'max_velocity': 1.0}
        ],
        output='screen'
    )

    sensor_processor = Node(
        package='my_robot_package',
        executable='sensor_processor',
        name='sensor_processor',
        parameters=[
            {'use_sim_time': use_sim_time_config},
            {'sensor_topic': '/laser_scan'},
            {'processing_rate': 10.0}
        ],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time,
        robot_controller,
        sensor_processor
    ])
```

### Advanced Launch Features

#### Conditional Launch

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    # Launch argument for GUI
    gui = LaunchConfiguration('gui', default='true')

    # Conditional node launch
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(gui)  # Only launch if gui is true
    )

    return LaunchDescription([
        rviz_node
    ])
```

#### Launch File Includes

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Include another launch file
    other_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('other_package'),
            '/launch/other_launch.py'
        ])
    )

    return LaunchDescription([
        other_launch
    ])
```

## Parameter Management

Parameters in ROS 2 allow you to configure nodes without recompiling. They can be set at runtime through various methods.

### Parameter Declaration in Nodes

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('debug_mode', False)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.debug_mode = self.get_parameter('debug_mode').value

        # Set up parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info(f'Initialized with robot name: {self.robot_name}')

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity':
                if param.value > 5.0:
                    return SetParametersResult(successful=False, reason='Max velocity too high')

        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### YAML Parameter Files

Create parameter files in YAML format for easy configuration:

```yaml
# config/robot_params.yaml
/**:  # Applies to all nodes
  ros__parameters:
    use_sim_time: false
    robot_name: "my_humanoid_robot"
    control_frequency: 50.0

robot_controller:  # Applies to specific node
  ros__parameters:
    max_velocity: 1.0
    acceleration_limit: 2.0
    position_tolerance: 0.01
    velocity_tolerance: 0.05

sensor_processor:
  ros__parameters:
    sensor_topic: "/laser_scan"
    processing_rate: 10.0
    noise_threshold: 0.01

navigation_server:
  ros__parameters:
    planner_frequency: 5.0
    controller_frequency: 20.0
    recovery_enabled: true
    oscillation_timeout: 0.0
    oscillation_distance: 0.5
```

### Using Parameter Files in Launch Files

```python
# launch/robot_with_params.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get path to parameter file
    params_file = os.path.join(
        get_package_share_directory('my_robot_package'),
        'config',
        'robot_params.yaml'
    )

    # Node with parameter file
    robot_controller = Node(
        package='my_robot_package',
        executable='robot_controller',
        name='robot_controller',
        parameters=[params_file],
        output='screen'
    )

    return LaunchDescription([
        robot_controller
    ])
```

## Command Line Parameter Management

### Setting Parameters at Launch

```bash
# Set parameters when launching
ros2 run my_robot_package robot_node --ros-args -p robot_name:=my_robot -p max_velocity:=2.0

# Use parameter file
ros2 run my_robot_package robot_node --ros-args --params-file config/params.yaml

# Set parameters for specific node in launch
ros2 launch my_robot_package robot_launch.py --ros-args -p use_sim_time:=true
```

### Runtime Parameter Management

```bash
# List all parameters of a node
ros2 param list /robot_controller

# Get parameter value
ros2 param get /robot_controller robot_name

# Set parameter value
ros2 param set /robot_controller max_velocity 1.5

# List parameter descriptions
ros2 param describe /robot_controller robot_name

# Dump all parameters to a file
ros2 param dump /robot_controller
```

## Advanced Package Organization

### Metapackages

Metapackages group related packages together:

```xml
<!-- package.xml for metapackage -->
<?xml version="1.0"?>
<package format="3">
  <name>my_robot_metapackage</name>
  <version>0.0.0</version>
  <description>Metapackage for my robot system</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <exec_depend>my_robot_description</exec_depend>
  <exec_depend>my_robot_control</exec_depend>
  <exec_depend>my_robot_navigation</exec_depend>
  <exec_depend>my_robot_bringup</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### Package Dependencies

Managing dependencies properly is crucial:

```xml
<!-- In package.xml -->
<depend>rclcpp</depend>                    <!-- Build, exec, and test dependency -->
<build_depend>geometry_msgs</build_depend> <!-- Build-only dependency -->
<exec_depend>sensor_msgs</exec_depend>     <!-- Runtime dependency -->
<test_depend>ament_lint_auto</test_depend> <!-- Test-only dependency -->
```

## Build System Integration

### CMakeLists.txt for C++ Packages

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_robot_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Add executable
add_executable(robot_controller src/robot_controller.cpp)
ament_target_dependencies(robot_controller
  rclcpp
  std_msgs
  sensor_msgs)

# Install targets
install(TARGETS
  robot_controller
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

## Best Practices

### Package Organization

1. **Single Responsibility**: Each package should have a clear, focused purpose
2. **Consistent Naming**: Use descriptive, consistent names
3. **Proper Dependencies**: Only depend on what you actually use
4. **Documentation**: Include README.md with usage instructions

### Launch File Best Practices

1. **Modular Design**: Create reusable launch file components
2. **Parameter Flexibility**: Use launch arguments for configurable options
3. **Clear Structure**: Organize nodes logically in launch files
4. **Error Handling**: Include proper error checking and logging

### Parameter Management Best Practices

1. **Default Values**: Always provide sensible defaults
2. **Validation**: Implement parameter validation callbacks
3. **Documentation**: Document parameters clearly
4. **Grouping**: Group related parameters logically

## Exercise: Complete Package Setup

Create a complete ROS 2 package for a simple robot controller that includes:

1. A Python package with proper structure
2. A robot controller node with configurable parameters
3. A launch file that starts the controller with parameters
4. A YAML parameter file with different configurations
5. A README.md file with instructions

Test your package by launching it with different parameter configurations.

## Common Issues and Solutions

### 1. Package Not Found
- **Issue**: `command not found` or `package not found`
- **Solution**: Ensure package is built and workspace is sourced
  ```bash
  cd ~/ros2_workspace
  colcon build --packages-select my_robot_package
  source install/setup.bash
  ```

### 2. Parameter Type Mismatches
- **Issue**: Parameters are not the expected type
- **Solution**: Ensure parameter declarations match expected types and provide proper defaults

### 3. Launch File Errors
- **Issue**: Launch files fail to execute
- **Solution**: Check for syntax errors, proper package names, and correct file paths

## Summary

ROS 2 packages, launch files, and parameter management form the backbone of well-organized robot software. Understanding how to structure packages, create flexible launch files, and manage parameters effectively is crucial for developing maintainable and configurable robot systems. Proper use of these tools enables rapid prototyping, testing, and deployment of complex robotic applications.

In the next section, we'll work on exercises to reinforce all the concepts learned in Module 1.