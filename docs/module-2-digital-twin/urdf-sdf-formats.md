---
sidebar_position: 4
---

# URDF/SDF Formats: Robot Description and Simulation Formats

## Learning Objectives

By the end of this section, you will be able to:
- Understand the differences between URDF and SDF formats
- Convert between URDF and SDF for different simulation needs
- Create SDF files for Gazebo simulation from URDF models
- Work with both formats effectively in robotics applications
- Choose the appropriate format for specific use cases

## Introduction to URDF vs SDF

**URDF (Unified Robot Description Format)** and **SDF (Simulation Description Format)** are two XML-based formats used to describe robots and simulation environments. While URDF is primarily focused on robot structure for ROS, SDF is designed for comprehensive simulation environments in Gazebo.

Understanding both formats is crucial because:
- **URDF** is used for robot description in ROS (kinematics, dynamics, visualization)
- **SDF** is used for simulation environments in Gazebo (physics, sensors, world)
- **Conversion** between formats is often necessary for complete robotics workflows

## URDF: The Robot Description Format

### URDF Overview

URDF is specifically designed for robot description and is deeply integrated with ROS. It focuses on describing the robot's structure, kinematics, and basic dynamics.

### URDF Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links: rigid parts of the robot -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints: connections between links -->
  <link name="upper_body">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
    </visual>
  </link>

  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="upper_body"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>
</robot>
```

### URDF Features

- **Robot Structure**: Links and joints defining the kinematic tree
- **Visual Properties**: How the robot appears in RViz and other visualization tools
- **Collision Properties**: How the robot interacts with obstacles
- **Inertial Properties**: Mass, center of mass, and inertia tensor for dynamics
- **Transmission Information**: How joints connect to actuators (ROS-specific)

### URDF Limitations for Simulation

URDF alone is insufficient for full simulation because it lacks:
- **World Description**: Environment, lighting, terrain
- **Sensor Models**: Detailed sensor physics and noise models
- **Physics Parameters**: Damping, friction, surface properties
- **Plugin Integration**: Custom simulation behaviors and controllers

## SDF: The Simulation Description Format

### SDF Overview

SDF is designed for complete simulation environments and is the native format for Gazebo. It encompasses everything needed for physics simulation, from robot models to world environments.

### SDF Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Models in the world -->
    <model name="my_robot">
      <!-- Pose of the model in the world -->
      <pose>0 0 0 0 0 0</pose>

      <!-- Links with more detailed physics -->
      <link name="base_link">
        <pose>0 0 0 0 0 0</pose>

        <!-- Inertial properties -->
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>

        <!-- Visual properties -->
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>0.6</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 0 0.8 1</ambient>
            <diffuse>0 0 0.8 1</diffuse>
          </material>
        </visual>

        <!-- Collision properties -->
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>0.6</length>
            </cylinder>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
            <bounce>
              <restitution_coefficient>0.0</restitution_coefficient>
              <threshold>100000</threshold>
            </bounce>
            <contact>
              <ode>
                <soft_cfm>0</soft_cfm>
                <soft_erp>0.2</soft_erp>
                <kp>1000000000000.0</kp>
                <kd>1.0</kd>
                <max_vel>0.01</max_vel>
                <min_depth>0.001</min_depth>
              </ode>
            </contact>
          </surface>
        </collision>

        <!-- Sensors -->
        <sensor name="imu_sensor" type="imu">
          <always_on>true</always_on>
          <update_rate>100</update_rate>
          <imu>
            <angular_velocity>
              <x>
                <noise type="gaussian">
                  <mean>0.0</mean>
                  <stddev>0.001</stddev>
                </noise>
              </x>
              <y>
                <noise type="gaussian">
                  <mean>0.0</mean>
                  <stddev>0.001</stddev>
                </noise>
              </y>
              <z>
                <noise type="gaussian">
                  <mean>0.0</mean>
                  <stddev>0.001</stddev>
                </noise>
              </z>
            </angular_velocity>
          </imu>
        </sensor>
      </link>

      <!-- Joints -->
      <joint name="torso_joint" type="fixed">
        <parent>base_link</parent>
        <child>upper_body</child>
        <pose>0 0 0.5 0 0 0</pose>
      </joint>
    </model>

    <!-- World elements -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### SDF Features

- **Complete World Description**: Environment, physics, lighting
- **Advanced Physics**: Detailed friction, contact, and collision models
- **Sensor Integration**: Built-in sensor models with noise characteristics
- **Plugin Architecture**: Extensible functionality through plugins
- **Multi-Model Support**: Multiple robots and objects in one environment

## Converting URDF to SDF

### Using gazebo_ros_pkgs

The easiest way to use URDF models in Gazebo is through the ROS integration:

```xml
<!-- In a Gazebo world file -->
<include>
  <name>my_robot</name>
  <pose>0 0 0 0 0 0</pose>
  <uri>model://my_robot_description/urdf/robot.urdf</uri>
</include>
```

### Direct Conversion

To convert URDF to SDF manually:

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Convert and specify SDF version
gz sdf -p --in-format urdf --out-format sdf robot.urdf > robot.sdf
```

### Programmatic Conversion

```python
import xml.etree.ElementTree as ET
import subprocess

def urdf_to_sdf(urdf_file_path, sdf_file_path):
    """Convert URDF to SDF using gz sdf tool"""
    try:
        # Use gz sdf to convert
        result = subprocess.run(
            ['gz', 'sdf', '-p', urdf_file_path],
            capture_output=True,
            text=True,
            check=True
        )

        with open(sdf_file_path, 'w') as sdf_file:
            sdf_file.write(result.stdout)

        print(f"Successfully converted {urdf_file_path} to {sdf_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting URDF to SDF: {e}")
    except FileNotFoundError:
        print("Error: gz command not found. Make sure Gazebo is installed.")

# Usage
urdf_to_sdf('robot.urdf', 'robot.sdf')
```

## Detailed Format Comparison

### Element Mapping

| URDF Element | SDF Equivalent | Notes |
|--------------|----------------|-------|
| `<robot>` | `<model>` | Robot becomes a model in SDF |
| `<link>` | `<link>` | Similar structure |
| `<joint>` | `<joint>` | Similar structure |
| `<visual>` | `<visual>` | Similar structure |
| `<collision>` | `<collision>` | Enhanced with surface properties |
| `<inertial>` | `<inertial>` | Similar structure |

### SDF Extensions

SDF includes many elements that URDF doesn't have:

#### Physics Extensions
```xml
<collision name="collision">
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
        <fdir1>0 0 1</fdir1>
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.5</restitution_coefficient>
    </bounce>
    <contact>
      <ode>
        <soft_cfm>0.0</soft_cfm>
        <soft_erp>0.2</soft_erp>
      </ode>
    </contact>
  </surface>
</collision>
```

#### Sensor Definitions
```xml
<sensor name="camera" type="camera">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>camera</namespace>
      <remapping>~/image_raw:=image</remapping>
    </ros>
  </plugin>
</sensor>
```

#### Plugin Integration
```xml
<model name="my_robot">
  <!-- Custom controller plugin -->
  <plugin name="diff_drive_controller" filename="libgazebo_ros_diff_drive.so">
    <ros>
      <namespace>diff_drive</namespace>
    </ros>
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.3</wheel_separation>
    <wheel_diameter>0.1</wheel_diameter>
  </plugin>
</model>
```

## Practical Examples

### Complete URDF to SDF Workflow

1. **Create URDF for Robot Description**
   - Define links, joints, and basic properties
   - Focus on kinematics and basic dynamics

2. **Create World File in SDF**
   - Define environment and physics
   - Include your robot using URDF

```xml
<!-- world_with_robot.sdf -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="robot_world">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Include robot from URDF -->
    <include>
      <name>my_robot</name>
      <pose>0 0 0.1 0 0 0</pose>
      <uri>model://my_robot_description/urdf/robot.urdf</uri>
    </include>

    <!-- Environment -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### Advanced SDF Features

#### Custom Sensors
```xml
<sensor name="custom_lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>1</samples>
        <resolution>1</resolution>
        <min_angle>0</min_angle>
        <max_angle>0</max_angle>
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <namespace>lidar</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
  </plugin>
</sensor>
```

#### Physics Materials
```xml
<collision name="wheel_collision">
  <surface>
    <friction>
      <ode>
        <mu>10.0</mu>
        <mu2>10.0</mu2>
        <slip1>0.0</slip1>
        <slip2>0.0</slip2>
      </ode>
      <torsional>
        <coefficient>1.0</coefficient>
        <use_patch_radius>false</use_patch_radius>
        <surface_radius>0.01</surface_radius>
      </torsional>
    </friction>
  </surface>
</collision>
```

## Tools for Format Management

### URDF Tools

```bash
# Validate URDF
check_urdf robot.urdf

# Convert URDF to graph for visualization
urdf_to_graphiz robot.urdf

# Generate SDF from URDF
gz sdf -p robot.urdf > robot.sdf
```

### SDF Tools

```bash
# Validate SDF
gz sdf -k world.sdf

# Pretty-print SDF
gz sdf -p world.sdf

# Check for errors
gz sdf -c world.sdf
```

### Xacro for Both Formats

Xacro can be used with both URDF and SDF:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robot_with_sensors">
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />

  <!-- Define a macro that can be used in both URDF and SDF contexts -->
  <xacro:macro name="simple_wheel" params="prefix parent x_pos y_pos z_pos">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <origin xyz="0 0 0" rpy="1.570796 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <origin xyz="0 0 0" rpy="1.570796 0 0"/>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${x_pos} ${y_pos} ${z_pos}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:simple_wheel prefix="front_left" parent="base_link" x_pos="0.2" y_pos="0.15" z_pos="0"/>
  <xacro:simple_wheel prefix="front_right" parent="base_link" x_pos="0.2" y_pos="-0.15" z_pos="0"/>
</robot>
```

## Best Practices for Format Usage

### When to Use URDF

- **Robot Description**: When defining robot kinematics and basic dynamics
- **ROS Integration**: When working primarily within the ROS ecosystem
- **Visualization**: When using RViz for robot visualization
- **Motion Planning**: When using MoveIt! or similar planning frameworks
- **Hardware Interface**: When connecting to real robots

### When to Use SDF

- **Physics Simulation**: When detailed physics simulation is required
- **Sensor Simulation**: When using complex sensor models with noise
- **World Definition**: When creating complex simulation environments
- **Plugin Integration**: When using custom simulation plugins
- **Multi-Robot Simulation**: When simulating multiple robots together

### Combined Approach

The most effective approach often combines both formats:

1. **Design Robot in URDF**: Create the basic robot model in URDF format
2. **Define World in SDF**: Create simulation environments in SDF
3. **Include URDF in SDF**: Use URDF models within SDF worlds
4. **Add SDF Extensions**: Add sensors and physics properties in the SDF context

### Example Workflow
```bash
# 1. Create and validate URDF
check_urdf robot.urdf

# 2. Use URDF in SDF world
gz sim -r robot_world.sdf

# 3. Test and refine both formats
# 4. Use Xacro to manage complexity in both formats
```

## Exercise: Format Conversion and Integration

1. Take a simple robot URDF file (like the one from Module 1)
2. Create a Gazebo world file that includes your robot
3. Add a simple sensor (camera or LiDAR) to the robot in the SDF world
4. Launch the simulation and verify the robot appears correctly
5. Experiment with different physics parameters
6. Create a launch file that starts Gazebo with your world

This exercise will help you understand the relationship between URDF and SDF and how to use both effectively.

## Troubleshooting Common Issues

### URDF to SDF Conversion Issues
- **Missing elements**: SDF may not preserve all URDF extensions
- **Coordinate system differences**: Check frame conventions
- **Inertia issues**: Verify mass and inertia values transfer correctly

### Simulation Issues
- **Physics instability**: Adjust physics parameters in SDF
- **Joint constraints**: Verify joint limits and types
- **Collision issues**: Check collision geometries and properties

### Integration Issues
- **ROS topic names**: Ensure proper namespace and topic mapping
- **Frame transforms**: Verify TF tree structure
- **Plugin loading**: Check plugin dependencies and paths

## Summary

URDF and SDF serve complementary but distinct purposes in robotics. URDF excels at robot description for ROS integration and kinematic analysis, while SDF provides comprehensive simulation capabilities for physics, sensors, and environments. Understanding both formats and how to work with them together is essential for effective robotics simulation and development.

The key is to leverage each format's strengths: use URDF for robot description and ROS integration, and SDF for simulation environments and advanced physics modeling. Together, they form a complete solution for robotics simulation and development.

In the next section, we'll explore how to simulate various sensors in digital environments, building on the format knowledge we've gained.