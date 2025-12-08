---
sidebar_position: 4
---

# URDF for Humanoid Robots

## Learning Objectives

By the end of this section, you will be able to:
- Understand the structure and components of URDF files
- Create URDF descriptions for humanoid robots
- Define joints, links, and physical properties
- Use Xacro to simplify complex URDF files
- Visualize and validate URDF models

## Introduction to URDF

**URDF (Unified Robot Description Format)** is an XML-based format used to describe robot models in ROS. It defines the physical and visual properties of a robot, including its links (rigid parts), joints (connections between links), and inertial properties. For humanoid robots, URDF is essential for simulation, visualization, and control.

URDF stands for "Unified Robot Description Format" and is used throughout the ROS ecosystem for:
- Robot simulation in Gazebo
- Robot visualization in RViz
- Kinematic analysis
- Motion planning
- Robot state publishing

## URDF Structure

A basic URDF file consists of:
- **Links**: Rigid parts of the robot (e.g., torso, arms, legs)
- **Joints**: Connections between links (e.g., hinges, prismatic joints)
- **Visual**: How the robot appears in simulation
- **Collision**: How the robot interacts with the environment
- **Inertial**: Physical properties for simulation

### Basic URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
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

  <!-- Child link connected by a joint -->
  <link name="upper_body">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="upper_body"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>
</robot>
```

## Links in URDF

### Link Components

A link in URDF defines a rigid body part of the robot. Each link can contain:

1. **Visual**: How the link appears in visualization
2. **Collision**: How the link interacts with the environment in simulation
3. **Inertial**: Physical properties for dynamics simulation

### Visual Properties

```xml
<link name="link_name">
  <visual>
    <!-- Position and orientation offset -->
    <origin xyz="1 0 0" rpy="0 0 0"/>

    <!-- Geometry definition -->
    <geometry>
      <!-- Box: width, depth, height -->
      <box size="0.1 0.2 0.3"/>

      <!-- Cylinder: radius, length -->
      <!-- <cylinder radius="0.1" length="0.5"/> -->

      <!-- Sphere: radius -->
      <!-- <sphere radius="0.1"/> -->

      <!-- Mesh: external file -->
      <!-- <mesh filename="package://my_robot/meshes/link_name.dae" scale="1 1 1"/> -->
    </geometry>

    <!-- Material -->
    <material name="red">
      <color rgba="0.8 0.0 0.0 1.0"/>
    </material>
  </visual>
</link>
```

### Collision Properties

```xml
<collision>
  <!-- Similar to visual but for collision detection -->
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.1 0.2 0.3"/>
  </geometry>
</collision>
```

### Inertial Properties

```xml
<inertial>
  <!-- Mass in kg -->
  <mass value="1.0"/>

  <!-- Inertia matrix -->
  <inertia
    ixx="0.01" ixy="0.0" ixz="0.0"
    iyy="0.01" iyz="0.0"
    izz="0.01"/>
</inertial>
```

## Joints in URDF

Joints define the connection between links and specify how they can move relative to each other.

### Joint Types

1. **Fixed**: No movement between links
2. **Revolute**: Rotational movement around an axis (like a hinge)
3. **Continuous**: Like revolute but unlimited rotation
4. **Prismatic**: Linear sliding movement
5. **Planar**: Movement in a plane
6. **Floating**: 6DOF movement (rarely used)

### Joint Definition

```xml
<joint name="joint_name" type="revolute">
  <!-- Parent and child links -->
  <parent link="parent_link_name"/>
  <child link="child_link_name"/>

  <!-- Position and orientation of joint -->
  <origin xyz="0 0 0.1" rpy="0 0 0"/>

  <!-- Axis of rotation/translation -->
  <axis xyz="0 0 1"/>

  <!-- Joint limits (for revolute/prismatic) -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>

  <!-- Joint properties -->
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## Humanoid Robot URDF Example

Here's a simplified example of a humanoid robot with a torso, head, two arms, and two legs:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Torso (base link) -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="arm_color">
        <color rgba="0.4 0.4 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="1"/>
  </joint>

  <!-- Right Arm -->
  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="arm_color"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.2 -0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="1"/>
  </joint>

  <!-- Left Leg -->
  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="leg_color">
        <color rgba="0.2 0.6 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
    <origin xyz="-0.1 0.1 -0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="30" velocity="1"/>
  </joint>

  <!-- Right Leg -->
  <link name="right_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="leg_color"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.1 -0.1 -0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="30" velocity="1"/>
  </joint>
</robot>
```

## Using Xacro for Complex URDFs

**Xacro (XML Macros)** is a macro language that simplifies complex URDF files by allowing:
- Variables and constants
- Macros for repeated structures
- Mathematical expressions
- File inclusion

### Xacro Example

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">
  <!-- Define constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_height" value="0.5" />
  <xacro:property name="torso_radius" value="0.15" />

  <!-- Macro for arm links -->
  <xacro:macro name="arm_link" params="name side position">
    <link name="${side}_${name}">
      <visual>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 0.8 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.5"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>

    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="torso"/>
      <child link="${side}_${name}"/>
      <origin xyz="0.2 ${position} 0.1" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-${M_PI/2}" upper="${M_PI/2}" effort="20" velocity="1"/>
    </joint>
  </xacro:macro>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <cylinder length="${torso_height}" radius="${torso_radius}"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${torso_height}" radius="${torso_radius}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Use the macro to create arms -->
  <xacro:arm_link name="upper_arm" side="left" position="0.1"/>
  <xacro:arm_link name="upper_arm" side="right" position="-0.1"/>
</robot>
```

## URDF Validation and Visualization

### Checking URDF Files

```bash
# Validate URDF syntax
check_urdf /path/to/robot.urdf

# Print URDF information
urdf_to_graphiz /path/to/robot.urdf
```

### Visualizing URDF in RViz

```bash
# Launch RViz with robot state publisher
ros2 run rviz2 rviz2

# Add RobotModel display and set topic to /robot_description
```

### Loading URDF in Simulation

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from sensor_msgs.msg import JointState
import math

class StatePublisher(Node):
    def __init__(self):
        super().__init__('state_publisher')
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.publish_joint_states)

    def publish_joint_states(self):
        # Create message
        msg = JointState()
        msg.name = ['left_shoulder_joint', 'right_shoulder_joint']
        msg.position = [math.sin(self.get_clock().now().nanoseconds * 1e-9),
                        math.cos(self.get_clock().now().nanoseconds * 1e-9)]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = StatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Humanoid URDF

### 1. Proper Mass and Inertia Values

Accurate mass and inertia values are crucial for realistic simulation:

```xml
<!-- Use proper calculations for complex shapes -->
<inertial>
  <mass value="1.234"/>
  <!-- For a cylinder: Ixx = Iyy = (1/12)*m*(3*r² + h²), Izz = (1/2)*m*r² -->
  <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
</inertial>
```

### 2. Appropriate Joint Limits

Set realistic joint limits based on human anatomy or robot design:

```xml
<!-- Human shoulder joint limits -->
<joint name="left_shoulder_joint" type="revolute">
  <limit lower="${-90*M_PI/180}" upper="${90*M_PI/180}" effort="50" velocity="2"/>
</joint>
```

### 3. Collision Avoidance

Consider potential collisions between robot parts:

```xml
<!-- Use smaller collision geometries to avoid self-collision issues -->
<collision>
  <geometry>
    <capsule radius="0.04" length="0.25"/>
  </geometry>
</collision>
```

### 4. Consistent Naming

Use consistent and descriptive naming conventions:

```xml
<!-- Good naming conventions -->
<link name="left_upper_arm"/>
<link name="right_lower_leg"/>
<joint name="left_elbow_joint"/>
<joint name="right_knee_joint"/>
```

## Exercise: Create Your Own Humanoid URDF

Create a URDF file for a simple humanoid robot with:
1. A torso, head, and neck joint
2. Two arms with shoulder and elbow joints
3. Two legs with hip and knee joints
4. Proper mass, inertia, and visual properties
5. Use Xacro to simplify repetitive elements

Validate your URDF file and visualize it in RViz to ensure it's properly structured.

## Common URDF Issues and Solutions

### 1. Self-Collision Problems
- **Issue**: Robot parts collide with each other in simulation
- **Solution**: Adjust collision geometries or add self-collision checking parameters

### 2. Kinematic Chain Issues
- **Issue**: Robot has disconnected parts or multiple base links
- **Solution**: Ensure all links are connected through joints in a proper tree structure

### 3. Inertia Problems
- **Issue**: Robot behaves unrealistically in simulation
- **Solution**: Verify mass and inertia values, ensure they're physically realistic

## Summary

URDF is a fundamental component of ROS robotics, especially for humanoid robots. It provides a standardized way to describe robot geometry, kinematics, and dynamics. Understanding URDF structure, proper use of links and joints, and techniques like Xacro for complex models is essential for effective robot simulation and control.

In the next section, we'll explore ROS 2 packages, launch files, and parameter management.