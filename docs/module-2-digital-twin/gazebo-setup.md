---
sidebar_position: 2
---

# Gazebo Setup: Physics Simulation Environment

## Learning Objectives

By the end of this section, you will be able to:
- Install and configure Gazebo simulation environment
- Understand Gazebo's physics engine and simulation capabilities
- Set up Gazebo worlds with proper physics parameters
- Integrate Gazebo with ROS 2 using Gazebo ROS packages
- Launch and control robots in Gazebo simulation

## Introduction to Gazebo

**Gazebo** is a 3D simulation environment that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in robotics research and development for testing algorithms, training AI models, and validating robot designs before deployment on physical hardware.

Gazebo's key features include:
- **Realistic Physics**: Accurate simulation of rigid body dynamics, contact forces, and collisions
- **Multiple Physics Engines**: Support for ODE, Bullet, Simbody, and DART physics engines
- **Sensor Simulation**: Comprehensive simulation of cameras, LiDAR, IMUs, GPS, and other sensors
- **Visual Quality**: High-quality rendering with shadows, lighting, and textures
- **ROS Integration**: Seamless integration with ROS and ROS 2 through Gazebo ROS packages

## Gazebo Architecture

Gazebo consists of several key components:

### 1. Gazebo Server (`gz sim` or `gazebo`)
The core simulation engine that handles physics, rendering, and plugin management.

### 2. Gazebo Client (`gz sim -g` or `gzclient`)
The graphical user interface that connects to the server to visualize the simulation.

### 3. Plugins
Dynamically loaded libraries that extend Gazebo's functionality:
- **Model Plugins**: Control robot models and their behavior
- **Sensor Plugins**: Interface with simulated sensors
- **World Plugins**: Modify world behavior and physics
- **GUI Plugins**: Extend the graphical interface

### 4. Fuel Server
An online repository for sharing simulation assets like robots, worlds, and models.

## Installing Gazebo

### Installing Gazebo Garden (Recommended)

For Ubuntu 22.04 with ROS 2 Humble:

```bash
# Add the Gazebo repository
sudo curl -sSL http://get.gazebosim.org | sh

# Install Gazebo Garden
sudo apt install gazebo-garden
```

### Installing Gazebo ROS Packages

To integrate Gazebo with ROS 2:

```bash
# Install Gazebo ROS packages
sudo apt install ros-humble-gazebo-ros-pkgs

# Install additional ROS 2 packages for simulation
sudo apt install ros-humble-gazebo-ros2-control ros-humble-gazebo-ros2-control-demos
```

## Gazebo Configuration

### Environment Variables

Gazebo uses several environment variables for configuration:

```bash
# Add Gazebo models to the model path
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH}:/path/to/your/models"

# Set the physics engine (options: ogre2, ogre, or dart)
export GZ_SIM_RENDER_ENGINE=ogre2

# Increase the maximum update rate for faster simulation
export GZ_SIM_MAX_STEP_SIZE=0.001
```

### Configuration File

Create a Gazebo configuration file at `~/.gazebo/config.yaml`:

```yaml
gazebo:
  gui:
    fullscreen: false
    width: 1280
    height: 720
  physics:
    engine: "ode"
    max_step_size: 0.001
    real_time_factor: 1.0
    real_time_update_rate: 1000.0
```

## Basic Gazebo Commands

### Starting Gazebo

```bash
# Start Gazebo with an empty world
gz sim

# Start Gazebo with a specific world file
gz sim -r -v 4 /path/to/world.sdf

# Start Gazebo with GUI
gz sim -g

# Start Gazebo with a specific world and run immediately
gz sim -r empty.sdf
```

### Useful Gazebo Commands

```bash
# List all topics
gz topic -l

# Echo a topic (e.g., clock)
gz topic -e -t /clock

# Publish to a topic
gz topic -t /cmd_vel -m ignition.msgs.Twist -p 'linear: {x: 1.0}'

# List all services
gz service -l

# List all entities
gz topic -e -t /world/default/state
```

## Creating Gazebo Worlds

### World File Structure (SDF Format)

A basic Gazebo world file in SDF (Simulation Description Format):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Physics Engine -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Include default ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include default sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add a simple box -->
    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
      </link>
    </model>
  </world>
</sdf>
```

### Advanced World Features

#### Adding Models from Fuel

```xml
<include>
  <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Double Pendulum</uri>
  <pose>0 0 0 0 0 0</pose>
</include>
```

#### Custom Physics Parameters

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Integrating with ROS 2

### Gazebo ROS Packages

The Gazebo ROS packages provide the bridge between Gazebo and ROS 2:

- **gazebo_ros**: Core ROS 2 interface to Gazebo
- **gazebo_plugins**: Common plugins for ROS 2 integration
- **gazebo_ros_pkgs**: Tools for launching and controlling Gazebo with ROS 2

### Launching Gazebo with ROS 2

Create a launch file to start Gazebo with ROS 2 integration:

```python
# launch/gazebo_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration
    world = LaunchConfiguration('world')

    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Choose one of the world files from `/gazebo_ros/worlds`'
    )

    # Gazebo server node
    gzserver_cmd = Node(
        package='ros_gz_sim',
        executable='gzserver',
        arguments=[world, '-r'],
        output='screen'
    )

    # Gazebo client node
    gzclient_cmd = Node(
        package='ros_gz_sim',
        executable='gzclient',
        output='screen'
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(world_arg)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)

    return ld
```

### Spawning Robots in Gazebo

To spawn a robot model in Gazebo:

```python
# Python script to spawn a robot
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from ros_gz_interfaces.srv import Spawn

class RobotSpawner(Node):
    def __init__(self):
        super().__init__('robot_spawner')
        self.cli = self.create_client(Spawn, '/spawn')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = Spawn.Request()

    def send_request(self, name, xml, pose, type_):
        self.req.name = name
        self.req.xml = xml
        self.req.pose = pose
        self.req.type = type_
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)

    # Read robot URDF from file
    with open('/path/to/robot.urdf', 'r') as f:
        robot_xml = f.read()

    spawner = RobotSpawner()

    # Create pose
    pose = Pose()
    pose.position.x = 0.0
    pose.position.y = 0.0
    pose.position.z = 1.0

    # Spawn the robot
    result = spawner.send_request('my_robot', robot_xml, pose, 'urdf')
    spawner.get_logger().info(f'Spawn result: {result.success}')

    spawner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Working with Models

### Model Structure

A Gazebo model typically has the following structure:

```
~/.gazebo/models/my_robot/
├── model.config          # Model metadata
├── model.sdf             # Model definition
├── meshes/               # 3D mesh files
│   ├── link1.dae
│   └── link2.stl
└── materials/
    └── textures/
        └── texture.png
```

### Model Configuration File

```xml
<?xml version="1.0"?>
<model>
  <name>My Robot</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A sample robot model</description>
</model>
```

## Sensor Simulation in Gazebo

### Camera Sensors

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
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### LiDAR Sensors

```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1.0</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

## Physics Configuration

### Understanding Physics Parameters

- **max_step_size**: Simulation time step (smaller = more accurate but slower)
- **real_time_factor**: Target simulation speed (1.0 = real-time)
- **real_time_update_rate**: Update rate in Hz
- **gravity**: Gravitational acceleration vector

### Tuning Physics for Accuracy

For accurate physics simulation:
1. Use smaller time steps (0.001 or smaller)
2. Increase solver iterations
3. Adjust constraint parameters appropriately
4. Match real-world physics parameters

## Best Practices

### 1. Performance Optimization
- Use appropriate collision geometries (simpler than visual geometries)
- Limit update rates for sensors that don't need high frequency
- Use fixed joints instead of very stiff constraints

### 2. Accuracy Considerations
- Match real-world physical properties in simulation
- Use appropriate friction and damping coefficients
- Validate simulation results against real-world data

### 3. Model Development
- Start with simple models and add complexity gradually
- Use realistic mass and inertia values
- Test individual components before integration

## Exercise: Basic Gazebo Setup

1. Install Gazebo Garden and verify the installation
2. Launch Gazebo with the empty world
3. Add a simple box model to the world
4. Save the world file and load it again
5. Use Gazebo's command-line tools to interact with the simulation

This exercise will help you become familiar with the Gazebo interface and basic operations.

## Troubleshooting Common Issues

### Gazebo Won't Start
- Check graphics drivers and OpenGL support
- Try running with software rendering: `gz sim --render-engine ogre2`
- Verify installation with: `gz --versions`

### Physics Issues
- Objects falling through surfaces: Check collision geometries and physics parameters
- Unstable simulation: Reduce time step or adjust solver parameters
- Penetration between objects: Increase physics update rate

### Performance Problems
- Slow simulation: Reduce model complexity or sensor update rates
- High CPU usage: Optimize collision meshes and reduce update frequencies

## Summary

Gazebo provides a powerful physics simulation environment for robotics development. Understanding its architecture, configuration, and integration with ROS 2 is essential for creating effective digital twins. Proper setup of physics parameters, sensors, and models ensures accurate simulation results that can be reliably transferred to real-world applications.

In the next section, we'll explore Unity as an alternative simulation platform for high-fidelity visualization and human-robot interaction.