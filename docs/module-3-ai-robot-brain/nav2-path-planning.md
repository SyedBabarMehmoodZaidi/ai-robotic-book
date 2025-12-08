---
sidebar_position: 4
---

# Nav2 Path Planning: Advanced Navigation and Autonomous Movement

## Learning Objectives

By the end of this section, you will be able to:
- Install and configure the Navigation2 (Nav2) framework for autonomous navigation
- Understand Nav2's architecture and core components for path planning
- Create and configure costmaps for obstacle avoidance and navigation
- Implement global and local path planners for different scenarios
- Integrate Nav2 with perception systems for dynamic navigation
- Configure Nav2 for multi-robot coordination and fleet management
- Validate navigation performance in simulation and real-world environments

## Introduction to Navigation2 (Nav2)

**Navigation2 (Nav2)** is the next-generation navigation framework for ROS 2, designed to provide robust, reliable, and flexible navigation capabilities for mobile robots. Building upon the success of ROS Navigation (NavFn), Nav2 introduces a more modular, plugin-based architecture that enables advanced navigation capabilities for complex robotic systems.

Nav2 addresses the evolving needs of robotics navigation:
- **Modular Architecture**: Plugin-based system for custom planners and controllers
- **Dynamic Environments**: Real-time obstacle detection and path replanning
- **Multi-Robot Coordination**: Fleet management and collision avoidance
- **Perception Integration**: Seamless integration with sensor systems
- **Behavior Trees**: Advanced decision-making for navigation behaviors
- **Safety-Critical Systems**: Built-in safety mechanisms and recovery behaviors

## Nav2 Architecture

### Core Components

Nav2 consists of several key components that work together to provide comprehensive navigation capabilities:

#### 1. Navigation Server
The central component that coordinates all navigation activities:
- Manages navigation lifecycle
- Coordinates between different navigation components
- Handles action requests from clients
- Manages navigation state and recovery behaviors

#### 2. Global Planner
Responsible for computing optimal paths from start to goal:
- A* (A-star) algorithm implementation
- Dijkstra's algorithm
- Theta* for any-angle path planning
- Custom plugin support for specialized algorithms

#### 3. Local Planner
Handles short-term path following and obstacle avoidance:
- Dynamic Window Approach (DWA)
- Timed Elastic Band (TEB)
- Trajectory Rollout
- Custom local planner plugins

#### 4. Costmap 2D
Dynamic obstacle mapping and collision checking:
- Static map layer (from SLAM or pre-built maps)
- Obstacle layer (from sensors like LiDAR, cameras)
- Inflation layer (safety margins around obstacles)
- Voxel layer (3D obstacle representation)

#### 5. Controller
Low-level trajectory execution and robot control:
- Pure pursuit controller
- PID-based controllers
- Model Predictive Control (MPC)
- Custom controller plugins

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Goal Request  │───▶│  Navigation      │───▶│   Robot         │
│   (Action)      │    │  Server          │    │   Controller    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                           │    │    │
                           ▼    ▼    ▼
                    ┌──────────────────┐
                    │  Global Planner  │───▶ Global Path
                    └──────────────────┘
                           │
                    ┌──────────────────┐
                    │  Local Planner   │───▶ Local Trajectory
                    └──────────────────┘
                           │
                    ┌──────────────────┐
                    │   Costmaps       │◀─── Sensor Data
                    │ (Global/Local)   │
                    └──────────────────┘
```

## Installing Nav2

### Prerequisites

Before installing Nav2, ensure you have:
- ROS 2 Humble Hawksbill installed
- Basic understanding of ROS 2 concepts (topics, services, actions)
- A working robot platform or simulation environment
- Sensor data (LiDAR, cameras, IMU) for navigation

### Installation Methods

#### Method 1: Binary Installation (Recommended)

```bash
# Install Nav2 packages
sudo apt update
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
sudo apt install ros-humble-nav2-rviz-plugins
sudo apt install ros-humble-nav2-msgs
sudo apt install ros-humble-nav2-map-server
sudo apt install ros-humble-nav2-utils
sudo apt install ros-humble-nav2-controllers
sudo apt install ros-humble-nav2-planners
sudo apt install ros-humble-nav2-behaviors
sudo apt install ros-humble-nav2-lifecycle-manager
sudo apt install ros-humble-nav2-amcl
```

#### Method 2: Source Installation (For Development)

```bash
# Create a new workspace
mkdir -p ~/nav2_ws/src
cd ~/nav2_ws/src

# Clone Nav2 repositories
git clone -b humble https://github.com/ros-planning/navigation2.git
cd navigation2
git checkout humble

# Install dependencies
cd ~/nav2_ws
rosdep install -i --from-path src --rosdistro humble -y

# Build Nav2
colcon build --packages-select navigation2
source install/setup.bash
```

## Nav2 Configuration

### Basic Configuration Files

Nav2 uses YAML configuration files to define navigation parameters. Here's a comprehensive configuration:

#### 1. Main Nav2 Configuration (`nav2_params.yaml`)

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: $(find my_robot_bringup)/behavior_trees/navigate_w_replanning_and_recovery.xml
    default_nav_to_pose_bt_xml: $(find my_robot_bringup)/behavior_trees/navigate_w_replanning_and_recovery.xml
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_smooth_path_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_drive_on_heading_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_have_remaining_waypoints_condition_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_truncate_path_local_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node
    - nav2_spin_cancel_bt_node
    - nav2_back_up_cancel_bt_node
    - nav2_drive_on_heading_cancel_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # Controller parameters
    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      progress_checker_plugin: "progress_checker"
      goal_checker_plugin: "goal_checker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        map_subscribe_transient_local: True
      always_send_full_costmap: True
  local_costmap_client:
    ros__parameters:
      use_sim_time: True
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.22
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True
  global_costmap_client:
    ros__parameters:
      use_sim_time: True
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

map_server:
  ros__parameters:
    use_sim_time: True
    yaml_filename: "turtlebot3_world.yaml"

map_saver:
  ros__parameters:
    use_sim_time: True
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

smoother_server:
  ros__parameters:
    use_sim_time: True
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: true

behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "wait", "drive_on_heading"]
    spin:
      plugin: "nav2_behaviors/Spin"
      spin_dist: 1.57
    backup:
      plugin: "nav2_behaviors/BackUp"
      backup_dist: 0.15
      backup_speed: 0.025
    wait:
      plugin: "nav2_behaviors/Wait"
      wait_duration: 1.0
    drive_on_heading:
      plugin: "nav2_behaviors/DriveOnHeading"
      drive_on_heading_angle_tol: 0.785
      drive_on_heading_dist_tol: 0.25
      drive_on_heading_forward_dist: 0.3
      drive_on_heading_max_duration: 15.0

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      wait_time: 1.0
```

### Behavior Trees Configuration

Nav2 uses Behavior Trees (BT) for navigation decision-making. Here's an example behavior tree configuration:

```xml
<!-- behavior_trees/navigate_w_replanning_and_recovery.xml -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="6" name="NavigateRecovery">
      <PipelineSequence name="NavigateWithReplanning">
        <RateController hz="1.0">
          <RecoveryNode number_of_retries="1" name="ComputePathToPose">
            <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
            <RecoveryNode number_of_retries="1" name="SmoothPath">
              <SmoothPath path="{path}" smoothed_path="{path}" smoother_id="simple_smoother"/>
            </RecoveryNode>
          </RecoveryNode>
        </RateController>
        <RecoveryNode number_of_retries="2" name="FollowPath">
          <FollowPath path="{path}" controller_id="FollowPath"/>
        </RecoveryNode>
      </PipelineSequence>
      <RecoveryNode number_of_retries="2" name="RecoveryFallback">
        <ClearEntireCostmap name="ClearGlobalCostmap-Context" service_name="global_costmap/clear_entirely_global_costmap"/>
        <ClearEntireCostmap name="ClearLocalCostmap-Context" service_name="local_costmap/clear_entirely_local_costmap"/>
        <Spin spin_dist="1.57" name="Spin"/>
      </RecoveryNode>
    </RecoveryNode>
  </BehaviorTree>
</root>
```

## Launching Nav2

### Basic Nav2 Launch File

```python
# launch/nav2_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_bringup'),
            'config', 'nav2_params.yaml'
        ]),
        description='Full path to params file to load'
    )

    map_file = DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_bringup'),
            'maps', 'turtlebot3_world.yaml'
        ]),
        description='Full path to map file to load'
    )

    # Map server
    map_server_cmd = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )

    # Lifecycle manager
    lifecycle_manager_cmd = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                    {'autostart': True},
                    {'node_names': ['map_server',
                                   'planner_server',
                                   'controller_server',
                                   'bt_navigator',
                                   'amcl']}]
    )

    # AMCL
    amcl_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_amcl'),
                'launch', 'amcl.launch.py'
            ])
        ]),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time'),
                         'params_file': LaunchConfiguration('params_file')}.items()
    )

    # Planner server
    planner_server_cmd = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )

    # Controller server
    controller_server_cmd = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )

    # BT Navigator
    bt_navigator_cmd = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )

    # Waypoint follower
    waypoint_follower_cmd = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )

    return LaunchDescription([
        use_sim_time,
        params_file,
        map_server_cmd,
        lifecycle_manager_cmd,
        amcl_cmd,
        planner_server_cmd,
        controller_server_cmd,
        bt_navigator_cmd,
        waypoint_follower_cmd
    ])
```

## Costmap Configuration

### Understanding Costmaps

Costmaps are essential for navigation, representing the environment as a grid of cells with associated costs:

- **Static Layer**: Represents permanent obstacles from the map
- **Obstacle Layer**: Represents dynamic obstacles from sensors
- **Inflation Layer**: Adds safety margins around obstacles
- **Voxel Layer**: 3D representation for height-based obstacle detection

### Costmap Parameters

```yaml
# Costmap parameters
costmap_parameters:
  # Global costmap (for path planning)
  global_costmap:
    plugins:
      - {name: static_layer, type: "nav2_costmap_2d::StaticLayer"}
      - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
      - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}

    # Static layer parameters
    static_layer:
      map_subscribe_transient_local: True

    # Obstacle layer parameters
    obstacle_layer:
      enabled: True
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_max_range: 3.0
        raytrace_min_range: 0.0
        obstacle_max_range: 2.5
        obstacle_min_range: 0.0

    # Inflation layer parameters
    inflation_layer:
      cost_scaling_factor: 3.0
      inflation_radius: 0.55

  # Local costmap (for obstacle avoidance)
  local_costmap:
    plugins:
      - {name: voxel_layer, type: "nav2_costmap_2d::VoxelLayer"}
      - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}

    # Voxel layer parameters
    voxel_layer:
      enabled: True
      publish_voxel_map: True
      origin_z: 0.0
      z_resolution: 0.05
      z_voxels: 16
      max_obstacle_height: 2.0
      mark_threshold: 0
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_max_range: 3.0
        raytrace_min_range: 0.0
        obstacle_max_range: 2.5
        obstacle_min_range: 0.0
```

## Path Planning Algorithms

### Global Planners

Nav2 supports several global path planning algorithms:

#### 1. Navfn Planner (A* based)

```yaml
planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```

#### 2. Smac Planner (State Lattice based)

```yaml
planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_smac_planner/SmacPlanner"
      tolerance: 0.5
      downsample_costmap: false
      downsampling_factor: 1
      allow_unknown: true
      max_iterations: 1000000
      motion_model_for_search: "DUBIN"
```

#### 3. Smoother

```yaml
smoother_server:
  ros__parameters:
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: true
```

### Local Planners

#### 1. DWA Local Planner

```yaml
controller_server:
  ros__parameters:
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "nav2_dwb_controller/DWBLocalPlanner"
      debug_trajectory_details: True
      min_vel_x: 0.0
      min_vel_y: 0.0
      max_vel_x: 0.26
      max_vel_y: 0.0
      max_vel_theta: 1.0
      min_vel_theta: -1.0
      acc_lim_x: 2.5
      acc_lim_y: 0.0
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_y: 0.0
      decel_lim_theta: -3.2
      vx_samples: 20
      vy_samples: 5
      vtheta_samples: 20
      sim_time: 1.7
      linear_granularity: 0.05
      angular_granularity: 0.025
      transform_tolerance: 0.2
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.15
      stateful: True
      oscillation_reset_dist: 0.05
      forward_point_distance: 0.325
      scaling_speed: 0.25
      max_scaling_factor: 0.2
```

#### 2. TEB Local Planner

```yaml
controller_server:
  ros__parameters:
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "nav2_te_babbling/TEBLocalPlanner"
      max_vel_x: 0.4
      max_vel_x_backwards: 0.2
      max_vel_theta: 0.3
      acc_lim_x: 0.5
      acc_lim_theta: 0.5
      xy_goal_tolerance: 0.2
      yaw_goal_tolerance: 0.1
      global_plan_overwrite_orientation: True
      global_plan_viapoint_sep: -1
      feasibility_check_no_poses: 5
      publish_feedback: False
      min_samples: 3
      max_samples: 500
      optimization: True
      no_inner_iterations: 5
      no_outer_iterations: 4
      optimization_verbose: False
      penalty_epsilon: 0.1
      weight_max_vel_x: 2
      weight_max_vel_theta: 1
      weight_acc_lim_x: 1
      weight_acc_lim_theta: 1
      weight_kinematics_nh: 1000
      weight_kinematics_forward_drive: 1
      weight_kinematics_turning_radius: 1
      weight_optimaltime: 1
      weight_shortest_path: 0
      weight_obstacle: 50
      weight_inflation: 0.1
      weight_dynamic_obstacle: 10
      weight_dynamic_obstacle_inflation: 0.1
      weight_viapoint: 1
      weight_prefer_viapoint: 0
      weight_adapt_factor: 2
      obstacle_poses_affected: 30
      dynamic_obstacle_inflation_dist: 0.6
      include_dynamic_obstacles: True
      include_costmap_obstacles: True
      costmap_obstacles_behind_robot_dist: 1.5
      obstacle_association_force_inclusion_factor: 1.5
      obstacle_association_cutoff_factor: 5
      costmap_converter_plugin: ""
      costmap_converter_spin_thread: True
      costmap_converter_rate: 5
      enable_homotopy_class_planning: False
      enable_multithreading: True
      simple_exploration: False
      max_number_classes: 5
      selection_cost_hysteresis: 1.0
      selection_prefer_initial_plan: 0.95
      selection_obst_cost_scale: 1.0
      selection_alternative_time_cost: False
      roadmap_graph_no_samples: 15
      roadmap_graph_area_width: 5
      roadmap_graph_area_length_scale: 1.0
      h_signature_prescaler: 0.5
      h_signature_threshold: 0.1
      obstacle_heading_threshold: 0.45
      visualize_hc_graph: False
      visualize_with_time_as_z_axis_scale: 0.0
      publish_cost_grid_pc: False
      global_plan_overwrite_orientation: True
      allow_init_with_path_guess: True
      free_goal_vel: False
      publish_traj_pc: True
      publish_transformed_plan: False
      trajectory_step_size: 0.5
      trajectory_lookahead_sampler_dist: 0.5
      exact_arc_length: False
      force_reinit_new_goal_dist: 1.0
      force_penalty_initial_weight: 0.0
      include_unknow_in_costmap: True
      costmap_converter_id: 0
      costmap_converter_max_desired_linear_vel: 0.3
      costmap_converter_max_desired_angular_vel: 0.3
      costmap_converter_rate: 5.0
```

## Nav2 with Isaac ROS Integration

### Perception-Enhanced Navigation

```python
# nodes/perception_aware_navigation.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np
from scipy.spatial import distance

class PerceptionAwareNavigation(Node):
    def __init__(self):
        super().__init__('perception_aware_navigation')

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Perception data
        self.perception_sub = self.create_subscription(
            PointCloud2, 'perception_points', self.perception_callback, 10)

        # Store perceived obstacles
        self.obstacles = []

        # Navigation parameters
        self.safety_margin = 0.5  # meters
        self.min_path_distance = 1.0  # meters

    def perception_callback(self, msg):
        """Process perception data for navigation planning"""
        # Convert PointCloud2 to numpy array (simplified)
        # In practice, use pcl_ros or similar for efficient conversion
        self.obstacles = self.extract_obstacles_from_pointcloud(msg)

    def extract_obstacles_from_pointcloud(self, pointcloud_msg):
        """Extract obstacle positions from point cloud"""
        # This is a simplified implementation
        # In practice, use proper point cloud processing
        obstacles = []
        # Process point cloud data here
        return obstacles

    def navigate_with_obstacle_avoidance(self, target_pose):
        """Navigate with dynamic obstacle avoidance"""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Navigation server not available")
            return

        # Check if target is blocked by obstacles
        if self.is_path_blocked(target_pose):
            self.get_logger().warn("Path to target is blocked by obstacles")
            # Implement obstacle avoidance logic
            safe_pose = self.find_safe_alternative_pose(target_pose)
            if safe_pose:
                self.navigate_to_pose(safe_pose)
            else:
                self.get_logger().error("No safe alternative path found")
                return
        else:
            self.navigate_to_pose(target_pose)

    def is_path_blocked(self, target_pose):
        """Check if path to target is blocked by obstacles"""
        # Calculate path and check for obstacles
        # This is a simplified implementation
        for obstacle in self.obstacles:
            dist_to_target = distance.euclidean(
                [obstacle.x, obstacle.y],
                [target_pose.pose.position.x, target_pose.pose.position.y]
            )
            if dist_to_target < self.safety_margin:
                return True
        return False

    def find_safe_alternative_pose(self, original_pose):
        """Find a safe alternative pose near the original target"""
        # Generate alternative poses in a circle around the target
        for angle in np.linspace(0, 2*np.pi, 8):
            offset_x = self.safety_margin * np.cos(angle)
            offset_y = self.safety_margin * np.sin(angle)

            alternative_pose = PoseStamped()
            alternative_pose.header.frame_id = 'map'
            alternative_pose.pose.position.x = original_pose.pose.position.x + offset_x
            alternative_pose.pose.position.y = original_pose.pose.position.y + offset_y
            alternative_pose.pose.position.z = original_pose.pose.position.z
            alternative_pose.pose.orientation = original_pose.pose.orientation

            # Check if this alternative is safe
            if not self.is_obstacle_near_pose(alternative_pose.pose):
                return alternative_pose

        return None

    def is_obstacle_near_pose(self, pose):
        """Check if there are obstacles near a given pose"""
        for obstacle in self.obstacles:
            dist = distance.euclidean(
                [obstacle.x, obstacle.y],
                [pose.position.x, pose.position.y]
            )
            if dist < self.safety_margin:
                return True
        return False

    def navigate_to_pose(self, pose_stamped):
        """Send navigation goal to Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped

        self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

    def feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        self.get_logger().info(f'Navigation progress: {feedback_msg.feedback.current_distance}')

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionAwareNavigation()

    # Example: Navigate to a specific pose
    target_pose = PoseStamped()
    target_pose.header.frame_id = 'map'
    target_pose.pose.position.x = 5.0
    target_pose.pose.position.y = 5.0
    target_pose.pose.position.z = 0.0
    target_pose.pose.orientation.w = 1.0

    node.navigate_with_obstacle_avoidance(target_pose)

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

## Multi-Robot Navigation

### Fleet Management Configuration

```yaml
# Multi-robot navigation configuration
multi_robot_nav:
  # Robot-specific namespaces
  robot1:
    nav2_params:
      use_sim_time: true
      amcl:
        ros__parameters:
          base_frame_id: "robot1/base_footprint"
          global_frame_id: "map"
          odom_frame_id: "robot1/odom"
      global_costmap:
        ros__parameters:
          global_frame: "map"
          robot_base_frame: "robot1/base_link"
      local_costmap:
        ros__parameters:
          global_frame: "robot1/odom"
          robot_base_frame: "robot1/base_link"

  robot2:
    nav2_params:
      use_sim_time: true
      amcl:
        ros__parameters:
          base_frame_id: "robot2/base_footprint"
          global_frame_id: "map"
          odom_frame_id: "robot2/odom"
      global_costmap:
        ros__parameters:
          global_frame: "map"
          robot_base_frame: "robot2/base_link"
      local_costmap:
        ros__parameters:
          global_frame: "robot2/odom"
          robot_base_frame: "robot2/base_link"
```

## Performance Optimization

### Navigation Performance Tuning

```python
# utils/nav_performance_monitor.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import time

class NavigationPerformanceMonitor(Node):
    def __init__(self):
        super().__init__('nav_performance_monitor')

        self.path_sub = self.create_subscription(
            Path, 'global_plan', self.path_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, 'goal_pose', self.goal_callback, 10)

        self.path_start_time = None
        self.path_calculation_times = []

    def path_callback(self, msg):
        """Monitor path calculation performance"""
        if self.path_start_time:
            calculation_time = time.time() - self.path_start_time
            self.path_calculation_times.append(calculation_time)

            # Log performance metrics
            avg_time = sum(self.path_calculation_times) / len(self.path_calculation_times)
            self.get_logger().info(f'Path calculation time: {calculation_time:.3f}s, avg: {avg_time:.3f}s')

            self.path_start_time = None

    def goal_callback(self, msg):
        """Record path calculation start time"""
        self.path_start_time = time.time()

def main(args=None):
    rclpy.init(args=args)
    node = NavigationPerformanceMonitor()

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

## Troubleshooting Nav2

### Common Issues and Solutions

#### 1. Navigation Timeout Issues
```bash
# Check navigation parameters
ros2 param list | grep timeout

# Increase timeout values
ros2 param set /bt_navigator default_server_timeout 60
```

#### 2. Costmap Not Updating
```bash
# Check costmap topics
ros2 topic echo /global_costmap/costmap --field data

# Verify sensor data
ros2 topic echo /scan
```

#### 3. Robot Not Following Path
```bash
# Check controller status
ros2 service call /controller_server/load_node std_srvs/srv/Trigger

# Verify robot controller connection
ros2 topic echo /cmd_vel
```

#### 4. AMCL Localization Issues
```bash
# Check AMCL status
ros2 service call /amcl/transition_lifecycle std_srvs/srv/Trigger

# Reset AMCL pose
ros2 topic pub /initialpose geometry_msgs/PoseWithCovarianceStamped "..."
```

## Exercise: Nav2 Path Planning Implementation

1. Install Nav2 in your ROS 2 workspace
2. Create a map of your environment using SLAM or manually
3. Configure Nav2 with appropriate parameters for your robot
4. Launch Nav2 with your robot in simulation
5. Send navigation goals and observe the robot's behavior
6. Modify costmap parameters to improve obstacle avoidance
7. Test navigation in various scenarios (narrow passages, dynamic obstacles)
8. Implement a simple obstacle avoidance node that integrates with Nav2

This exercise will give you hands-on experience with autonomous navigation systems.

## Summary

Navigation2 provides a comprehensive, modular framework for autonomous robot navigation with advanced capabilities for path planning, obstacle avoidance, and multi-robot coordination. Its plugin-based architecture allows for customization to specific robot platforms and application requirements. Proper configuration of costmaps, planners, and controllers is essential for robust navigation performance.

In the next section, we'll explore reinforcement learning techniques for bipedal locomotion control.