---
sidebar_position: 2
---

# ROS 2 Architecture: Nodes, Topics, Services, and Actions

## Learning Objectives

By the end of this section, you will be able to:
- Explain the fundamental components of ROS 2 architecture
- Distinguish between Nodes, Topics, Services, and Actions
- Understand when to use each communication pattern
- Create basic examples of each communication type

## Introduction to ROS 2 Architecture

ROS 2 (Robot Operating System 2) provides a flexible framework for writing robot software. It is a collection of libraries, tools, and conventions that enable different software components to communicate with each other. Understanding the architecture is crucial for developing effective robotic systems.

The ROS 2 architecture is built around several core concepts that work together to enable communication, coordination, and control of robotic systems. These components form the backbone of any ROS 2 application.

## Nodes

### What is a Node?

A **Node** is the fundamental unit of computation in ROS 2. It's an executable process that performs specific computations and communicates with other nodes. Think of nodes as the "cells" in the robotic nervous system - each one performs a specific function while working together as part of a larger system.

### Node Characteristics

- **Process**: Each node runs as a separate process
- **Communication**: Nodes communicate with other nodes through topics, services, and actions
- **Responsibility**: Each node typically has a single responsibility (e.g., sensor processing, control algorithm, visualization)
- **Lifecycle**: Nodes can be started, stopped, and managed independently

### Creating a Node

In Python using rclpy:

```python
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('my_robot_node')
        self.get_logger().info('MyRobotNode has been started')

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics

### What is a Topic?

A **Topic** is a named bus over which nodes exchange messages. Topics implement a publish-subscribe communication pattern where publishers send messages and subscribers receive messages. This is ideal for continuous data streams like sensor data or robot state information.

### Topic Characteristics

- **Asynchronous**: Publishers and subscribers don't need to be synchronized
- **Many-to-many**: Multiple publishers can send to a topic, and multiple subscribers can receive from it
- **Unidirectional**: Data flows in one direction (publisher → topic → subscriber)
- **Typed**: Each topic has a specific message type that defines its structure

### Example: Sensor Data Topic

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')
        self.publisher = self.create_publisher(LaserScan, 'laser_scan', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = LaserScan()
        # Fill in message data
        msg.ranges = [1.0, 2.0, 3.0]  # Example ranges
        self.publisher.publish(msg)
        self.get_logger().info('Publishing laser scan data')

class ProcessingNode(Node):
    def __init__(self):
        super().__init__('processing_node')
        self.subscription = self.create_subscription(
            LaserScan,
            'laser_scan',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received laser scan with {len(msg.ranges)} readings')
```

## Services

### What is a Service?

A **Service** provides a request-response communication pattern. A service client sends a request and waits for a response from a service server. This is ideal for operations that require immediate results, such as changing robot parameters or triggering specific actions.

### Service Characteristics

- **Synchronous**: Client waits for response from server
- **One-to-one**: One client requests from one server at a time
- **Bidirectional**: Request goes to server, response comes back to client
- **Blocking**: Client blocks until response is received (unless using async)

### Example: Navigation Service

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetBool

class NavigationServer(Node):
    def __init__(self):
        super().__init__('navigation_server')
        self.srv = self.create_service(
            SetBool,
            'enable_navigation',
            self.enable_navigation_callback)

    def enable_navigation_callback(self, request, response):
        if request.data:
            self.get_logger().info('Navigation enabled')
            response.success = True
            response.message = 'Navigation system enabled'
        else:
            self.get_logger().info('Navigation disabled')
            response.success = True
            response.message = 'Navigation system disabled'
        return response

class NavigationClient(Node):
    def __init__(self):
        super().__init__('navigation_client')
        self.cli = self.create_client(SetBool, 'enable_navigation')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = SetBool.Request()

    def send_request(self, enable):
        self.req.data = enable
        future = self.cli.call_async(self.req)
        return future
```

## Actions

### What is an Action?

An **Action** is a goal-oriented communication pattern that extends services to support long-running operations. Actions include feedback during execution and the ability to cancel ongoing operations. This is ideal for tasks like navigation to a goal or manipulation operations.

### Action Characteristics

- **Long-running**: Designed for operations that take significant time
- **Feedback**: Provides intermediate feedback during execution
- **Cancel**: Allows clients to cancel ongoing operations
- **Status**: Tracks the status of the goal (pending, active, succeeded, etc.)

### Example: Navigation Action

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
import time

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing navigation goal...')

        # Simulate navigation progress
        for i in range(10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return NavigateToPose.Result()

            # Publish feedback
            feedback_msg = NavigateToPose.Feedback()
            feedback_msg.distance_remaining = 10.0 - i
            goal_handle.publish_feedback(feedback_msg)

            time.sleep(1)  # Simulate navigation time

        goal_handle.succeed()
        result = NavigateToPose.Result()
        result.result = True
        return result

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT
```

## Communication Pattern Selection Guide

| Use Case | Communication Pattern | Reason |
|----------|----------------------|---------|
| Sensor data streaming | Topic | Continuous, asynchronous data flow |
| Robot state monitoring | Topic | Multiple subscribers need the same information |
| Parameter updates | Service | Request-response pattern, immediate result needed |
| Triggering specific actions | Service | Simple, synchronous operation |
| Navigation to goal | Action | Long-running, needs feedback and cancel capability |
| Manipulation tasks | Action | Complex, long-running operations with status |
| Control commands | Topic | Fast, continuous command updates |

## Best Practices

### Node Design
- Keep nodes focused on a single responsibility
- Use descriptive names that indicate the node's function
- Implement proper error handling and logging
- Consider the node's lifecycle and resource management

### Topic Usage
- Use appropriate message types for your data
- Set appropriate queue sizes based on your application's needs
- Consider the frequency of message publishing
- Use reliable QoS for critical data, best-effort for less critical data

### Service Design
- Keep service operations relatively quick
- Design clear request and response message structures
- Handle errors gracefully and provide meaningful error messages
- Consider if an action might be more appropriate for long-running operations

### Action Implementation
- Provide meaningful feedback during long operations
- Implement proper cancellation handling
- Set appropriate timeouts for operations
- Consider the state management for complex actions

## Exercise: Understanding Communication Patterns

Create a simple ROS 2 system with the following components:
1. A sensor node that publishes temperature data to a topic
2. A logger node that subscribes to the temperature topic and logs values
3. A service server that converts Celsius to Fahrenheit
4. A service client that requests temperature conversion
5. An action server that simulates a heating/cooling process

This exercise will help you understand the differences between the various communication patterns and when to use each one.

## Summary

ROS 2's architecture provides flexible communication patterns that enable the development of complex robotic systems. Understanding when and how to use nodes, topics, services, and actions is fundamental to creating effective robot applications. Each pattern serves specific purposes and choosing the right one for your use case is crucial for system performance and maintainability.

In the next section, we'll explore how to implement these concepts using Python and the rclpy library.