---
sidebar_position: 3
---

# Python Agent Integration with ROS Controllers using rclpy

## Learning Objectives

By the end of this section, you will be able to:
- Use rclpy to create ROS 2 nodes in Python
- Implement publishers, subscribers, services, and actions using Python
- Create Python agents that interact with ROS controllers
- Handle parameters and configuration in Python nodes
- Debug and test Python-based ROS nodes

## Introduction to rclpy

**rclpy** is the Python client library for ROS 2. It provides Python bindings for the ROS 2 client library (rcl), allowing you to create ROS 2 nodes, publish and subscribe to topics, provide and use services, and create and handle actions using Python.

Python is an excellent choice for ROS 2 development due to its simplicity, extensive libraries, and rapid prototyping capabilities. It's particularly well-suited for:
- Prototyping and testing robot algorithms
- Data processing and analysis
- Simulation and visualization
- High-level robot behaviors and decision-making

## Setting Up Your Python Environment

### Virtual Environment Setup

It's recommended to use a virtual environment for your ROS 2 Python projects:

```bash
# Create a virtual environment
python3 -m venv ~/ros2_python_env

# Activate the environment
source ~/ros2_python_env/bin/activate

# Install additional Python packages for robotics
pip install numpy scipy matplotlib pandas jupyter
```

### ROS 2 Python Dependencies

Most ROS 2 Python dependencies are installed with the ROS 2 distribution. Verify your installation:

```bash
# Check if rclpy is available
python3 -c "import rclpy; print('rclpy version:', rclpy.__version__)"
```

## Creating Your First Python Node

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Implementing Publishers and Subscribers

### Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
import random

class RobotControllerPublisher(Node):
    def __init__(self):
        super().__init__('robot_controller_publisher')

        # Create multiple publishers for different robot data
        self.velocity_publisher = self.create_publisher(
            Int32,
            'robot_velocity',
            10
        )
        self.status_publisher = self.create_publisher(
            String,
            'robot_status',
            10
        )

        # Timer to publish data at regular intervals
        self.timer = self.create_timer(0.1, self.publish_robot_data)
        self.get_logger().info('Robot controller publisher started')

    def publish_robot_data(self):
        # Publish velocity data
        velocity_msg = Int32()
        velocity_msg.data = random.randint(-100, 100)  # Random velocity
        self.velocity_publisher.publish(velocity_msg)

        # Publish status data
        status_msg = String()
        status_msg.data = 'Operating' if velocity_msg.data != 0 else 'Idle'
        self.status_publisher.publish(status_msg)

        self.get_logger().debug(f'Published velocity: {velocity_msg.data}, status: {status_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    publisher = RobotControllerPublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info('Interrupted by user')
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

class RobotControllerSubscriber(Node):
    def __init__(self):
        super().__init__('robot_controller_subscriber')

        # Create subscribers for different robot data
        self.velocity_subscriber = self.create_subscription(
            Int32,
            'robot_velocity',
            self.velocity_callback,
            10
        )
        self.status_subscriber = self.create_subscription(
            String,
            'robot_status',
            self.status_callback,
            10
        )

        # Store latest values
        self.latest_velocity = 0
        self.latest_status = 'Unknown'

        self.get_logger().info('Robot controller subscriber started')

    def velocity_callback(self, msg):
        self.latest_velocity = msg.data
        self.get_logger().debug(f'Received velocity: {self.latest_velocity}')

        # Process velocity data
        if abs(self.latest_velocity) > 50:
            self.get_logger().warn('High velocity detected!')
        elif self.latest_velocity == 0:
            self.get_logger().info('Robot is stationary')

    def status_callback(self, msg):
        self.latest_status = msg.data
        self.get_logger().debug(f'Received status: {self.latest_status}')

def main(args=None):
    rclpy.init(args=args)
    subscriber = RobotControllerSubscriber()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        subscriber.get_logger().info('Interrupted by user')
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Implementing Services

### Service Server

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetBool, Trigger

class RobotServiceServer(Node):
    def __init__(self):
        super().__init__('robot_service_server')

        # Create service for enabling/disabling robot
        self.enable_service = self.create_service(
            SetBool,
            'enable_robot',
            self.enable_robot_callback
        )

        # Create service for triggering robot actions
        self.trigger_service = self.create_service(
            Trigger,
            'trigger_action',
            self.trigger_action_callback
        )

        self.robot_enabled = False
        self.get_logger().info('Robot service server started')

    def enable_robot_callback(self, request, response):
        self.robot_enabled = request.data

        if self.robot_enabled:
            self.get_logger().info('Robot enabled')
            response.success = True
            response.message = 'Robot enabled successfully'
        else:
            self.get_logger().info('Robot disabled')
            response.success = True
            response.message = 'Robot disabled successfully'

        return response

    def trigger_action_callback(self, request, response):
        if not self.robot_enabled:
            response.success = False
            response.message = 'Robot is not enabled'
            return response

        # Simulate robot action
        self.get_logger().info('Triggering robot action')
        # Perform the actual action here
        response.success = True
        response.message = 'Action completed successfully'

        return response

def main(args=None):
    rclpy.init(args=args)
    service_server = RobotServiceServer()

    try:
        rclpy.spin(service_server)
    except KeyboardInterrupt:
        service_server.get_logger().info('Interrupted by user')
    finally:
        service_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetBool, Trigger

class RobotServiceClient(Node):
    def __init__(self):
        super().__init__('robot_service_client')

        # Create clients for the services
        self.enable_client = self.create_client(SetBool, 'enable_robot')
        self.trigger_client = self.create_client(Trigger, 'trigger_action')

        while not self.enable_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Enable service not available, waiting again...')

        while not self.trigger_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Trigger service not available, waiting again...')

        self.get_logger().info('Service clients ready')

    def enable_robot(self, enable):
        request = SetBool.Request()
        request.data = enable

        future = self.enable_client.call_async(request)
        return future

    def trigger_action(self):
        request = Trigger.Request()

        future = self.trigger_client.call_async(request)
        return future

def main(args=None):
    rclpy.init(args=args)
    client = RobotServiceClient()

    # Example usage
    import time

    # Enable robot
    future = client.enable_robot(True)
    rclpy.spin_until_future_complete(client, future)
    response = future.result()
    print(f'Enable response: {response.success}, {response.message}')

    time.sleep(1)

    # Trigger action
    future = client.trigger_action()
    rclpy.spin_until_future_complete(client, future)
    response = future.result()
    print(f'Trigger response: {response.success}, {response.message}')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Implementing Actions

### Action Server

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class RobotActionServer(Node):
    def __init__(self):
        super().__init__('robot_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'robot_fibonacci',
            self.execute_callback,
            cancel_callback=self.cancel_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing robot fibonacci goal...')

        # Initialize result
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')

            # Simulate processing time
            from time import sleep
            sleep(0.5)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Result: {result.sequence}')

        return result

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

def main(args=None):
    rclpy.init(args=args)
    action_server = RobotActionServer()

    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        action_server.get_logger().info('Interrupted by user')
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Client

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class RobotActionClient(Node):
    def __init__(self):
        super().__init__('robot_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'robot_fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

def main(args=None):
    rclpy.init(args=args)
    action_client = RobotActionClient()

    # Send goal
    action_client.send_goal(10)

    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        action_client.get_logger().info('Interrupted by user')
    finally:
        action_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Working with Parameters

### Parameter Declaration and Usage

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'my_robot')
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

## Advanced Python Patterns

### Node Composition

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(String, 'cmd_vel', 10)
        self.status_publisher = self.create_publisher(String, 'status', 10)

        # Subscribers
        self.sensor_subscriber = self.create_subscription(
            String, 'sensor_data', self.sensor_callback, 10)

        # Timers
        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.status_timer = self.create_timer(1.0, self.status_update)

        # Internal state
        self.current_state = 'idle'
        self.target_velocity = 0.0
        self.sensor_data = None

        self.get_logger().info('Robot controller initialized')

    def sensor_callback(self, msg):
        self.sensor_data = msg.data
        self.get_logger().debug(f'Received sensor data: {self.sensor_data}')

    def control_loop(self):
        # Implement your control algorithm here
        if self.current_state == 'moving':
            cmd_msg = String()
            cmd_msg.data = f'move {self.target_velocity}'
            self.cmd_vel_publisher.publish(cmd_msg)

    def status_update(self):
        status_msg = String()
        status_msg.data = f'Current state: {self.current_state}, Target vel: {self.target_velocity}'
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = RobotControllerNode()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Controller stopped by user')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Python Integration

### Error Handling

```python
import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException

class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')

        try:
            self.declare_parameter('critical_param', 'default_value')
            self.critical_param = self.get_parameter('critical_param').value
        except Exception as e:
            self.get_logger().error(f'Failed to declare parameter: {e}')
            self.critical_param = 'default_value'

    def safe_publish(self, publisher, msg):
        try:
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish message: {e}')
```

### Resource Management

```python
class ResourceManagedNode(Node):
    def __init__(self):
        super().__init__('resource_managed_node')
        self.resources = []

    def destroy_node(self):
        # Clean up resources before destroying node
        for resource in self.resources:
            try:
                resource.cleanup()
            except Exception as e:
                self.get_logger().warn(f'Error cleaning up resource: {e}')

        super().destroy_node()
```

## Exercise: Python Agent Implementation

Create a Python agent that:
1. Subscribes to sensor data from a simulated robot
2. Processes the sensor data to determine if obstacles are present
3. Publishes velocity commands to avoid obstacles
4. Provides a service to change the robot's operating mode
5. Uses parameters to configure sensitivity thresholds

This exercise will help you integrate all the concepts learned in this section into a cohesive Python-based robot controller.

## Summary

Python integration with ROS 2 through rclpy provides a powerful and flexible way to create robot applications. The combination of Python's simplicity with ROS 2's communication patterns enables rapid development and prototyping of complex robotic systems. Understanding how to properly implement publishers, subscribers, services, actions, and parameters in Python is essential for creating robust and maintainable robot software.

In the next section, we'll explore URDF (Unified Robot Description Format) for describing humanoid robots.