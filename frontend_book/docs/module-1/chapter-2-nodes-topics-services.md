# Chapter 2: ROS 2 Communication - Nodes, Topics, and Services

## Overview

In this chapter, we'll dive deep into the communication patterns that make ROS 2 powerful. We'll explore nodes, topics, and services - the fundamental building blocks of ROS 2 communication.

## Nodes

Nodes are processes that perform computation in the ROS 2 system. They are the basic execution units of a ROS program. Each node can perform specific functions such as:

- Reading data from sensors
- Processing sensor data
- Controlling actuators
- Providing user interfaces
- Running algorithms

### Creating a Node

Every ROS 2 node inherits from the `rclpy.Node` class. Here's a basic example:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        # Initialize node-specific components here

def main(args=None):
    rclpy.init(args=args)
    minimal_node = MinimalNode()
    
    # Spin to keep the node alive
    rclpy.spin(minimal_node)
    
    # Cleanup
    minimal_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics and Message Passing

Topics enable asynchronous communication between nodes using a publish/subscribe pattern. Publishers send messages to topics, and subscribers receive messages from topics.

### Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
```

## Services

Services provide synchronous request/response communication between nodes. A service client sends a request to a service server, which processes the request and returns a response.

### Service Server Example

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {response.sum}')
        return response
```

### Service Client Example

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self):
        self.req.a = 41
        self.req.b = 1
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Quality of Service (QoS)

ROS 2 provides Quality of Service profiles to configure communication characteristics:

- **Reliability**: Best effort or reliable delivery
- **Durability**: Volatile or transient local
- **History**: Keep all or keep last N messages
- **Depth**: Size of the message queue

Example with custom QoS:

```python
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy, DurabilityPolicy

qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)
```

## Working with Multiple Nodes

In humanoid robotics, you'll typically have many nodes running simultaneously:

- Sensor driver nodes (camera, IMU, LIDAR)
- Perception nodes (object detection, SLAM)
- Control nodes (walking, manipulation)
- Planning nodes (path planning, motion planning)
- Interface nodes (GUI, teleoperation)

### Launch Files

Launch files allow you to start multiple nodes with a single command:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='publisher',
            name='talker'
        ),
        Node(
            package='my_package',
            executable='subscriber',
            name='listener'
        )
    ])
```

## Best Practices

1. **Node Design**: Keep nodes focused on a single responsibility
2. **Topic Naming**: Use descriptive, consistent naming conventions
3. **Message Types**: Use standard message types when possible
4. **Error Handling**: Implement proper error handling and logging
5. **Resource Management**: Clean up resources properly in destructors

## Summary

In this chapter, we've covered the core communication patterns in ROS 2: nodes, topics, and services. We've seen how to implement publishers, subscribers, and service clients/servers. In the next chapter, we'll explore how to model humanoid robots using URDF and control them with Python.