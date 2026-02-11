# Chapter 3: Humanoid Modeling with URDF and Python Control

## Overview

In this chapter, we'll explore how to model humanoid robots using URDF (Unified Robot Description Format) and control them with Python through the rclpy library. This chapter bridges the gap between abstract AI algorithms and physical robot components.

## Unified Robot Description Format (URDF)

URDF is an XML format for representing a robot. It defines the physical and visual properties of a robot, including:

- Links: Rigid parts of the robot (e.g., torso, limbs)
- Joints: Connections between links (e.g., hinges, prismatic joints)
- Visual elements: How the robot appears in simulation
- Collision elements: How the robot interacts physically with the environment
- Inertial properties: Mass, center of mass, and inertia tensor

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <!-- A joint connecting to another link -->
  <joint name="base_to_head" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.5"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Humanoid Robot Anatomy

A typical humanoid robot consists of:

- **Torso**: Main body containing computational units and power
- **Head**: Contains cameras, sensors, and processing units
- **Arms**: Manipulation capability with shoulders, elbows, wrists
- **Legs**: Locomotion capability with hips, knees, ankles
- **Hands**: Fine manipulation with multiple degrees of freedom

## URDF for Humanoid Robots

Humanoid robots require special attention to:

- **Kinematic chains**: Ensuring proper connectivity from torso to extremities
- **Degrees of freedom**: Providing sufficient mobility for human-like movements
- **Actuation**: Defining how joints are controlled
- **Sensors**: Including IMUs, force/torque sensors, cameras

### Example Humanoid URDF Snippet

```xml
<!-- Hip joint definition -->
<joint name="left_hip_joint" type="revolute">
  <parent link="torso"/>
  <child link="left_thigh"/>
  <origin xyz="0 -0.1 -0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="1.0" friction="0.0"/>
</joint>

<link name="left_thigh">
  <visual>
    <geometry>
      <cylinder length="0.4" radius="0.05"/>
    </geometry>
    <origin xyz="0 0 -0.2"/>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.4" radius="0.05"/>
    </geometry>
    <origin xyz="0 0 -0.2"/>
  </collision>
  <inertial>
    <mass value="2.0"/>
    <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.005"/>
  </inertial>
</link>
```

## Controlling Robots with Python (rclpy)

Once we have a robot model, we need to control it. The rclpy library provides Python bindings for ROS 2.

### Joint State Publisher

To visualize joint movements, we often use the joint_state_publisher:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Timer to periodically publish joint states
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.update_joints)
        
        # Initialize joint positions
        self.joint_names = ['left_hip_joint', 'right_hip_joint', 'left_knee_joint', 'right_knee_joint']
        self.joint_positions = [0.0, 0.0, 0.0, 0.0]

    def update_joints(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        self.joint_pub.publish(msg)
```

### Joint Trajectory Controller

For more sophisticated control, we use trajectory messages:

```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class TrajectoryController(Node):
    def __init__(self):
        super().__init__('trajectory_controller')
        self.traj_pub = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10
        )

    def send_trajectory(self, joint_names, positions, time_from_start=2.0):
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(time_from_start), nanosec=0)
        
        traj_msg.points.append(point)
        self.traj_pub.publish(traj_msg)
```

## Robot State Publisher

The robot_state_publisher takes URDF and joint positions to compute the forward kinematics and publish TF transforms:

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class CustomStatePublisher(Node):
    def __init__(self):
        super().__init__('custom_state_publisher')
        self.tf_broadcaster = TransformBroadcaster(self)
        
    def broadcast_transforms(self, joint_positions):
        # Calculate transforms based on joint angles
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        
        # Set translation and rotation based on kinematics
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(t)
```

## Simulation Integration

To use our URDF models in simulation:

1. **Gazebo**: Load the URDF into the physics simulator
2. **RViz**: Visualize the robot with proper TF frames
3. **Controllers**: Implement PID controllers for joint actuation

### Gazebo Configuration

Include Gazebo-specific tags in your URDF:

```xml
<gazebo reference="left_thigh">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>

<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <joint_name>left_hip_joint</joint_name>
  </plugin>
</gazebo>
```

## Control Strategies for Humanoid Robots

Humanoid robots require specialized control strategies:

- **Balance control**: Maintaining center of mass over support polygon
- **Walking gaits**: Coordinated leg movement for locomotion
- **Whole-body control**: Coordinating multiple limbs simultaneously
- **Impedance control**: Adapting to environmental contacts

## Practical Exercise

Create a simple humanoid model with:
1. A torso
2. Two legs with hip, knee, and ankle joints
3. Two arms with shoulder, elbow, and wrist joints
4. A head with neck joint

Then implement a Python node that controls the joint positions to make the robot wave.

## Summary

In this chapter, we've learned how to model humanoid robots using URDF and control them with Python through rclpy. We've covered the essential components of robot descriptions and control mechanisms that bridge AI algorithms with physical robot components. This knowledge forms the foundation for more advanced humanoid robotics applications.