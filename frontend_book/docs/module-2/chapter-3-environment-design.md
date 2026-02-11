# Chapter 3: Environment Design and World Building in Gazebo

## Overview

In this chapter, we'll explore how to design and build environments in Gazebo for humanoid robot testing and training. We'll cover creating realistic worlds, configuring lighting, and setting up diverse scenarios for comprehensive robot evaluation.

## Gazebo World Structure

A Gazebo world file is an SDF (Simulation Description Format) file that defines:

- Physical environment layout
- Lighting conditions
- Weather effects
- Static and dynamic objects
- Ground surfaces and materials

### Basic World File Structure

```xml
<sdf version="1.7">
  <world name="my_world">
    <!-- Global properties -->
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    
    <!-- Physics engine -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    
    <!-- Scene lighting -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
    </scene>
    
    <!-- Light sources -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.707 -0.707 -0.707</direction>
    </light>
    
    <!-- Models in the environment -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Custom models -->
    <model name="my_robot">
      <!-- Model definition -->
    </model>
  </world>
</sdf>
```

## Creating Ground Surfaces

Ground surfaces significantly affect humanoid robot locomotion:

### Flat Ground Plane
```xml
<model name="ground_plane">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <plane>
          <normal>0 0 1</normal>
          <size>100 100</size>
        </plane>
      </geometry>
      <surface>
        <friction>
          <ode>
            <mu>1.0</mu>
            <mu2>1.0</mu2>
          </ode>
        </friction>
      </surface>
    </collision>
    <visual name="visual">
      <geometry>
        <plane>
          <normal>0 0 1</normal>
          <size>100 100</size>
        </plane>
      </geometry>
      <material>
        <ambient>0.7 0.7 0.7 1</ambient>
        <diffuse>0.7 0.7 0.7 1</diffuse>
        <specular>0.01 0.01 0.01 1</specular>
      </material>
    </visual>
  </link>
</model>
```

### Textured Ground
```xml
<visual name="textured_ground_visual">
  <geometry>
    <plane>
      <normal>0 0 1</normal>
      <size>10 10</size>
    </plane>
  </geometry>
  <material>
    <script>
      <uri>file://media/materials/scripts/gazebo.material</uri>
      <name>Gazebo/GrassLawn</name>
    </script>
  </material>
</visual>
```

## Building Complex Environments

### Indoor Environments
For humanoid robots, indoor environments might include:

- Rooms with furniture
- Doorways and corridors
- Stairs and ramps
- Obstacles and clutter

```xml
<!-- Room with walls -->
<model name="room_walls">
  <static>true</static>
  <!-- Wall definitions -->
  <link name="wall_front">
    <pose>0 5 1 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box><size>10 0.2 2</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>10 0.2 2</size></box>
      </geometry>
      <material>
        <ambient>0.8 0.8 0.8 1</ambient>
        <diffuse>0.8 0.8 0.8 1</diffuse>
      </material>
    </visual>
  </link>
  <!-- Additional walls... -->
</model>
```

### Outdoor Environments
Outdoor environments for humanoid robots might include:

- Terrain variations
- Natural obstacles (trees, rocks)
- Weather considerations
- Urban structures

## Creating Obstacle Courses

Obstacle courses are essential for testing humanoid robot capabilities:

### Stepping Stones
```xml
<model name="stepping_stones">
  <static>true</static>
  <!-- Series of platforms at varying heights -->
  <link name="stone_1">
    <pose>1 0 0.1 0 0 0</pose>
    <collision><geometry><box><size>0.3 0.3 0.2</size></box></geometry></collision>
    <visual><geometry><box><size>0.3 0.3 0.2</size></box></geometry></visual>
  </link>
  <link name="stone_2">
    <pose>1.5 0.5 0.15 0 0 0</pose>
    <collision><geometry><box><size>0.3 0.3 0.2</size></box></geometry></collision>
    <visual><geometry><box><size>0.3 0.3 0.2</size></box></geometry></visual>
  </link>
  <!-- Additional stones... -->
</model>
```

### Narrow Passages
```xml
<model name="narrow_passage">
  <static>true</static>
  <link name="barrier_left">
    <pose>-0.5 0 1 0 0 0</pose>
    <collision><geometry><box><size>0.1 5 2</size></box></geometry></collision>
  </link>
  <link name="barrier_right">
    <pose>0.5 0 1 0 0 0</pose>
    <collision><geometry><box><size>0.1 5 2</size></box></geometry></collision>
  </link>
</model>
```

## Dynamic Elements

Dynamic elements add realism and challenge:

### Moving Obstacles
```xml
<model name="moving_obstacle">
  <link name="obstacle_body">
    <pose>0 0 0.5 0 0 0</pose>
    <inertial>
      <mass>1.0</mass>
      <inertia>
        <ixx>0.1</ixx><ixy>0</ixy><ixz>0</ixz>
        <iyy>0.1</iyy><iyz>0</iyz><izz>0.1</izz>
      </inertia>
    </inertial>
    <collision><geometry><sphere><radius>0.3</radius></sphere></geometry></collision>
    <visual><geometry><sphere><radius>0.3</radius></sphere></geometry></visual>
  </link>
  
  <!-- Plugin to move the obstacle -->
  <plugin name="model_pusher" filename="libgazebo_ros_p3d.so">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <body_name>obstacle_body</body_name>
    <topic_name>obstacle_position</topic_name>
  </plugin>
</model>
```

## Lighting and Visual Effects

Proper lighting enhances realism and affects sensor simulation:

### Directional Lights (Sun)
```xml
<light name="sun" type="directional">
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <attenuation>
    <range>1000</range>
    <constant>0.9</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <direction>-0.707 -0.707 -0.707</direction>
</light>
```

### Point Lights
```xml
<light name="room_light" type="point">
  <pose>0 0 3 0 0 0</pose>
  <diffuse>1 1 1 1</diffuse>
  <specular>0.5 0.5 0.5 1</specular>
  <attenuation>
    <range>10</range>
    <constant>0.2</constant>
    <linear>0.5</linear>
    <quadratic>0.2</quadratic>
  </attenuation>
</light>
```

## Environmental Effects

### Fog
```xml
<scene>
  <fog type="exp" density="0.01">
    <color>0.8 0.8 0.8 1</color>
  </fog>
</scene>
```

### Sky Properties
```xml
<scene>
  <sky>
    <time>14:00</time>
    <sun_direction>-0.707 -0.707 -0.707</sun_direction>
    <clouds>
      <speed>0.5</speed>
      <direction>0.8 0.6 0</direction>
      <humidity>0.5</humidity>
      <mean_size>0.5</mean_size>
    </clouds>
  </sky>
</scene>
```

## Model Libraries and Asset Management

### Using Built-in Models
Gazebo comes with a library of models:
- ground_plane
- sun
- box, sphere, cylinder
- table, chair, etc.

### Creating Custom Models
Custom models should be placed in `~/.gazebo/models/` with the structure:
```
~/.gazebo/models/my_model/
├── model.config
└── model.sdf
```

### model.config file
```xml
<?xml version="1.0"?>
<model>
  <name>My Custom Model</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A custom model for humanoid robot testing.</description>
</model>
```

## Best Practices for Environment Design

1. **Progressive Complexity**: Start with simple environments and gradually increase complexity
2. **Realistic Scaling**: Ensure all objects are properly scaled relative to the humanoid robot
3. **Performance Considerations**: Balance visual fidelity with simulation performance
4. **Safety Margins**: Include buffer zones around critical areas
5. **Repeatability**: Design deterministic scenarios for consistent testing
6. **Variety**: Create diverse environments to test different capabilities

## Testing Environments

Design environments that test specific humanoid robot capabilities:

- **Balance**: Uneven terrain, narrow surfaces
- **Navigation**: Obstacle avoidance, path planning
- **Manipulation**: Objects of various sizes and weights
- **Locomotion**: Stairs, slopes, different ground materials

## Summary

In this chapter, we've explored how to design and build comprehensive environments in Gazebo for humanoid robot testing. We've covered ground surfaces, complex environments, obstacle courses, dynamic elements, and environmental effects. In the next chapter, we'll discuss high-fidelity visualization with Unity.