# Chapter 5: Sensor Simulation - LiDAR, Depth Cameras, and IMUs

## Overview

In this chapter, we'll explore how to simulate various sensors commonly used in humanoid robotics: LiDAR, depth cameras, and IMUs. Proper sensor simulation is crucial for developing and testing perception systems in virtual environments before deploying on real robots.

## Sensor Simulation in Robotics

Sensor simulation plays a vital role in robotics development by:

- **Providing training data**: Generating labeled datasets for machine learning
- **Testing perception algorithms**: Validating computer vision and sensor fusion techniques
- **Reducing hardware costs**: Minimizing the need for expensive sensors during development
- **Ensuring safety**: Testing in dangerous scenarios without risk to hardware
- **Increasing reproducibility**: Creating consistent experimental conditions

## LiDAR Simulation

LiDAR (Light Detection and Ranging) sensors provide 3D point cloud data that's essential for humanoid robots for navigation, mapping, and obstacle detection.

### LiDAR Physics in Simulation

LiDAR simulation models the physical properties of laser beams:

- **Range**: Maximum and minimum detection distances
- **Resolution**: Angular resolution of the sensor
- **Accuracy**: Measurement precision and noise characteristics
- **Field of View**: Horizontal and vertical scanning angles

### Gazebo LiDAR Implementation

```xml
<!-- Example LiDAR sensor configuration -->
<sensor name="lidar_sensor" type="ray">
  <pose>0 0 0.5 0 0 0</pose>
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
      <namespace>/humanoid_robot</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
    <frame_name>lidar_link</frame_name>
  </plugin>
</sensor>
```

### Unity LiDAR Simulation

In Unity, LiDAR can be simulated using raycasting:

```csharp
using UnityEngine;
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityLidarSimulator : MonoBehaviour
{
    [Header("LiDAR Configuration")]
    public int horizontalSamples = 720;
    public int verticalSamples = 1;
    public float minAngle = -Mathf.PI;
    public float maxAngle = Mathf.PI;
    public float maxRange = 30.0f;
    public float minRange = 0.1f;
    
    [Header("ROS Settings")]
    public string scanTopic = "/scan";
    
    private ROSConnection ros;
    private float[] ranges;
    private LaserScanMsg laserScanMsg;
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ranges = new float[horizontalSamples];
        laserScanMsg = new LaserScanMsg();
        
        // Initialize laser scan message
        laserScanMsg.angle_min = minAngle;
        laserScanMsg.angle_max = maxAngle;
        laserScanMsg.angle_increment = (maxAngle - minAngle) / horizontalSamples;
        laserScanMsg.time_increment = 0.0f;
        laserScanMsg.scan_time = 0.1f;
        laserScanMsg.range_min = minRange;
        laserScanMsg.range_max = maxRange;
        laserScanMsg.ranges = ranges;
    }
    
    void Update()
    {
        SimulateLidarScan();
        PublishLidarData();
    }
    
    void SimulateLidarScan()
    {
        for (int i = 0; i < horizontalSamples; i++)
        {
            float angle = minAngle + (i * (maxAngle - minAngle) / horizontalSamples);
            
            // Perform raycast in the calculated direction
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            RaycastHit hit;
            
            if (Physics.Raycast(transform.position, transform.TransformDirection(direction), 
                              out hit, maxRange))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = float.PositiveInfinity; // Or maxRange for invalid readings
            }
        }
    }
    
    void PublishLidarData()
    {
        laserScanMsg.ranges = ranges;
        laserScanMsg.header.stamp = new TimeStamp(ros.Clock.Now());
        ros.Publish(scanTopic, laserScanMsg);
    }
}
```

## Depth Camera Simulation

Depth cameras provide both RGB and depth information, which is crucial for 3D scene understanding and manipulation tasks.

### Depth Camera in Gazebo

```xml
<!-- Depth camera configuration -->
<sensor name="depth_camera" type="depth">
  <pose>0 0 0.5 0 0 0</pose>
  <camera name="depth_cam">
    <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
    <baseline>0.2</baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
    <point_cloud_cutoff>0.1</point_cloud_cutoff>
    <point_cloud_cutoff_max>3.0</point_cloud_cutoff_max>
    <CxPrime>0</CxPrime>
    <Cx>0</Cx>
    <Cy>0</Cy>
    <focal_length>0</focal_length>
    <hack_baseline>0</hack_baseline>
    <frame_name>depth_camera_optical_frame</frame_name>
  </plugin>
</sensor>
```

### Unity Depth Camera Implementation

```csharp
using UnityEngine;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityDepthCamera : MonoBehaviour
{
    [Header("Camera Settings")]
    public int width = 640;
    public int height = 480;
    public float nearClip = 0.1f;
    public float farClip = 10.0f;
    
    [Header("ROS Settings")]
    public string rgbTopic = "/camera/rgb/image_raw";
    public string depthTopic = "/camera/depth/image_raw";
    
    private Camera cam;
    private RenderTexture renderTexture;
    private Texture2D rgbTexture;
    private Texture2D depthTexture;
    private ROSConnection ros;
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        cam = GetComponent<Camera>();
        
        // Set camera parameters
        cam.fieldOfView = 60f; // 60 degree FOV
        cam.nearClipPlane = nearClip;
        cam.farClipPlane = farClip;
        
        // Create render texture for depth
        renderTexture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGBFloat);
        cam.targetTexture = renderTexture;
        
        // Create textures for publishing
        rgbTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
        depthTexture = new Texture2D(width, height, TextureFormat.RFloat, false);
    }
    
    void Update()
    {
        CaptureAndPublishImages();
    }
    
    void CaptureAndPublishImages()
    {
        // Capture RGB image
        RenderTexture.active = renderTexture;
        rgbTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        rgbTexture.Apply();
        
        // Convert to ROS message and publish
        ImageMsg rgbMsg = TextureToImageMsg(rgbTexture);
        ros.Publish(rgbTopic, rgbMsg);
        
        // For depth, we'd use a depth shader or compute shader
        // This is a simplified example
        ProcessDepthData();
    }
    
    ImageMsg TextureToImageMsg(Texture2D texture)
    {
        // Convert texture to ROS Image message
        var imageMsg = new ImageMsg();
        imageMsg.header.stamp = new TimeStamp(ros.Clock.Now());
        imageMsg.header.frame_id = "camera_optical_frame";
        imageMsg.height = (uint)texture.height;
        imageMsg.width = (uint)texture.width;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(texture.width * 3); // 3 bytes per pixel for RGB
        imageMsg.data = texture.GetRawTextureData<byte>().ToArray();
        
        return imageMsg;
    }
    
    void ProcessDepthData()
    {
        // In a real implementation, this would capture depth data
        // using a depth shader or compute shader
    }
}
```

## IMU Simulation

IMUs (Inertial Measurement Units) provide crucial information about robot orientation, acceleration, and angular velocity.

### IMU Physics in Simulation

IMU simulation models:

- **Accelerometer**: Linear acceleration in 3 axes
- **Gyroscope**: Angular velocity in 3 axes
- **Magnetometer**: Magnetic field direction (optional)
- **Noise characteristics**: Realistic sensor noise and bias

### Gazebo IMU Implementation

```xml
<!-- IMU sensor configuration -->
<sensor name="imu_sensor" type="imu">
  <pose>0 0 0 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <ros>
      <namespace>/humanoid_robot</namespace>
      <remapping>~/out:=imu/data</remapping>
    </ros>
    <frame_name>imu_link</frame_name>
    <body_name>base_link</body_name>
    <update_rate>100</update_rate>
  </plugin>
</sensor>
```

### Unity IMU Simulation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityImuSimulator : MonoBehaviour
{
    [Header("IMU Noise Parameters")]
    public float gyroNoiseStdDev = 2e-4f;
    public float accelNoiseStdDev = 1.7e-2f;
    public float gyroBiasMean = 0.0000075f;
    public float accelBiasMean = 0.1f;
    
    [Header("ROS Settings")]
    public string imuTopic = "/imu/data";
    
    private ROSConnection ros;
    private ImuMsg imuMsg;
    private Rigidbody rb;
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        imuMsg = new ImuMsg();
        rb = GetComponent<Rigidbody>();
        
        // Initialize header
        imuMsg.header.frame_id = "imu_link";
    }
    
    void Update()
    {
        SimulateImuData();
        PublishImuData();
    }
    
    void SimulateImuData()
    {
        // Get angular velocity from rigidbody
        Vector3 angularVelocity = rb.angularVelocity;
        
        // Add noise to gyroscope readings
        angularVelocity.x += GaussianRandom(gyroNoiseStdDev) + gyroBiasMean;
        angularVelocity.y += GaussianRandom(gyroNoiseStdDev) + gyroBiasMean;
        angularVelocity.z += GaussianRandom(gyroNoiseStdDev) + gyroBiasMean;
        
        // Get linear acceleration (remove gravity)
        Vector3 linearAcceleration = rb.velocity; // This is simplified
        // In practice, you'd need to account for gravity and differentiate velocity
        
        // Add noise to accelerometer readings
        linearAcceleration.x += GaussianRandom(accelNoiseStdDev) + accelBiasMean;
        linearAcceleration.y += GaussianRandom(accelNoiseStdDev) + accelBiasMean;
        linearAcceleration.z += GaussianRandom(accelNoiseStdDev) + accelBiasMean;
        
        // Convert to ROS message format
        imuMsg.angular_velocity.x = angularVelocity.x;
        imuMsg.angular_velocity.y = angularVelocity.y;
        imuMsg.angular_velocity.z = angularVelocity.z;
        
        imuMsg.linear_acceleration.x = linearAcceleration.x;
        imuMsg.linear_acceleration.y = linearAcceleration.y;
        imuMsg.linear_acceleration.z = linearAcceleration.z;
        
        // Set orientation (this would come from the robot's pose)
        imuMsg.orientation.w = transform.rotation.w;
        imuMsg.orientation.x = transform.rotation.x;
        imuMsg.orientation.y = transform.rotation.y;
        imuMsg.orientation.z = transform.rotation.z;
    }
    
    void PublishImuData()
    {
        imuMsg.header.stamp = new TimeStamp(ros.Clock.Now());
        ros.Publish(imuTopic, imuMsg);
    }
    
    float GaussianRandom(float stdDev)
    {
        // Box-Muller transform for Gaussian random numbers
        float u1 = Random.Range(0.0000001f, 1f);
        float u2 = Random.Range(0f, 1f);
        float normalRand = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return normalRand * stdDev;
    }
}
```

## Sensor Fusion and Calibration

### Multi-Sensor Integration

For humanoid robots, combining data from multiple sensors improves perception:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class SensorFusion : MonoBehaviour
{
    [Header("Sensor References")]
    public UnityImuSimulator imu;
    public UnityLidarSimulator lidar;
    public UnityDepthCamera camera;
    
    private Dictionary<string, object> sensorData;
    
    void Start()
    {
        sensorData = new Dictionary<string, object>();
    }
    
    void Update()
    {
        CollectSensorData();
        FuseSensorData();
    }
    
    void CollectSensorData()
    {
        // Collect data from all sensors
        sensorData["imu"] = imu.imuMsg;
        sensorData["lidar"] = lidar.laserScanMsg;
        // Add camera data as needed
    }
    
    void FuseSensorData()
    {
        // Implement sensor fusion algorithm
        // Example: Extended Kalman Filter or Particle Filter
        EstimateRobotState();
    }
    
    void EstimateRobotState()
    {
        // Combine sensor readings to estimate robot pose, velocity, etc.
        // This is where complex fusion algorithms would run
    }
}
```

## Quality Metrics for Sensor Simulation

### Accuracy Assessment

Evaluate sensor simulation quality with:

- **RMSE**: Root Mean Square Error compared to real sensors
- **Precision/Recall**: For detection tasks
- **Temporal consistency**: Smoothness of sensor readings over time
- **Spatial accuracy**: Correct positioning of detected objects

### Validation Techniques

- **Cross-validation**: Compare with real sensor data
- **Statistical analysis**: Verify noise characteristics match real sensors
- **Perception pipeline testing**: Ensure downstream algorithms work correctly

## Best Practices

1. **Realistic Noise Modeling**: Include appropriate sensor noise and biases
2. **Computational Efficiency**: Balance simulation fidelity with performance
3. **Calibration**: Ensure simulated sensors match real sensor characteristics
4. **Validation**: Regularly compare simulation output with real sensors
5. **Modularity**: Design sensor simulators to be easily configurable
6. **Documentation**: Maintain clear specifications of sensor models

## Challenges and Limitations

- **The Reality Gap**: Differences between simulated and real sensor data
- **Computational Cost**: High-fidelity simulation requires significant resources
- **Complex Interactions**: Difficult to model all environmental factors
- **Dynamic Environments**: Changing conditions affect sensor performance

## Summary

In this chapter, we've explored the simulation of key sensors used in humanoid robotics: LiDAR, depth cameras, and IMUs. We've covered both Gazebo and Unity implementations, discussing the physics behind each sensor type and how to properly configure them for realistic simulation. Proper sensor simulation is essential for developing robust perception systems that can transfer from simulation to reality.