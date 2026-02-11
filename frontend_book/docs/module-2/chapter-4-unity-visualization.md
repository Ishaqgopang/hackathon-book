# Chapter 4: High-Fidelity Interaction and Visualization with Unity

## Overview

In this chapter, we'll explore how Unity can be used for high-fidelity visualization and interaction in robotics applications. While Gazebo excels at physics simulation, Unity offers unparalleled visual quality and user interaction capabilities that complement robotic systems.

## Unity in Robotics Context

Unity has emerged as a powerful tool for robotics visualization and simulation due to its:

- **Photorealistic rendering**: High-quality graphics for immersive experiences
- **VR/AR support**: Direct integration with virtual and augmented reality systems
- **User interface capabilities**: Sophisticated GUI systems for robot control
- **Asset ecosystem**: Extensive library of 3D models and environments
- **Cross-platform deployment**: Support for various devices and platforms

## Unity Robotics Simulation Pipeline

The typical workflow involves:

1. **Model Import**: Bringing robot models into Unity
2. **Environment Creation**: Building high-fidelity scenes
3. **Physics Setup**: Configuring Unity's physics engine
4. **ROS Integration**: Connecting to ROS/ROS 2 systems
5. **Visualization**: Displaying sensor data and robot state
6. **Interaction**: Enabling user control and monitoring

## Setting Up Unity for Robotics

### Installing Unity
- Download Unity Hub from unity.com
- Install Unity Editor (recommended version 2021.3 LTS or newer)
- Install required packages through the Package Manager

### Unity Robotics Packages
Install the following packages:
- **ROS-TCP-Connector**: For ROS communication
- **URDF-Importer**: For importing robot models
- **XR packages**: For VR/AR support if needed

## Importing Robot Models

### Using URDF Importer
Unity's URDF Importer allows direct import of URDF files:

```csharp
// Example script to import and configure a robot
using UnityEngine;
using Unity.Robotics.URDFImport;

public class RobotSetup : MonoBehaviour
{
    public GameObject robotPrefab;
    
    void Start()
    {
        // Configure joint limits and motor properties
        ConfigureJoints();
    }
    
    void ConfigureJoints()
    {
        var joints = GetComponentsInChildren<ArticulationBody>();
        foreach (var joint in joints)
        {
            // Set joint limits based on URDF specifications
            var drive = joint.xDrive;
            drive.lowerLimit = -1.57f;  // Example: -90 degrees
            drive.upperLimit = 1.57f;   // Example: 90 degrees
            joint.xDrive = drive;
        }
    }
}
```

### Manual Model Setup
For complex robots, manual setup might be required:

```csharp
using UnityEngine;

public class ManualRobotSetup : MonoBehaviour
{
    [Header("Joint Configuration")]
    public ArticulationBody[] joints;
    public float[] jointLimitsMin;
    public float[] jointLimitsMax;
    
    void Start()
    {
        ConfigureRobotJoints();
    }
    
    void ConfigureRobotJoints()
    {
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] != null)
            {
                var drive = joints[i].xDrive;
                drive.lowerLimit = jointLimitsMin[i];
                drive.upperLimit = jointLimitsMax[i];
                joints[i].xDrive = drive;
                
                // Set stiffness and damping
                drive.stiffness = 1000f;
                drive.damping = 100f;
                joints[i].xDrive = drive;
            }
        }
    }
}
```

## Physics Configuration for Humanoid Robots

### Articulation Bodies
Unity uses ArticulationBody components for articulated rigid bodies:

```csharp
using UnityEngine;

public class HumanoidPhysics : MonoBehaviour
{
    [Header("Balance Parameters")]
    public float balanceStrength = 10f;
    public Transform centerOfMassTarget;
    
    private ArticulationBody[] robotJoints;
    
    void Start()
    {
        robotJoints = GetComponentsInChildren<ArticulationBody>();
        ConfigurePhysicsParameters();
    }
    
    void ConfigurePhysicsParameters()
    {
        foreach (var joint in robotJoints)
        {
            // Configure joint physics properties
            var drive = joint.linearXDrive;
            drive.forceLimit = 1000f;
            joint.linearXDrive = drive;
            
            // Set joint friction
            joint.jointFriction = 0.1f;
        }
    }
    
    void FixedUpdate()
    {
        MaintainBalance();
    }
    
    void MaintainBalance()
    {
        // Implement balance control logic
        // This is a simplified example
        foreach (var joint in robotJoints)
        {
            // Apply corrective forces based on center of mass
            if (centerOfMassTarget != null)
            {
                Vector3 direction = (centerOfMassTarget.position - transform.position).normalized;
                joint.AddForce(direction * balanceStrength);
            }
        }
    }
}
```

## ROS Integration with Unity

### TCP Connection Setup
```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;

public class UnityROSConnection : MonoBehaviour
{
    private ROSConnection ros;
    
    [Header("ROS Topics")]
    public string jointStatesTopic = "/joint_states";
    public string cmdVelTopic = "/cmd_vel";
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<JointStateMsg>(cmdVelTopic);
        ros.Subscribe<JointStateMsg>(jointStatesTopic, OnJointStatesReceived);
    }
    
    void OnJointStatesReceived(JointStateMsg jointState)
    {
        // Update Unity robot model based on received joint states
        UpdateRobotPose(jointState);
    }
    
    void UpdateRobotPose(JointStateMsg jointState)
    {
        // Map joint positions to Unity transforms
        for (int i = 0; i < jointState.name.Count; i++)
        {
            string jointName = jointState.name[i];
            float jointPosition = (float)jointState.position[i];
            
            // Find corresponding joint in Unity model
            Transform jointTransform = FindJointByName(jointName);
            if (jointTransform != null)
            {
                // Update joint rotation based on received position
                jointTransform.localRotation = Quaternion.Euler(0, 0, jointPosition * Mathf.Rad2Deg);
            }
        }
    }
    
    Transform FindJointByName(string name)
    {
        // Search for joint by name in the hierarchy
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            if (child.name == name)
                return child;
        }
        return null;
    }
    
    public void SendJointCommands(float[] positions)
    {
        // Send joint commands to ROS
        var jointCmd = new JointStateMsg();
        jointCmd.position = positions.Select(p => (double)p).ToArray();
        ros.Publish(cmdVelTopic, jointCmd);
    }
}
```

## High-Fidelity Visualization Techniques

### Shader Optimization
For realistic robot visualization:

```hlsl
Shader "Robot/RealisticRobot"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Color ("Color", Color) = (1,1,1,1)
        _Metallic ("Metallic", Range(0,1)) = 0.0
        _Smoothness ("Smoothness", Range(0,1)) = 0.5
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma target 3.0

        sampler2D _MainTex;
        fixed4 _Color;
        half _Metallic;
        half _Smoothness;

        struct Input
        {
            float2 uv_MainTex;
        };

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
            o.Albedo = c.rgb;
            o.Metallic = _Metallic;
            o.Smoothness = _Smoothness;
            o.Alpha = c.a;
        }
        ENDCG
    }
}
```

### Post-Processing Effects
Enhance visual quality with post-processing:

```csharp
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;

public class RobotVisualizationEffects : MonoBehaviour
{
    public PostProcessVolume postProcessVolume;
    private DepthOfField dof;
    
    void Start()
    {
        // Get depth of field effect
        postProcessVolume.profile.TryGetSettings(out dof);
    }
    
    void Update()
    {
        // Dynamically adjust focus based on robot position
        if (dof != null)
        {
            // Focus on robot or specific point of interest
            Vector3 robotPosition = transform.position;
            float distanceToCamera = Vector3.Distance(Camera.main.transform.position, robotPosition);
            dof.focusDistance.value = distanceToCamera;
        }
    }
}
```

## User Interaction Systems

### VR Controller Integration
```csharp
using UnityEngine;
using UnityEngine.XR;

public class VRRobotControl : MonoBehaviour
{
    [Header("VR Controllers")]
    public XRNode leftControllerNode;
    public XRNode rightControllerNode;
    
    private InputDevice leftController;
    private InputDevice rightController;
    
    void Start()
    {
        leftController = InputDevices.GetDeviceAtXRNode(leftControllerNode);
        rightController = InputDevices.GetDeviceAtXRNode(rightControllerNode);
    }
    
    void Update()
    {
        UpdateControllerInputs();
    }
    
    void UpdateControllerInputs()
    {
        if (leftController.isValid)
        {
            // Get trigger and grip values
            leftController.TryGetFeatureValue(CommonUsages.trigger, out float triggerValue);
            leftController.TryGetFeatureValue(CommonUsages.grip, out float gripValue);
            
            // Use inputs to control robot
            if (triggerValue > 0.1f)
            {
                MoveRobotLeftArm(triggerValue);
            }
        }
        
        if (rightController.isValid)
        {
            rightController.TryGetFeatureValue(CommonUsages.primaryButton, out bool primaryButton);
            
            if (primaryButton)
            {
                ToggleRobotMode();
            }
        }
    }
    
    void MoveRobotLeftArm(float amount)
    {
        // Implement arm movement logic
    }
    
    void ToggleRobotMode()
    {
        // Switch between different robot modes
    }
}
```

## Sensor Visualization

### Camera Feed Integration
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class CameraFeedVisualizer : MonoBehaviour
{
    public RawImage cameraDisplay;
    private Texture2D texture;
    
    void Start()
    {
        // Initialize texture for camera feed
        texture = new Texture2D(640, 480, TextureFormat.RGB24, false);
        cameraDisplay.texture = texture;
        
        // Subscribe to camera topic
        ROSConnection.GetOrCreateInstance()
            .Subscribe<ImageMsg>("/camera/image_raw", OnCameraImageReceived);
    }
    
    void OnCameraImageReceived(ImageMsg imageMsg)
    {
        // Convert ROS image to Unity texture
        byte[] imageData = imageMsg.data;
        
        // Update texture with image data
        texture.LoadRawTextureData(imageData);
        texture.Apply();
    }
}
```

## Performance Optimization

### Level of Detail (LOD) System
```csharp
using UnityEngine;

public class RobotLODManager : MonoBehaviour
{
    [System.Serializable]
    public class LODLevel
    {
        public float distance;
        public GameObject lodObject;
    }
    
    public LODLevel[] lodLevels;
    public Transform viewer;
    
    void Update()
    {
        float distance = Vector3.Distance(transform.position, viewer.position);
        
        // Activate appropriate LOD level
        for (int i = 0; i < lodLevels.Length; i++)
        {
            bool isActive = i == 0 || distance <= lodLevels[i].distance;
            lodLevels[i].lodObject.SetActive(isActive);
        }
    }
}
```

## Best Practices

1. **Performance Monitoring**: Regularly monitor frame rates and optimize accordingly
2. **Asset Optimization**: Use appropriate polygon counts and textures
3. **Physics Tuning**: Balance accuracy with performance for real-time simulation
4. **Modular Design**: Create reusable components for different robots
5. **Testing**: Validate Unity visualization against real robot behavior
6. **Documentation**: Maintain clear documentation of Unity-ROS interfaces

## Integration with Other Tools

Unity can be integrated with other robotics tools:

- **RViz**: For complementary visualization
- **Gazebo**: For physics simulation alongside Unity visuals
- **Blender**: For asset creation and modeling
- **MATLAB/Simulink**: For control algorithm development

## Summary

In this chapter, we've explored how Unity can be used for high-fidelity visualization and interaction in robotics applications. We've covered robot model import, physics configuration, ROS integration, visualization techniques, and user interaction systems. In the next chapter, we'll discuss sensor simulation for perception pipeline development.