---
sidebar_position: 5
---

# Sensor Simulation: LiDAR, Depth Cameras, and IMUs

## Learning Objectives

By the end of this section, you will be able to:
- Understand the principles of sensor simulation in robotics
- Configure and simulate LiDAR sensors in Gazebo and Unity
- Set up depth camera simulation with realistic parameters
- Implement IMU simulation with appropriate noise models
- Visualize and process sensor data from simulation
- Compare simulated vs. real sensor characteristics

## Introduction to Sensor Simulation

Sensor simulation is a critical component of digital twin environments, allowing robots to perceive their virtual world in ways that closely approximate real-world sensors. Accurate sensor simulation is essential for:
- **Testing perception algorithms** without physical hardware
- **Training AI models** with synthetic data
- **Validating navigation systems** in controlled environments
- **Reducing development costs** and hardware wear
- **Enabling reproducible experiments** with known ground truth

## Sensor Simulation Principles

### Realism vs. Performance Trade-offs

When simulating sensors, there's always a trade-off between:
- **Accuracy**: How closely the simulation matches real sensor behavior
- **Performance**: Computational resources required for simulation
- **Complexity**: Difficulty of setup and tuning

### Key Simulation Aspects

1. **Geometric Accuracy**: Proper representation of sensor field of view, range, and resolution
2. **Noise Modeling**: Realistic sensor noise and imperfections
3. **Physics-based Effects**: Reflection, refraction, and environmental factors
4. **Timing**: Proper update rates and temporal consistency

## LiDAR Simulation

### Understanding LiDAR Sensors

LiDAR (Light Detection and Ranging) sensors emit laser pulses and measure the time for the light to return after reflecting off objects, creating precise distance measurements.

**Real LiDAR Characteristics**:
- **Range**: Typically 10-100 meters
- **Resolution**: Angular resolution of 0.1° to 1°
- **Accuracy**: Millimeter-level distance accuracy
- **Update Rate**: 5-20 Hz
- **Noise**: Distance-dependent measurement noise
- **Multi-return**: Ability to detect multiple reflections

### LiDAR Simulation in Gazebo

#### Basic LiDAR Configuration

```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>  <!-- -π radians -->
        <max_angle>3.14159</max_angle>    <!-- π radians -->
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

#### Advanced LiDAR with Noise and Physics

```xml
<sensor name="advanced_lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>0.5</resolution>
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
      <namespace>lidar</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
  </plugin>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>

  <!-- Noise model -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>
  </noise>
</sensor>
```

#### 3D LiDAR Configuration (Multi-line)

```xml
<sensor name="3d_lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>1024</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>64</samples>
        <resolution>1</resolution>
        <min_angle>-0.5236</min_angle>  <!-- -30 degrees -->
        <max_angle>0.3491</max_angle>   <!-- 20 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.3</min>
      <max>120.0</max>
      <resolution>0.001</resolution>
    </range>
  </ray>
  <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_gpu_laser.so">
    <ros>
      <namespace>velodyne</namespace>
      <remapping>~/out:=pointcloud</remapping>
    </ros>
    <topic_name>velodyne_points</topic_name>
    <frame_name>velodyne</frame_name>
    <min_range>0.9</min_range>
    <max_range>130.0</max_range>
    <gaussian_noise>0.008</gaussian_noise>
  </plugin>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>false</visualize>
</sensor>
```

### LiDAR Simulation in Unity

#### Unity LiDAR Simulation Script

```csharp
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class LiDARSimulator : MonoBehaviour
{
    public float scanRange = 10.0f;
    public int horizontalResolution = 360;
    public int verticalResolution = 1;
    public float updateRate = 10.0f;
    public bool visualizeRays = true;

    private LineRenderer[] rayVisualizers;
    private float[] ranges;
    private ROSConnection ros;
    private float timeSinceLastScan = 0.0f;

    void Start()
    {
        // Initialize ranges array
        ranges = new float[horizontalResolution * verticalResolution];

        // Initialize ray visualizers if needed
        if (visualizeRays)
        {
            InitializeRayVisualizers();
        }

        ros = ROSConnection.GetOrCreateInstance();
    }

    void Update()
    {
        timeSinceLastScan += Time.deltaTime;
        if (timeSinceLastScan >= 1.0f / updateRate)
        {
            GenerateLaserScan();
            timeSinceLastScan = 0.0f;
        }
    }

    void InitializeRayVisualizers()
    {
        rayVisualizers = new LineRenderer[horizontalResolution];
        for (int i = 0; i < horizontalResolution; i++)
        {
            GameObject rayGO = new GameObject($"LidarRay_{i}");
            rayGO.transform.SetParent(transform);
            LineRenderer lr = rayGO.AddComponent<LineRenderer>();
            lr.material = new Material(Shader.Find("Sprites/Default"));
            lr.widthMultiplier = 0.01f;
            lr.positionCount = 2;
            rayVisualizers[i] = lr;
        }
    }

    void GenerateLaserScan()
    {
        for (int i = 0; i < horizontalResolution; i++)
        {
            float angle = (2.0f * Mathf.PI * i) / horizontalResolution;

            // Cast ray in the direction of the current angle
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            RaycastHit hit;

            if (Physics.Raycast(transform.position, direction, out hit, scanRange))
            {
                ranges[i] = hit.distance;

                // Visualize the ray
                if (visualizeRays && i < rayVisualizers.Length)
                {
                    rayVisualizers[i].SetPosition(0, transform.position);
                    rayVisualizers[i].SetPosition(1, hit.point);
                }
            }
            else
            {
                ranges[i] = scanRange; // No obstacle detected

                if (visualizeRays && i < rayVisualizers.Length)
                {
                    rayVisualizers[i].SetPosition(0, transform.position);
                    rayVisualizers[i].SetPosition(1, transform.position + direction * scanRange);
                }
            }
        }

        // Publish to ROS
        PublishLaserScan();
    }

    void PublishLaserScan()
    {
        LaserScanMsg msg = new LaserScanMsg();
        msg.header = new Std_msgs.HeaderMsg();
        msg.header.stamp = new builtin_interfaces.TimeMsg(0, 0);
        msg.header.frame_id = "lidar_frame";

        msg.angle_min = -Mathf.PI;
        msg.angle_max = Mathf.PI;
        msg.angle_increment = (2.0f * Mathf.PI) / horizontalResolution;
        msg.time_increment = 0.0f; // Not applicable for simulated sensor
        msg.scan_time = 1.0f / updateRate;
        msg.range_min = 0.1f;
        msg.range_max = scanRange;

        msg.ranges = new float[ranges.Length];
        for (int i = 0; i < ranges.Length; i++)
        {
            msg.ranges[i] = ranges[i];
        }

        // Add noise if needed
        ApplyNoiseToScan(msg);

        ros.Publish("/scan", msg);
    }

    void ApplyNoiseToScan(LaserScanMsg scan)
    {
        // Add Gaussian noise to simulate real sensor imperfections
        for (int i = 0; i < scan.ranges.Length; i++)
        {
            float noise = Random.insideUnitSphere.x * 0.01f; // 1cm std deviation
            scan.ranges[i] += noise;

            // Ensure values stay within valid range
            scan.ranges[i] = Mathf.Clamp(scan.ranges[i], scan.range_min, scan.range_max);
        }
    }
}
```

## Depth Camera Simulation

### Understanding Depth Cameras

Depth cameras provide both color and depth information for each pixel, enabling 3D scene understanding.

**Real Depth Camera Characteristics**:
- **Resolution**: Typically 640x480 to 1920x1080
- **Depth Range**: 0.3m to 10m for RGB-D cameras
- **Accuracy**: 1-10mm depending on distance
- **FOV**: 57° to 90° diagonal
- **Update Rate**: 30-60 Hz

### Depth Camera Simulation in Gazebo

#### Basic Depth Camera Configuration

```xml
<sensor name="depth_camera" type="depth">
  <camera>
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
  </camera>
  <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
    <ros>
      <namespace>camera</namespace>
    </ros>
    <camera_name>depth_camera</camera_name>
    <image_topic_name>/image_raw</image_topic_name>
    <depth_image_topic_name>/depth/image_raw</depth_image_topic_name>
    <point_cloud_topic_name>/depth/points</point_cloud_topic_name>
    <frame_name>depth_camera_frame</frame_name>
    <baseline>0.1</baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
  </plugin>
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

#### Advanced Depth Camera with Noise

```xml
<sensor name="advanced_depth_camera" type="depth">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>1280</width>
      <height>720</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>20.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <plugin name="advanced_depth_controller" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>camera</namespace>
    </ros>
    <camera_name>rgb_depth</camera_name>
    <image_topic_name>/rgb/image_raw</image_topic_name>
    <camera_info_topic_name>/rgb/camera_info</camera_info_topic_name>
    <depth_image_topic_name>/depth/image_raw</depth_image_topic_name>
    <depth_image_camera_info_topic_name>/depth/camera_info</depth_image_camera_info_topic_name>
    <point_cloud_topic_name>/depth/points</point_cloud_topic_name>
    <frame_name>camera_depth_frame</frame_name>
    <min_depth>0.1</min_depth>
    <max_depth>20.0</max_depth>
    <point_cloud_cutoff>0.1</point_cloud_cutoff>
    <point_cloud_cutoff_max>20.0</point_cloud_cutoff_max>
  </plugin>
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>false</visualize>
</sensor>
```

### Depth Camera Simulation in Unity

#### Unity Depth Camera Script

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections;

public class DepthCameraSimulator : MonoBehaviour
{
    public Camera unityCamera;
    public int width = 640;
    public int height = 480;
    public float updateRate = 30.0f;
    public float maxDepth = 10.0f;
    public float minDepth = 0.1f;

    private RenderTexture depthTexture;
    private Texture2D depthTexture2D;
    private ROSConnection ros;
    private float timeSinceLastUpdate = 0.0f;

    void Start()
    {
        // Create depth texture
        depthTexture = new RenderTexture(width, height, 24, RenderTextureFormat.Depth);
        unityCamera.targetTexture = depthTexture;

        ros = ROSConnection.GetOrCreateInstance();
    }

    void Update()
    {
        timeSinceLastUpdate += Time.deltaTime;
        if (timeSinceLastUpdate >= 1.0f / updateRate)
        {
            PublishDepthImage();
            timeSinceLastUpdate = 0.0f;
        }
    }

    void PublishDepthImage()
    {
        // Read depth buffer
        RenderTexture.active = depthTexture;

        if (depthTexture2D == null)
        {
            depthTexture2D = new Texture2D(width, height, TextureFormat.RFloat, false);
        }

        depthTexture2D.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        depthTexture2D.Apply();

        RenderTexture.active = null;

        // Convert to ROS format and publish
        PublishToROS();
    }

    void PublishToROS()
    {
        ImageMsg depthMsg = new ImageMsg();
        depthMsg.header = new Std_msgs.HeaderMsg();
        depthMsg.header.stamp = new builtin_interfaces.TimeMsg(0, 0);
        depthMsg.header.frame_id = "camera_depth_frame";

        depthMsg.height = (uint)height;
        depthMsg.width = (uint)width;
        depthMsg.encoding = "32FC1"; // 32-bit float per channel
        depthMsg.is_bigendian = 0;
        depthMsg.step = (uint)(width * sizeof(float)); // 4 bytes per float

        // Convert texture data to float array
        Color[] pixels = depthTexture2D.GetPixels();
        float[] depthValues = new float[pixels.Length];

        for (int i = 0; i < pixels.Length; i++)
        {
            // Convert from color format to depth value
            depthValues[i] = pixels[i].r * maxDepth; // Assuming depth is stored in red channel
        }

        // Convert to byte array for ROS message
        depthMsg.data = new byte[depthValues.Length * sizeof(float)];
        for (int i = 0; i < depthValues.Length; i++)
        {
            byte[] floatBytes = System.BitConverter.GetBytes(depthValues[i]);
            for (int j = 0; j < floatBytes.Length; j++)
            {
                depthMsg.data[i * sizeof(float) + j] = floatBytes[j];
            }
        }

        ros.Publish("/camera/depth/image_raw", depthMsg);
    }
}
```

## IMU Simulation

### Understanding IMU Sensors

IMUs (Inertial Measurement Units) combine accelerometers, gyroscopes, and sometimes magnetometers to measure orientation, velocity, and gravitational forces.

**Real IMU Characteristics**:
- **Accelerometer Range**: ±2g to ±16g
- **Gyroscope Range**: ±250°/s to ±2000°/s
- **Magnetometer Range**: ±4800 µT
- **Update Rate**: 100-1000 Hz
- **Bias**: Long-term drift and offset
- **Noise**: Random walk and quantization noise

### IMU Simulation in Gazebo

#### Basic IMU Configuration

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
          <bias_mean>0.0001</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
          <bias_mean>0.0001</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
          <bias_mean>0.0001</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
          <bias_mean>0.01</bias_mean>
          <bias_stddev>0.1</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
          <bias_mean>0.01</bias_mean>
          <bias_stddev>0.1</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
          <bias_mean>0.01</bias_mean>
          <bias_stddev>0.1</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_controller" filename="libgazebo_ros_imu_sensor.so">
    <ros>
      <namespace>imu</namespace>
      <remapping>~/out:=data</remapping>
    </ros>
    <frame_name>imu_link</frame_name>
    <topic_name>imu</topic_name>
  </plugin>
</sensor>
```

#### Advanced IMU with Magnetometer

```xml
<sensor name="imu_with_mag" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <topic>__default_topic__</topic>
  <visualize>false</visualize>

  <imu>
    <!-- Gyroscope properties -->
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev> <!-- 0.0002 rad/s -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.001</bias_stddev>
          <dynamic_bias_stddev>0.001</dynamic_bias_stddev>
          <dynamic_bias_correlation_time>1.0</dynamic_bias_correlation_time>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.001</bias_stddev>
          <dynamic_bias_stddev>0.001</dynamic_bias_stddev>
          <dynamic_bias_correlation_time>1.0</dynamic_bias_correlation_time>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.001</bias_stddev>
          <dynamic_bias_stddev>0.001</dynamic_bias_stddev>
          <dynamic_bias_correlation_time>1.0</dynamic_bias_correlation_time>
        </noise>
      </z>
    </angular_velocity>

    <!-- Accelerometer properties -->
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev> <!-- 0.017 m/s² -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.1</bias_stddev>
          <dynamic_bias_stddev>0.01</dynamic_bias_stddev>
          <dynamic_bias_correlation_time>1.0</dynamic_bias_correlation_time>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.1</bias_stddev>
          <dynamic_bias_stddev>0.01</dynamic_bias_stddev>
          <dynamic_bias_correlation_time>1.0</dynamic_bias_correlation_time>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.1</bias_stddev>
          <dynamic_bias_stddev>0.01</dynamic_bias_stddev>
          <dynamic_bias_correlation_time>1.0</dynamic_bias_correlation_time>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

### IMU Simulation in Unity

#### Unity IMU Simulation Script

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class IMUSimulator : MonoBehaviour
{
    public float updateRate = 100.0f;
    public float gyroNoiseStd = 0.01f;
    public float accelNoiseStd = 0.017f;

    private ROSConnection ros;
    private float timeSinceLastUpdate = 0.0f;

    // IMU state with drift
    private Vector3 gyroBias = Vector3.zero;
    private Vector3 accelBias = Vector3.zero;
    private Vector3 trueAngularVelocity = Vector3.zero;
    private Vector3 trueLinearAcceleration = Vector3.zero;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Initialize bias with small random values
        gyroBias = new Vector3(Random.Range(-0.001f, 0.001f),
                              Random.Range(-0.001f, 0.001f),
                              Random.Range(-0.001f, 0.001f));
        accelBias = new Vector3(Random.Range(-0.01f, 0.01f),
                               Random.Range(-0.01f, 0.01f),
                               Random.Range(-0.01f, 0.01f));
    }

    void Update()
    {
        timeSinceLastUpdate += Time.deltaTime;
        if (timeSinceLastUpdate >= 1.0f / updateRate)
        {
            SimulateAndPublishIMU();
            timeSinceLastUpdate = 0.0f;
        }
    }

    void SimulateAndPublishIMU()
    {
        // Get the true motion from the robot's movement
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            // True angular velocity from physics
            trueAngularVelocity = rb.angularVelocity;

            // True linear acceleration (remove gravity)
            Vector3 gravity = Physics.gravity;
            Vector3 totalForce = rb.velocity / Time.fixedDeltaTime;
            trueLinearAcceleration = totalForce - gravity;
        }
        else
        {
            // If no rigidbody, use transform changes
            trueAngularVelocity = GetAngularVelocityFromTransform();
            trueLinearAcceleration = GetLinearAccelerationFromTransform();
        }

        // Add noise and bias to measurements
        Vector3 measuredGyro = trueAngularVelocity + gyroBias +
                              new Vector3(Random.insideUnitSphere.x * gyroNoiseStd,
                                        Random.insideUnitSphere.y * gyroNoiseStd,
                                        Random.insideUnitSphere.z * gyroNoiseStd);

        Vector3 measuredAccel = trueLinearAcceleration + accelBias +
                               new Vector3(Random.insideUnitSphere.x * accelNoiseStd,
                                         Random.insideUnitSphere.y * accelNoiseStd,
                                         Random.insideUnitSphere.z * accelNoiseStd);

        // Publish to ROS
        PublishIMUData(measuredGyro, measuredAccel);
    }

    Vector3 GetAngularVelocityFromTransform()
    {
        // Approximate angular velocity from transform changes
        static Vector3 lastAngular = Vector3.zero;
        Vector3 currentAngular = transform.rotation.eulerAngles;

        Vector3 deltaAngular = (currentAngular - lastAngular) / Time.deltaTime;
        lastAngular = currentAngular;

        // Convert from degrees to radians
        return deltaAngular * Mathf.Deg2Rad;
    }

    Vector3 GetLinearAccelerationFromTransform()
    {
        // Approximate linear acceleration from transform changes
        static Vector3 lastVelocity = Vector3.zero;
        Vector3 currentVelocity = (transform.position - lastPosition) / Time.deltaTime;
        lastPosition = transform.position;

        Vector3 acceleration = (currentVelocity - lastVelocity) / Time.deltaTime;
        lastVelocity = currentVelocity;

        // Remove gravity
        return acceleration - Physics.gravity;
    }

    Vector3 lastPosition = Vector3.zero;

    void PublishIMUData(Vector3 gyro, Vector3 accel)
    {
        ImuMsg msg = new ImuMsg();
        msg.header = new Std_msgs.HeaderMsg();
        msg.header.stamp = new builtin_interfaces.TimeMsg(0, 0);
        msg.header.frame_id = "imu_link";

        // Convert Unity coordinates to ROS coordinates (Unity: x-right, y-up, z-forward; ROS: x-forward, y-left, z-up)
        msg.angular_velocity.x = gyro.z;  // Unity z-forward -> ROS x-forward
        msg.angular_velocity.y = -gyro.x; // Unity x-right -> ROS y-left (negative)
        msg.angular_velocity.z = gyro.y;  // Unity y-up -> ROS z-up

        msg.linear_acceleration.x = accel.z;
        msg.linear_acceleration.y = -accel.x;
        msg.linear_acceleration.z = accel.y;

        // Initialize orientation as unit quaternion
        msg.orientation.w = 1.0f;
        msg.orientation.x = 0.0f;
        msg.orientation.y = 0.0f;
        msg.orientation.z = 0.0f;

        // Set covariance (diagonal values only)
        msg.angular_velocity_covariance = new double[9];
        msg.linear_acceleration_covariance = new double[9];

        // Set covariance values based on noise parameters
        float gyroCov = gyroNoiseStd * gyroNoiseStd;
        float accelCov = accelNoiseStd * accelNoiseStd;

        for (int i = 0; i < 9; i += 4) // Diagonal elements: 0, 4, 8
        {
            msg.angular_velocity_covariance[i] = gyroCov;
            msg.linear_acceleration_covariance[i] = accelCov;
        }

        ros.Publish("/imu/data", msg);
    }
}
```

## Sensor Fusion and Integration

### Multi-Sensor Coordination

When simulating multiple sensors, it's important to ensure proper coordination:

```xml
<!-- In your robot URDF/SDF -->
<link name="sensor_mount">
  <!-- Mount point for all sensors -->
  <visual>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </visual>
</link>

<!-- LiDAR sensor -->
<sensor name="lidar" type="ray">
  <pose>0 0 0.1 0 0 0</pose>
  <!-- ... LiDAR configuration ... -->
</sensor>

<!-- Depth camera -->
<sensor name="camera" type="depth">
  <pose>0.05 0 0.1 0 0 0</pose>
  <!-- ... camera configuration ... -->
</sensor>

<!-- IMU -->
<sensor name="imu" type="imu">
  <pose>-0.05 0 0.1 0 0 0</pose>
  <!-- ... IMU configuration ... -->
</sensor>
```

### Sensor Processing Pipeline

```python
#!/usr/bin/env python3
# Example sensor processing pipeline

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from std_msgs.msg import Header
import numpy as np

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Subscribers for all sensors
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # Publishers for processed data
        self.occupancy_pub = self.create_publisher(
            OccupancyGrid, '/map', 10)
        self.odom_pub = self.create_publisher(
            Odometry, '/odom', 10)

        # Sensor data storage
        self.lidar_data = None
        self.imu_data = None
        self.depth_data = None

        self.get_logger().info('Sensor processor initialized')

    def lidar_callback(self, msg):
        self.lidar_data = msg
        self.process_lidar_data()

    def imu_callback(self, msg):
        self.imu_data = msg
        self.process_imu_data()

    def depth_callback(self, msg):
        self.depth_data = msg
        self.process_depth_data()

    def process_lidar_data(self):
        if self.lidar_data is not None:
            # Process LiDAR data to create occupancy grid
            ranges = np.array(self.lidar_data.ranges)
            angles = np.linspace(
                self.lidar_data.angle_min,
                self.lidar_data.angle_max,
                len(ranges)
            )

            # Convert to Cartesian coordinates
            x_points = ranges * np.cos(angles)
            y_points = ranges * np.sin(angles)

            # Create occupancy grid (simplified)
            # In practice, this would be more sophisticated
            occupancy_grid = self.create_occupancy_grid(x_points, y_points)

            # Publish processed data
            # self.occupancy_pub.publish(occupancy_grid)

    def process_imu_data(self):
        if self.imu_data is not None:
            # Process IMU data for odometry
            # Integrate angular velocity to get orientation
            # Use accelerometer for gravity compensation
            pass

    def process_depth_data(self):
        if self.depth_data is not None:
            # Process depth image for 3D reconstruction
            # Convert image to point cloud
            pass

    def create_occupancy_grid(self, x_points, y_points):
        # Simplified occupancy grid creation
        # In practice, this would be more sophisticated
        pass

def main(args=None):
    rclpy.init(args=args)
    processor = SensorProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Interrupted by user')
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Sensor Simulation

### 1. Realistic Noise Modeling
- Use appropriate noise models based on real sensor specifications
- Include bias, drift, and thermal effects
- Consider environmental factors (temperature, humidity, etc.)

### 2. Performance Optimization
- Balance sensor resolution with simulation performance
- Use appropriate update rates for your application
- Consider using simplified models for real-time applications

### 3. Validation and Verification
- Compare simulated vs. real sensor data when available
- Validate sensor characteristics against manufacturer specifications
- Test perception algorithms with both simulated and real data

### 4. Coordinate System Consistency
- Ensure consistent coordinate systems across all sensors
- Verify frame transforms and conventions
- Use TF for proper sensor frame relationships

## Exercise: Sensor Simulation Integration

1. Create a simple robot model with all three sensor types (LiDAR, depth camera, IMU)
2. Configure the sensors with realistic parameters
3. Implement a sensor processing node that subscribes to all sensor data
4. Visualize the sensor data in RViz
5. Create a launch file that starts the robot with all sensors
6. Test the simulation and verify sensor data quality

This exercise will help you integrate all sensor simulation concepts into a complete system.

## Troubleshooting Common Issues

### LiDAR Issues
- **Empty scans**: Check ray directions and collision geometries
- **Range problems**: Verify min/max range settings
- **Performance**: Reduce resolution or update rate if needed

### Camera Issues
- **Black images**: Check camera pose and lighting
- **Distorted images**: Verify intrinsic parameters
- **Depth issues**: Ensure proper depth texture setup

### IMU Issues
- **Noisy data**: Adjust noise parameters appropriately
- **Drift**: Implement proper bias modeling
- **Coordinate issues**: Verify frame conventions

## Summary

Sensor simulation is a complex but essential part of digital twin development. Each sensor type has specific characteristics that must be accurately modeled to create realistic simulation environments. Proper configuration of noise models, update rates, and coordinate systems is crucial for effective sensor simulation that can be used for algorithm development and testing.

Understanding the principles of each sensor type and how to implement them in both Gazebo and Unity environments provides the foundation for creating comprehensive digital twin systems that accurately represent real-world robotic perception capabilities.

In the next section, we'll work on exercises that demonstrate digital twin concepts with sensor data visualization.