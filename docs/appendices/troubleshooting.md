---
sidebar_position: 3
---

# Troubleshooting

This guide provides solutions to common issues encountered while working through the AI/Spec-Driven Book on Physical AI & Humanoid Robotics.

## General Troubleshooting Principles

1. **Check the Basics**: Verify power, network connections, and basic system status
2. **Isolate the Problem**: Test components individually to identify the source
3. **Check Logs**: Examine system logs for error messages and warnings
4. **Search First**: Many issues have been encountered by others; search documentation and forums
5. **Document Your Steps**: Keep track of what you've tried to avoid repeating efforts

## ROS 2 Troubleshooting

### Common Issues

#### 1. Nodes Not Communicating
**Symptoms**: Publishers and subscribers not connecting
**Solutions**:
```bash
# Check if ROS 2 environment is sourced
echo $ROS_DISTRO  # Should show 'humble'

# Verify network configuration
export ROS_LOCALHOST_ONLY=0  # If using multiple machines
export ROS_DOMAIN_ID=0       # Match on all machines

# Check for running nodes
ros2 node list

# Verify topic connections
ros2 topic list
ros2 topic info /topic_name
```

#### 2. Package Not Found
**Symptoms**: `command not found` or `package not found` errors
**Solutions**:
```bash
# Ensure ROS 2 environment is sourced
source /opt/ros/humble/setup.bash

# Check if workspace is sourced
source ~/your_workspace/install/setup.bash

# Verify package is in the right location
find ~/your_workspace/src -name "package.xml"
```

#### 3. Python Import Errors
**Symptoms**: `ModuleNotFoundError` or `ImportError`
**Solutions**:
```bash
# Check Python version (should be 3.10 for ROS 2 Humble)
python3 --version

# Verify Python packages are installed
pip3 list | grep -i ros
pip3 list | grep -i cv2

# Install missing packages
pip3 install --user package_name
```

### Performance Issues

#### 1. Slow Message Transport
**Symptoms**: High latency in topic communication
**Solutions**:
- Reduce message frequency
- Use message filters to reduce data rate
- Check network bandwidth if using multiple machines
- Consider using intraprocess communication for nodes in the same process

#### 2. Memory Leaks
**Symptoms**: Gradually increasing memory usage
**Solutions**:
- Use `ros2 lifecycle` nodes for better resource management
- Implement proper cleanup in callbacks
- Monitor memory usage: `ros2 run top_monitor top_monitor`

## Gazebo Troubleshooting

### Graphics and Performance Issues

#### 1. Gazebo Won't Start or Crashes
**Symptoms**: Gazebo fails to launch or crashes immediately
**Solutions**:
```bash
# Check graphics drivers
nvidia-smi  # For NVIDIA GPUs
glxinfo | grep "OpenGL renderer"  # Check OpenGL support

# Launch with software rendering
gz sim --render-engine ogre2

# Check for missing plugins
export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gz-sim7/plugins
```

#### 2. Poor Performance
**Symptoms**: Low frame rate, stuttering simulation
**Solutions**:
- Reduce physics update rate in world file
- Disable unnecessary visual effects
- Close other GPU-intensive applications
- Check for proper GPU drivers

#### 3. Model Loading Issues
**Symptoms**: Models fail to load or appear incorrectly
**Solutions**:
```bash
# Verify model paths
echo $GAZEBO_MODEL_PATH

# Check model files
ls ~/.gazebo/models/  # Default model directory
ls /usr/share/gazebo/models/  # System models

# Download missing models
gz sdf -p /path/to/model.sdf  # Validate model file
```

### Physics Simulation Issues

#### 1. Unstable Simulation
**Symptoms**: Objects behaving erratically, explosions in simulation
**Solutions**:
- Reduce time step in physics engine
- Adjust solver parameters (iterations, error reduction)
- Check mass and inertia values of models
- Verify collision geometries are properly defined

#### 2. Penetration Issues
**Symptoms**: Objects passing through each other
**Solutions**:
- Increase physics update rate
- Reduce maximum step size
- Improve collision mesh resolution
- Adjust contact parameters (surface mu, kp, kd)

## Unity Troubleshooting

### Import and Build Issues

#### 1. Package Import Failures
**Symptoms**: Unity packages fail to download or import
**Solutions**:
- Check internet connection
- Verify Unity version compatibility
- Clear Package Manager cache: `Edit → Preferences → Cache Server → Clear Cache`
- Try manual package installation from .tgz files

#### 2. Build Failures
**Symptoms**: Build process fails with errors
**Solutions**:
- Check target platform settings
- Verify all dependencies are installed
- Look for script compilation errors in Console
- Try building for different platforms to isolate the issue

### Performance Issues

#### 1. Slow Scene Loading
**Symptoms**: Long loading times for complex scenes
**Solutions**:
- Optimize mesh complexity
- Use occlusion culling
- Implement level of detail (LOD) systems
- Check for excessive draw calls

#### 2. Runtime Performance
**Symptoms**: Low frame rate during simulation
**Solutions**:
- Use Profiler window to identify bottlenecks
- Optimize shaders and materials
- Reduce polygon count where possible
- Consider using Unity's SRP (Scriptable Render Pipeline)

## NVIDIA Isaac Troubleshooting

### Isaac Sim Issues

#### 1. Isaac Sim Won't Launch
**Symptoms**: Isaac Sim fails to start or crashes
**Solutions**:
```bash
# Check CUDA compatibility
nvidia-smi
nvcc --version

# Verify Isaac Sim installation
ls ~/isaac-sim/  # Check installation directory

# Check logs
cat ~/isaac-sim/logs/isaac-sim.log
```

#### 2. GPU Memory Issues
**Symptoms**: "Out of memory" errors during simulation
**Solutions**:
- Close other GPU-intensive applications
- Reduce simulation complexity
- Lower rendering resolution
- Use less complex models or textures

### Isaac ROS Issues

#### 1. Package Build Failures
**Symptoms**: `colcon build` fails with errors
**Solutions**:
```bash
# Clean build directory
rm -rf build/ install/ log/

# Check dependencies
rosdep check --from-paths src --ignore-src

# Build specific package
colcon build --packages-select package_name

# Build with more verbose output
colcon build --event-handlers console_direct+
```

#### 2. Isaac ROS Nodes Not Working
**Symptoms**: Isaac ROS nodes fail to start or don't produce expected output
**Solutions**:
- Verify Isaac Sim is running before starting Isaac ROS nodes
- Check for proper network configuration
- Verify CUDA and graphics drivers
- Check Isaac ROS package versions compatibility

## Python Development Issues

### Virtual Environment Problems

#### 1. Mixed Package Versions
**Symptoms**: Conflicting package versions causing errors
**Solutions**:
```bash
# Create clean virtual environment
python3 -m venv ~/clean_robotics_env
source ~/clean_robotics_env/bin/activate

# Install packages with specific versions
pip install -r requirements.txt
```

#### 2. Permission Issues
**Symptoms**: Permission errors when installing packages
**Solutions**:
```bash
# Use user flag for local installation
pip install --user package_name

# Or use virtual environment (recommended)
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate
pip install package_name
```

## Simulation-to-Real Transfer Issues

### 1. Domain Gap Problems
**Symptoms**: Controller works in simulation but fails on real robot
**Solutions**:
- Add noise and disturbances to simulation
- Use system identification to model real robot dynamics
- Implement robust control techniques
- Gradually reduce simulation fidelity to match reality

### 2. Sensor Discrepancies
**Symptoms**: Sensor data differs between simulation and reality
**Solutions**:
- Calibrate sensors in both environments
- Add sensor noise models in simulation
- Use domain randomization techniques
- Implement sensor fusion to handle uncertainties

## Network and Communication Issues

### 1. Multi-Robot Communication
**Symptoms**: Robots can't communicate with each other
**Solutions**:
```bash
# Check network configuration
export ROS_DOMAIN_ID=0  # Match across all robots
export ROS_LOCALHOST_ONLY=0  # Enable network communication

# Verify network connectivity
ping other_robot_ip_address

# Check firewall settings
sudo ufw status
```

### 2. Remote Operation Issues
**Symptoms**: Can't control robot remotely
**Solutions**:
- Set up VPN for secure communication
- Configure proper ROS networking
- Use SSH tunneling for secure connections
- Implement proper security measures

## Hardware-Specific Troubleshooting

### Jetson Platform Issues

#### 1. Thermal Throttling
**Symptoms**: Performance degradation after extended operation
**Solutions**:
- Improve cooling with heatsinks or fans
- Reduce computational load
- Monitor temperature: `sudo tegrastats`
- Use power mode management

#### 2. Power Management
**Symptoms**: Unexpected shutdowns or performance changes
**Solutions**:
- Check power supply adequacy
- Use appropriate power mode: `sudo nvpmodel -m 0`
- Monitor power consumption

## Debugging Strategies

### 1. Logging and Monitoring
```bash
# ROS 2 logging
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=INFO
ros2 run your_package your_node --ros-args --log-level INFO

# System monitoring
htop          # CPU and memory usage
nvidia-smi    # GPU usage
iotop         # Disk I/O
```

### 2. Debugging Tools
- **ROS 2**: Use `rqt` for visualization, `ros2 bag` for data recording
- **Gazebo**: Use Gazebo GUI for visual debugging
- **Unity**: Use Unity Profiler and Console windows
- **Isaac**: Use Isaac Sim's debugging tools and visualization

## Common Error Messages and Solutions

### "Failed to load library"
- Check library paths: `echo $LD_LIBRARY_PATH`
- Verify library exists and has correct permissions
- Check for missing dependencies: `ldd /path/to/library.so`

### "Could not find a package configuration file"
- Ensure package is built and installed
- Check CMAKE_PREFIX_PATH: `echo $CMAKE_PREFIX_PATH`
- Source the workspace: `source install/setup.bash`

### "Permission denied" for device access
- Add user to appropriate group: `sudo usermod -a -G dialout $USER`
- Check device permissions: `ls -la /dev/ttyUSB*`
- Reboot after group changes: `newgrp dialout`

## Getting Help

### Documentation
- ROS 2 Documentation: docs.ros.org
- Gazebo Documentation: gazebo.org
- Unity Documentation: docs.unity3d.com
- Isaac Documentation: docs.nvidia.com/isaac

### Community Resources
- ROS Answers: answers.ros.org
- Gazebo Answers: answers.gazebosim.org
- Unity Community: unity.com/community
- NVIDIA Developer Forums: developer.nvidia.com

### RAG System Validation

When experiencing issues with the RAG (Retrieval-Augmented Generation) system integration, use the validation tools to diagnose problems:

import RAGValidation from '@site/src/components/RAGValidation/RAGValidation';

<RAGValidation />

### Common RAG Issues and Solutions

#### 1. Backend Connectivity Issues
**Symptoms**: Query interface shows connection errors or timeouts
**Solutions**:
- Verify RAG backend service is running on localhost:8000
- Check firewall settings that might block the connection
- Ensure backend API is properly configured

#### 2. Query Processing Failures
**Symptoms**: Queries fail to return responses or return errors
**Solutions**:
- Use the validation tools to test API functionality
- Verify query format and length requirements
- Check backend logs for error details

#### 3. Selected Text Capture Issues
**Symptoms**: Selected text not captured or not sent with queries
**Solutions**:
- Verify SelectedTextCapture component is properly loaded
- Check that selected text meets minimum length requirements (10 characters)
- Ensure proper event listener registration

## When to Ask for Help
1. You've spent more than 2 hours on a single issue
2. You've searched documentation and forums without success
3. You have a minimal reproducible example
4. You can clearly describe the expected vs. actual behavior

---

*Remember: Troubleshooting is a skill that improves with practice. Document your solutions to help others and create your own reference for future projects.*