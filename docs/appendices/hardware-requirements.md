---
sidebar_position: 1
---

# Hardware Requirements

This page outlines the hardware requirements for implementing the exercises and projects in the AI/Spec-Driven Book on Physical AI & Humanoid Robotics.

## Minimum System Requirements

### Digital Twin Workstation
For simulation and development work:

- **CPU**: Intel i7-8700K or AMD Ryzen 7 3700X (6+ cores recommended)
- **RAM**: 16 GB minimum, 32 GB recommended
- **GPU**: NVIDIA GTX 1060 6GB or AMD RX 580 8GB (NVIDIA RTX series recommended for Isaac Sim)
- **Storage**: 500 GB SSD minimum
- **OS**: Ubuntu 22.04 LTS, Windows 10/11, or macOS 12+
- **Network**: Stable internet connection for package downloads and updates

### Recommended System Specifications
For optimal performance with complex simulations:

- **CPU**: Intel i9-12900K or AMD Ryzen 9 5900X
- **RAM**: 32-64 GB
- **GPU**: NVIDIA RTX 3080/4080 or RTX A4000/A5000 series
- **Storage**: 1 TB NVMe SSD
- **Additional**: Dedicated cooling system for sustained performance

## Robot Platforms

### NVIDIA Jetson Kits
For embedded robotics applications:

- **Jetson Orin AGX**: 32 GB RAM, 2048-core NVIDIA Ampere GPU
  - Best for: Advanced perception and AI processing
  - Cost: ~$1,200-1,600

- **Jetson Orin NX**: 8 GB RAM, 1024-core NVIDIA Ampere GPU
  - Best for: Intermediate robotics projects
  - Cost: ~$400-500

- **Jetson Nano**: 4 GB RAM, 128-core NVIDIA Maxwell GPU
  - Best for: Entry-level projects and learning
  - Cost: ~$100-150

### ARM Single Board Computers (SBCs)

- **Raspberry Pi 4**: 8 GB RAM, quad-core ARM Cortex-A72
  - Best for: Simple robotic applications, sensor integration
  - Cost: ~$100-150
  - Limitations: Limited GPU performance, not suitable for heavy AI workloads

- **Odroid XU4**: ARM big.LITTLE architecture with 8 cores
  - Best for: Intermediate embedded robotics
  - Cost: ~$80-100

### Robot Lab Configurations

#### Educational Setup (Small Lab)
- 1-2 Digital Twin Workstations (as specified above)
- 2-3 Jetson Orin NX kits
- 1-2 Raspberry Pi 4 units
- Basic sensors (cameras, IMUs, LiDAR)
- Basic manipulator arms or mobile bases

#### Advanced Setup (Research Lab)
- 3-5 High-performance Digital Twin Workstations (as recommended above)
- 5-10 Jetson Orin AGX kits
- Multiple sensor configurations
- Advanced manipulators and mobile platforms
- Motion capture system (optional)
- Additional networking equipment for multi-robot systems

## Sensor Requirements

### Essential Sensors for Simulation
These sensors should be modeled in your simulation environments:

- **RGB-D Camera**: For visual perception and mapping
- **IMU (Inertial Measurement Unit)**: For orientation and motion sensing
- **LiDAR**: For 2D/3D mapping and obstacle detection
- **Force/Torque Sensors**: For manipulation tasks
- **GPS Module**: For outdoor navigation (simulation)

### Physical Sensor Options
For real robot implementations:

- **Intel RealSense D435/D455**: RGB-D camera with depth sensing
- **Intel Realsense T265**: Visual-inertial odometry tracker
- **Hokuyo UAM-05LP**: 2D LiDAR scanner
- **Velodyne Puck/VLP-16**: 3D LiDAR scanner
- **XSens MTi-30/60**: High-precision IMU
- **ATI Gamma F/T**: Force/torque sensor

## Cost Estimates

### Individual Student Setup
- Digital Twin Workstation: $1,500-3,000 (if building)
- Jetson Nano: $150-200
- Basic sensors: $300-500
- **Total**: $1,950-3,700

### Budget-Conscious Student Setup
- Mid-range laptop: $800-1,200 (with discrete GPU)
- Raspberry Pi 4: $100-150
- Basic camera: $50-100
- **Total**: $950-1,450 (simulation-focused approach)

### Educational Institution Setup
- 10 Digital Twin Workstations: $15,000-30,000
- 20 Jetson kits: $2,000-8,000
- Shared sensor kits: $2,000-5,000
- **Total**: $19,000-43,000

## Alternative Approaches

### Cloud-Based Simulation
- **AWS RoboMaker**: Cloud-based robotics simulation and deployment
- **Microsoft Azure Digital Twins**: Cloud-based simulation services
- **Google Cloud AI Platform**: For AI model training and deployment
- **Cost**: Variable, typically $0.10-$2.00/hour depending on instance type

### Container-Based Approach
- **Docker**: For consistent development environments
- **NVIDIA Container Toolkit**: For GPU-accelerated containers
- **Benefits**: Consistent environments, easier collaboration
- **Requirements**: Docker-compatible system with GPU support

## Recommendations by Use Case

### Academic Learning
- Focus on simulation capabilities
- Mid-range workstation with good GPU
- Entry-level Jetson kit for practical experience

### Research Applications
- High-performance workstation with professional GPU
- Multiple Jetson AGX units for distributed processing
- Professional-grade sensors

### Hobbyist/Enthusiast
- Use existing PC with decent GPU for simulation
- Raspberry Pi 4 for basic embedded projects
- Gradually upgrade as needed

## Troubleshooting Common Hardware Issues

### GPU Issues
- Verify CUDA compatibility for Isaac Sim
- Ensure proper driver installation
- Check power supply adequacy for high-end GPUs

### Network Issues
- Ensure stable internet for package downloads
- Configure proper network settings for multi-robot systems
- Use wired connections for critical applications

### Storage Issues
- Ensure sufficient space for simulation environments
- Use SSDs for better performance
- Regular cleanup of simulation caches

---

*Note: Hardware requirements may vary based on specific implementations and updates to software frameworks. Always verify compatibility with the latest versions of ROS 2, Gazebo, Unity, and NVIDIA Isaac tools.*