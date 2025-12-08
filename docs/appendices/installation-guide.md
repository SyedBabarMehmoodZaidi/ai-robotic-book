---
sidebar_position: 2
---

# Installation Guide

This guide provides step-by-step instructions for installing the software tools required for the AI/Spec-Driven Book on Physical AI & Humanoid Robotics.

## Prerequisites

Before beginning the installation process, ensure your system meets the hardware requirements outlined in the Hardware Requirements appendix.

## 1. System Preparation

### Ubuntu 22.04 LTS Setup
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install basic development tools
sudo apt install -y build-essential cmake git python3-pip python3-dev
```

### Windows Setup
1. Install Windows Subsystem for Linux (WSL2) with Ubuntu 22.04
2. Or install native tools as outlined in each section below

### macOS Setup
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install basic tools
brew install git cmake
```

## 2. ROS 2 Installation (Humble Hawksbill)

### Ubuntu Installation
```bash
# Set locale
locale  # check for UTF-8
sudo locale-gen en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe

# Add the ROS 2 GPG key
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add the repository to your sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-colcon-common-extensions
```

### Environment Setup
```bash
# Source ROS 2 in your shell
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Verification
```bash
# Test installation
ros2 topic list
```

## 3. Gazebo Installation

### Gazebo Garden
```bash
# Add Gazebo repository
sudo curl -sSL http://get.gazebosim.org | sh

# Install Gazebo Garden
sudo apt install gazebo-garden
```

### Alternative: Gazebo Fortress (LTS)
```bash
# Install Gazebo Fortress
sudo apt install gazebo-fortress
```

### Verification
```bash
# Test Gazebo
gz sim --verbose
```

## 4. Unity Installation

### Unity Hub Setup
1. Download Unity Hub from [Unity's official website](https://unity.com/download)
2. Install Unity Hub following the platform-specific instructions
3. Sign in with a Unity ID (free account required)

### Unity Editor Installation
1. Open Unity Hub
2. Click "Installs" → "Add"
3. Select Unity 2022.3.x LTS version (recommended for robotics)
4. In the installer, select the following modules:
   - Android Build Support (if needed)
   - iOS Build Support (if needed)
   - Visual Studio Community (Windows) or MonoDevelop (macOS)

### Unity Robotics Package Installation
1. Create a new 3D project in Unity Hub
2. In the project, go to Window → Package Manager
3. Install "Unity Robotics Hub" from the Package Manager
4. Install additional packages as needed:
   - Unity Simulation for Robotics
   - ROS-TCP-Connector
   - Perception package (for synthetic data generation)

## 5. NVIDIA Isaac Installation

### Prerequisites
```bash
# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-535
sudo reboot
```

### Isaac Sim Installation
1. Download Isaac Sim from [NVIDIA Developer website](https://developer.nvidia.com/isaac-sim)
2. Extract the downloaded package:
```bash
tar -xzf isaac-sim-2023.1.1.tar.gz
```

3. Install Isaac Sim:
```bash
cd isaac-sim-2023.1.1
bash install_dependencies.sh
bash install.sh
```

### Isaac ROS Setup
1. Create a ROS 2 workspace:
```bash
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws
```

2. Install Isaac ROS packages:
```bash
# Install dependencies
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep

# Initialize rosdep
sudo rosdep init
rosdep update

# Clone Isaac ROS packages
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git src/isaac_ros_common
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git src/isaac_ros_visual_slam
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_bi3d.git src/isaac_ros_bi3d
# Add more packages as needed for your applications
```

3. Build the workspace:
```bash
colcon build --symlink-install --packages-select isaac_ros_common
source install/setup.bash
```

## 6. Development Tools

### Python Environment
```bash
# Install virtual environment tools
pip3 install virtualenv

# Create a virtual environment for robotics projects
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate

# Install Python packages for robotics
pip install numpy scipy matplotlib pandas jupyter
pip install opencv-python open3d
```

### Git Configuration
```bash
# Configure Git for robotics development
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global core.editor "nano"  # or your preferred editor
```

### VS Code Setup (Recommended IDE)
```bash
# Install VS Code
sudo snap install --classic code

# Install useful extensions
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-vscode.cpptools
code --install-extension redhat.vscode-yaml
code --install-extension twxs.cmake
```

## 7. Testing the Installation

### ROS 2 Test
```bash
# Terminal 1: Start a ROS 2 talker
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker

# Terminal 2: Start a ROS 2 listener
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py listener
```

### Python Test
```bash
# Test Python robotics packages
python3 -c "import numpy as np; print('NumPy version:', np.__version__)"
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

## 8. Docusaurus Setup for This Book

### Node.js Installation
```bash
# Install Node.js 18+ (LTS version recommended)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installation
node --version
npm --version
```

### Book Repository Setup
```bash
# Clone the book repository
git clone https://github.com/your-username/ai-robotic-book.git
cd ai-robotic-book

# Install dependencies
npm install

# Start local development server
npm start
```

## 9. Troubleshooting Common Issues

### ROS 2 Issues
- **Package not found**: Ensure your ROS 2 environment is sourced (`source /opt/ros/humble/setup.bash`)
- **Permission errors**: Check that your user is in the dialout group: `sudo usermod -a -G dialout $USER`
- **Python path issues**: Ensure you're using Python 3.10 as required by ROS 2 Humble

### Gazebo Issues
- **Graphics errors**: Ensure proper GPU drivers are installed
- **Plugin errors**: Check Gazebo plugin paths and permissions
- **Performance issues**: Close other GPU-intensive applications

### Isaac Sim Issues
- **CUDA compatibility**: Verify CUDA version compatibility with Isaac Sim requirements
- **License issues**: Ensure proper NVIDIA Developer account setup
- **GPU memory**: Close other applications to free GPU memory

### Unity Issues
- **Graphics API**: If Unity crashes, try switching graphics API in Edit → Preferences
- **Package Manager**: Ensure stable internet connection for package downloads
- **Performance**: Disable unnecessary Unity services for better performance

## 10. Keeping Systems Updated

### ROS 2 Updates
```bash
sudo apt update
sudo apt upgrade ros-humble-*
```

### System Updates
```bash
# Regular system updates
sudo apt update && sudo apt upgrade -y

# Check for ROS 2 security updates
sudo apt list --upgradable | grep ros
```

## 11. Optional: Container Setup (Docker)

For consistent environments across different systems:

```bash
# Install Docker
sudo apt install docker.io
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit for GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
```

---

*Note: Installation procedures may vary slightly depending on your specific system configuration and the latest versions of software packages. Always refer to the official documentation for the most up-to-date installation instructions.*