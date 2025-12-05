# Detailed Setup Guide

This guide walks through the complete setup of the ARC-M project environment.

## Table of Contents

1. [System Preparation](#1-system-preparation)
2. [Isaac Lab Installation](#2-isaac-lab-installation)
3. [Magistral LLM Setup](#3-magistral-llm-setup)
4. [ROS 2 Installation](#4-ros-2-installation)
5. [Project Configuration](#5-project-configuration)
6. [Verification](#6-verification)
7. [macOS Docker Setup](#7-macos-dockers-setup)

---

## 1. System Preparation

### 1.1 Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 Install NVIDIA Drivers

Check current driver version:

```bash
nvidia-smi
```

If needed, install/update drivers (version 580.65.06+ recommended):

```bash
# Add NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install latest production driver
sudo apt install nvidia-driver-580
sudo reboot
```

### 1.3 Install CUDA Toolkit

```bash
# Install CUDA 12
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 1.4 Install System Dependencies

```bash
sudo apt install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    python3.11 \
    python3.11-venv \
    python3-pip
```

---

## 2. Isaac Lab Installation

Isaac Lab 2.2 is compatible with Isaac Sim 5.0 and supports pip-based installation.

### 2.1 Create Virtual Environment

```bash
# Using uv (recommended for speed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Create Isaac Lab environment with Python 3.11
uv venv --python 3.11 env_isaaclab
source env_isaaclab/bin/activate

# OR using conda
# conda create -n env_isaaclab python=3.11
# conda activate env_isaaclab
```

### 2.2 Install PyTorch

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2.3 Install Isaac Sim via pip

```bash
# Install Isaac Sim 5.x
pip install 'isaacsim[all,extscache]' --extra-index-url https://pypi.nvidia.com
```

### 2.4 Clone and Install Isaac Lab

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install Isaac Lab
./isaaclab.sh --install
```

### 2.5 Verify Installation

```bash
# Test Isaac Sim launches correctly
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py

# Run a sample RL training (headless)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-v0 \
    --num_envs 64 \
    --headless \
    --max_iterations 100
```

---

## 3. Magistral LLM Setup

### 3.1 Option A: API Access (Recommended for Development)

```bash
# Install Mistral SDK
pip install mistralai

# Set API key
export MISTRAL_API_KEY="your-api-key-here"
echo 'export MISTRAL_API_KEY="your-api-key-here"' >> ~/.bashrc
```

### 3.2 Option B: Local Deployment (Magistral Small 24B)

For local inference, Magistral Small requires ~25GB VRAM (quantized: ~16GB).

```bash
# Install vLLM for efficient inference
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# Serve Magistral Small
vllm serve mistralai/Magistral-Small-2506 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --tensor-parallel-size 2 \
    --max-model-len 40960
```

### 3.3 Option C: Quantized Local Model (Consumer GPUs)

```bash
# Using Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull hf.co/mistralai/Magistral-Small-2506_gguf:Q4_K_M
```

---

## 4. ROS 2 Installation

ROS 2 Jazzy Jalisco requires Ubuntu 24.04.

### 4.1 Set Locale

```bash
locale  # Check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### 4.2 Add ROS 2 Repository

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe

# Add ROS 2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 4.3 Install ROS 2 Jazzy

```bash
sudo apt update
sudo apt install ros-jazzy-desktop

# Install development tools
sudo apt install ros-dev-tools python3-colcon-common-extensions
```

### 4.4 Configure Shell

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 4.5 Verify ROS 2

```bash
# Terminal 1
source /opt/ros/jazzy/setup.bash
ros2 run demo_nodes_cpp talker

# Terminal 2
source /opt/ros/jazzy/setup.bash
ros2 run demo_nodes_py listener
```

---

## 5. Project Configuration

### 5.1 Clone ARC-M Project

```bash
cd ~/arc-m-project
```

### 5.2 Install Project Dependencies

```bash
source ~/env_isaaclab/bin/activate

pip install \
    gymnasium \
    stable-baselines3 \
    rl-games \
    rsl-rl \
    tensorboard \
    wandb \
    pyyaml \
    mistralai \
    onnx \
    onnxruntime-gpu
```

### 5.3 Build ROS 2 Workspace

```bash
cd ~/arc-m-project/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## 6. Verification

### 6.1 Test Isaac Lab Environment

```bash
cd ~/IsaacLab
source ~/env_isaaclab/bin/activate

# Test custom environment
./isaaclab.sh -p ~/arc-m-project/scripts/test_environment.py
```

### 6.2 Test Magistral Connection

```bash
python ~/arc-m-project/scripts/test_magistral.py
```

### 6.3 Test Full Pipeline

```bash
# Run training with LLM-guided reward generation (small test)
python ~/arc-m-project/scripts/train_recovery_policy.py \
    --num_envs 256 \
    --max_iterations 100 \
    --use_magistral
```

---

## 7. macOS Docker Setup

For macOS users, we provide a Dockerized ROS 2 Jazzy environment. This allows you to develop ROS 2 nodes and visualize outputs without a Linux machine.

### Prerequisites
- Docker Desktop
- [uv](https://github.com/astral-sh/uv)

### Setup

1. **Build the container**:
   ```bash
   ./scripts/ros2_docker.sh build
   ```

2. **Start the environment**:
   ```bash
   ./scripts/ros2_docker.sh up
   ```

3. **Enter the shell**:
   ```bash
   ./scripts/ros2_docker.sh shell
   ```

4. **Build the workspace (inside container)**:
   ```bash
   # Inside docker container
   colcon build --symlink-install
   source install/setup.bash
   ```

---
 
## Troubleshooting

### Isaac Sim Won't Launch

```bash
# Check GPU compatibility
isaacsim isaacsim.exp.compatibility_check

# Reset user configuration
isaacsim --reset-user
```

### CUDA Out of Memory

Reduce parallel environments:

```bash
--num_envs 1024  # Instead of 4096
```

### ROS 2 Nodes Not Communicating

```bash
# Check ROS_DOMAIN_ID
echo $ROS_DOMAIN_ID

# Ensure same domain
export ROS_DOMAIN_ID=0
```

---

## Next Steps

1. Read `docs/REWARD_ENGINEERING.md` for prompt design guidance
2. Customize `envs/recovery_env.py` for your robot
3. Run first training: `python scripts/train_recovery_policy.py`
4. Monitor with TensorBoard: `tensorboard --logdir logs/`
