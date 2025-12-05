# ARC-M ARISE

An advanced **Hierarchical Reinforcement Learning (HRL)** system that combines **Reasoning LLMs (Mistral Magistral)** with **RL policies trained in NVIDIA Isaac Lab** for autonomous robot recovery in unstructured environments.

## 🏗️ Architecture

- **Reasoning Layer**: Mistral Magistral (LLM) for reward function design and high-level strategy
- **Training Layer**: NVIDIA Isaac Lab + RSL-RL (PPO) for skill acquisition
- **Deployment Layer**: ROS 2 Jazzy (Linux) or Dockerized ROS 2 (macOS) for real-time control

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended)
- NVIDIA GPU with drivers (for Training)
- Docker Desktop (for macOS ROS 2)

### 2. Installation

```bash
# Clone
git clone https://github.com/arc-m/arise.git
cd arise

# Install dependencies
uv sync

# Configure Environment
cp .env.example .env
# Edit .env and set MISTRAL_API_KEY
```

### 3. Training (Linux + GPU)

```bash
# Verify environment
python scripts/test_environment.py

# Start training
python scripts/train_recovery_policy.py --num_envs 4096 --use_magistral
```

### 4. ROS 2 Deployment (macOS/cross-platform)

Use the Docker helper to run ROS 2 Jazzy:

```bash
# Build and start container
./scripts/ros2_docker.sh build
./scripts/ros2_docker.sh up

# Enter shell
./scripts/ros2_docker.sh shell

# Inside container:
ros2 launch arc_m_skills arc_m_launch.py
```

## 📂 Project Structure

- `envs/`: Isaac Lab environment definitions
- `scripts/`: Training and utility scripts
- `ros2_ws/`: ROS 2 workspace (arc_m_skills package)
- `config/`: Configuration files (Isaac Lab, Magistral, ROS 2)
- `docker/`: Docker configuration for macOS compatibility
- `models/`: Trained policy checkpoints

## 📚 Documentation

- [Setup Guide](docs/SETUP_GUIDE.md)
- [Reward Engineering](docs/REWARD_ENGINEERING.md)