#!/bin/bash
# ARC-M Project Environment Setup Script
# Automates installation of Isaac Lab, Magistral, and ROS 2 Jazzy

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
PYTHON_VERSION="3.11"
ISAAC_LAB_DIR="$HOME/IsaacLab"
VENV_NAME="env_isaaclab"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=================================================="
echo "     ARC-M Project Environment Setup"
echo "=================================================="
echo ""

# Check Ubuntu version
log_info "Checking Ubuntu version..."
if ! grep -q "24.04" /etc/os-release; then
    log_warn "This script is designed for Ubuntu 24.04. You have:"
    cat /etc/os-release | grep PRETTY_NAME
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check NVIDIA GPU
log_info "Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    log_error "NVIDIA drivers not found. Please install NVIDIA drivers first."
    log_info "Run: sudo apt install nvidia-driver-580"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
log_success "Found GPU: $GPU_INFO"

# Check VRAM
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_MB" -lt 8000 ]; then
    log_warn "GPU has ${VRAM_MB}MB VRAM. Recommended: 16GB+ for 4096 parallel environments."
fi

#==========================================
# Step 1: System Dependencies
#==========================================
log_info "Installing system dependencies..."
sudo apt update
sudo apt install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    software-properties-common \
    locales

# Set locale
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

log_success "System dependencies installed."

#==========================================
# Step 2: Python Virtual Environment
#==========================================
log_info "Setting up Python virtual environment..."

# Install uv (fast Python package manager)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment
cd "$HOME"
if [ ! -d "$VENV_NAME" ]; then
    uv venv --python ${PYTHON_VERSION} $VENV_NAME
fi

source "$HOME/$VENV_NAME/bin/activate"
pip install --upgrade pip

log_success "Virtual environment created: $HOME/$VENV_NAME"

#==========================================
# Step 3: PyTorch Installation
#==========================================
log_info "Installing PyTorch with CUDA support..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

log_success "PyTorch installed."

#==========================================
# Step 4: Isaac Sim & Isaac Lab
#==========================================
log_info "Installing Isaac Sim (this may take 10-20 minutes)..."

pip install 'isaacsim[all,extscache]' --extra-index-url https://pypi.nvidia.com

log_info "Cloning Isaac Lab..."
if [ ! -d "$ISAAC_LAB_DIR" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAAC_LAB_DIR"
fi

cd "$ISAAC_LAB_DIR"
./isaaclab.sh --install

log_success "Isaac Lab installed."

#==========================================
# Step 5: RL Libraries
#==========================================
log_info "Installing RL libraries..."

pip install \
    gymnasium \
    stable-baselines3 \
    rl-games \
    tensorboard \
    wandb \
    onnx \
    onnxruntime-gpu

# Install RSL-RL (optimized for Isaac Lab)
pip install rsl-rl

log_success "RL libraries installed."

#==========================================
# Step 6: Mistral/Magistral SDK
#==========================================
log_info "Installing Mistral SDK..."

pip install mistralai

log_info "For Magistral API access, set your API key:"
echo "  export MISTRAL_API_KEY='your-key-here'"

log_success "Mistral SDK installed."

#==========================================
# Step 7: ROS 2 Jazzy
#==========================================
log_info "Installing ROS 2 Jazzy..."

# Add ROS 2 repository
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-jazzy-desktop ros-dev-tools python3-colcon-common-extensions

log_success "ROS 2 Jazzy installed."

#==========================================
# Step 8: Configure Shell
#==========================================
log_info "Configuring shell environment..."

# Add to bashrc if not already present
BASHRC="$HOME/.bashrc"

if ! grep -q "env_isaaclab" "$BASHRC"; then
    cat >> "$BASHRC" << 'EOF'

# ARC-M Project Environment
alias arcm_activate='source ~/env_isaaclab/bin/activate && source /opt/ros/jazzy/setup.bash'
alias arcm_train='cd ~/IsaacLab && ./isaaclab.sh -p'

# Isaac Lab
export ISAAC_LAB_PATH="$HOME/IsaacLab"

# ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

EOF
fi

log_success "Shell configured. Use 'arcm_activate' to activate environment."

#==========================================
# Step 9: Project Setup
#==========================================
log_info "Setting up ARC-M project structure..."

cd "$PROJECT_DIR"

# Create placeholder files
touch models/.gitkeep

# Make scripts executable
chmod +x scripts/*.sh scripts/*.py 2>/dev/null || true

log_success "Project structure ready."

#==========================================
# Verification
#==========================================
echo ""
echo "=================================================="
echo "     Verification"
echo "=================================================="

log_info "Running verification checks..."

# Check Isaac Sim
python -c "import isaacsim; print('Isaac Sim: OK')" 2>/dev/null && \
    log_success "Isaac Sim: OK" || log_warn "Isaac Sim: May require first-run initialization"

# Check PyTorch
python -c "import torch; assert torch.cuda.is_available(); print('PyTorch CUDA: OK')" && \
    log_success "PyTorch CUDA: OK" || log_error "PyTorch CUDA: FAILED"

# Check Mistral
python -c "from mistralai import Mistral; print('Mistral SDK: OK')" && \
    log_success "Mistral SDK: OK" || log_error "Mistral SDK: FAILED"

# Check ROS 2
source /opt/ros/jazzy/setup.bash
ros2 --version && log_success "ROS 2 Jazzy: OK" || log_error "ROS 2: FAILED"

echo ""
echo "=================================================="
echo "     Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment:    source ~/env_isaaclab/bin/activate"
echo "  2. Set Mistral API key:     export MISTRAL_API_KEY='your-key'"
echo "  3. Test Isaac Lab:          cd ~/IsaacLab && ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py"
echo "  4. Start training:          python ~/arc-m-project/scripts/train_recovery_policy.py"
echo ""
echo "Documentation: $PROJECT_DIR/docs/SETUP_GUIDE.md"
echo ""
