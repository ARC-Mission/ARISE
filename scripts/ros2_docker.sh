#!/bin/bash
# Helper script for ARC-M ROS 2 Docker environment

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"

function show_help {
    echo "Usage: ./ros2_docker.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build       Build the ROS 2 Docker image"
    echo "  up          Start the ROS 2 container (background)"
    echo "  down        Stop the ROS 2 container"
    echo "  shell       Open a bash shell in the running container"
    echo "  build_ws    Build the ROS 2 workspace inside container"
    echo "  run [cmd]   Run a command inside the container"
    echo ""
}

if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

COMMAND=$1
shift

cd "$DOCKER_DIR"

case $COMMAND in
    build)
        docker compose build
        ;;
    up)
        docker compose up -d
        echo "ROS 2 container started. Use 'shell' to enter."
        ;;
    down)
        docker compose down
        ;;
    shell)
        docker compose exec ros2 bash
        ;;
    build_ws)
        echo "Building ROS 2 workspace..."
        docker compose exec ros2 bash -c "source /opt/ros/jazzy/setup.bash && cd /root/ros2_ws && colcon build --symlink-install"
        ;;
    run)
        docker compose exec ros2 bash -c "source /opt/ros/jazzy/setup.bash && source /root/ros2_ws/install/setup.bash && $@"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
