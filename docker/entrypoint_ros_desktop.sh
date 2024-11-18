#!/bin/bash
#set -e

function ros_source_env() 
{
	if [ -f "$1" ]; then
		echo "sourcing   $1"
		source "$1"
	else
		echo "notfound   $1"
	fi	
}

ros_source_env "/opt/ros/$ROS_DISTRO/setup.bash"
ros_source_env "$ROS_WS_ROOT/install/setup.bash"

echo "ROS_DISTRO $ROS_DISTRO"
echo "ROS_ROOT   $ROS_ROOT"

exec "$@"


