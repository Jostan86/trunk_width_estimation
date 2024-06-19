FROM osrf/ros:humble-desktop-full

USER root

# Install stuff, if you have any
RUN apt-get update \
    && apt-get install -y \
    git \
    python3-pyqt5 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Add vscode user with same UID and GID as your host system
# (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a non-root user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir /home/$USERNAME/.config && chown ${USER_UID}:${USER_GID} /home/$USERNAME/.config

# Set up sudo
RUN apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

RUN sudo apt-get update && \
    pip install ultralytics \
    pip install scikit-image \ 
    pip install opencv-python \
    && sudo rm -rf /var/lib/apt/lists/*

# Make a ros2 workspace
RUN mkdir -p ~/ros2_ws/src

# Clone the interface pacakge into the workspace
RUN cd ~/ros2_ws/src && git clone https://github.com/Jostan86/pf_orchard_interfaces.git

# set the ROS_DISTRO
ENV ROS_DISTRO=humble

# Build the workspace
RUN cd ~/ros2_ws && . /opt/ros/${ROS_DISTRO}/setup.sh && colcon build
    
# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# These (should / should help to) force the gui applications to show up only on the remote display window
RUN echo "export WIDTH_ESTIMATION_PACKAGE_PATH=/width_estimation" >> ~/.bashrc

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc

USER root