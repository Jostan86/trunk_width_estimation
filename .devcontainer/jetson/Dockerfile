# FROM dustynv/pytorch:1.11-r35.3.1
# FROM ultralytics/ultralytics:latest-jetson
FROM this_container:l4t-r36.3.0-onnxruntime
USER root

# Install stuff, if you have any
RUN apt-get update \
    && apt-get install -y \
    git \
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
    pip install importlib_resources \
    && sudo rm -rf /var/lib/apt/lists/*

RUN sudo apt-get update && \
    sudo apt-get install -y \
    libgl1-mesa-glx \
    python3-pyqt5 \
    && sudo rm -rf /var/lib/apt/lists/*

    
# # Update all packages
# RUN sudo apt update && sudo apt upgrade -y

# # Install Git
# RUN sudo apt install -y git

# These (should / should help to) force the gui applications to show up only on the remote display window
RUN echo "export DISPLAY=:0" >> ~/.bashrc
RUN echo "export REMOTE_CONTAINERS_DISPLAY_SOCK=/tmp/.X11-unix/X0" >> ~/.bashrc

USER root