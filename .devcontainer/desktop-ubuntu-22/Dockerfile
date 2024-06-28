FROM ubuntu:22.04

# ARG CACHEBUST=1

USER root

# turn off interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install stuff, if you have any
RUN apt-get update \
    && apt-get install -y \
    git \
    python3-pip \
    python3-opencv \
    # libgl1 \
    # libglib2.0-0 \
    # libxrender1 \
    # libsm6 \
    # libxext6 \
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

RUN sudo apt-get update && \
    pip install ultralytics \
    pip install scikit-image \ 
    pip install opencv-python \
    pip install pydantic \
    && sudo rm -rf /var/lib/apt/lists/*

   
# Update all packages
RUN sudo apt update && sudo apt upgrade -y && sudo rm -rf /var/lib/apt/lists/*

RUN echo "export WIDTH_ESTIMATION_PACKAGE_PATH=/trunk_width_estimation" >> ~/.bashrc