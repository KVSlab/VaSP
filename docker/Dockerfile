FROM condaforge/mambaforge

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /tmp
# Install ssh (missing dependency to run conda envs)
RUN apt-get update && \
    apt-get install -y ssh build-essential libgl1-mesa-glx mesa-utils

# Upgrade mamba
RUN mamba upgrade -y mamba

# Copy environment and requirements files into docker env
COPY . ${WORKDIR}

# Update environment file with new environment name
RUN mamba env update --file environment.yml --name dockerenv

RUN echo "source activate dockerenv" > ~/.bashrc

RUN mamba run -n dockerenv python -m pip install .[test]

SHELL ["mamba", "run", "-n", "dockerenv", "/bin/bash", "-c"]
ENV MESA_LOADER_DRIVER_OVERRIDE=""

