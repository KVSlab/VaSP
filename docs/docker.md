(install:docker)=

# Installing with Docker

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) must be installed on your system.

## Building our own docker image

Instead of pulling the Docker image for `software` from GitHub, you can build it yourself using the provided `Dockerfile`.
The Dockerfile can be found in the `docker` folder located in the root of the project repository.

To build the Docker image, open a terminal window and navigate to the `docker` folder of the project. Then, execute the
following command:

``` console
docker build -t software .
```

This command will use the instructions in the `Dockerfile` to create a Docker image with the name `software`. Once the
Docker image is built, you can run a container from it using the `docker run` command.
