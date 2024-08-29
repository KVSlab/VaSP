(install:docker)=

# Installing with Docker

The released Docker image can be found [here](https://github.com/KVSlab/VaSP/pkgs/container/vasp).
Below we assume that you have [Docker](https://docs.docker.com/get-docker/) installed on your system.
Ensure Docker is running on your system by executing the following command in your terminal:
```consolse
docker info
```
If Docker is running, this command will display detailed information about your Docker setup. If Docker is not running, you may see an error message, indicating that you need to start the Docker service.

## Step 1: Pull the Docker image

You can pull the Docker image for the latest `VaSP` package from its
official [GitHub container registry](https://github.com/KVSlab/VaSP/pkgs/container/vasp) using the following command:

``` console
docker pull ghcr.io/kvslab/vasp:latest
```

You can verify that the image has been successfully added by checking the output of 

```console
docker image -ls
```
which should give you (not exactly the same but something very similar)

```
ghcr.io/kvslab/vasp   master       856878575721   2 months ago   8.87GB
```

## Step 2: Run the Docker container

After pulling the Docker image, you can run the container using the following command:

``` console
docker run -w /home/shared/ -v $PWD:/home/shared/ -it ghcr.io/kvslab/vasp:latest
```
The `-w` flag sets the working directory inside a Docker container. The `-v` flag is used to mount a volume. In this case, it mounts the current directory ($PWD—a shell variable that represents the present working directory) on your host machine to the `/home/shared/` directory inside the Docker container. This allows you to share files between your host system and the Docker container. You may adjust `/home/shared` depending on your system and structure of the folders.

## Step 3: Verify the installation

You can verify that `VaSP` is installed correctly by running vasp commands, e.g.,

```console
vasp-generate-mesh --help
```

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
