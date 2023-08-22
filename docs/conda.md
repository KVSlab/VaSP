(install:conda)=

# Installing with `conda`

## Prerequisites

- The `conda` package manager must be installed on your computer. It can be installed
  through [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

(install:linux)=

## Installation on Linux or macOS

## Installation on Windows

We recommend Windows users to use [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install)
and follow the [Linux](install:linux) instructions, or use [Docker](install:docker).

Alternatively, Windows users may install the `FEniCS` dependency from source, by following
the [FEniCS Reference Manual](https://fenics.readthedocs.io/en/latest/installation.html). Then, download the remaining
dependencies through `conda` by removing the `fenics` dependency inside `environment.yml` and follow the steps of
the [Linux/macOS](install:linux) installation instructions.

## Editable installation of ___

If you want to make changes to any of the scripts included in `software`, you can install an editable version on your
machine by supplying the `--editable` flag:

```
$ python3 -m pip install --editable .
```

The `--editable` flag installs the project in editable mode meaning that any changes to the original package will be
reflected directly in your environment.

