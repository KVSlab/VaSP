# VaSP - ***Va***scular Fluid-***S***tructure Interaction ***P***ipeline

[![GPL-3.0](https://img.shields.io/github/license/KVSlab/VaSP)](LICENSE)
[![codecov](https://codecov.io/gh/KVSlab/VaSP/graph/badge.svg?token=LNyRxL8Uyw)](https://codecov.io/gh/KVSlab/VaSP)
[![CI](https://github.com/KVSlab/VaSP/actions/workflows/check_and_test_package.yml/badge.svg)](https://github.com/KVSlab/VaSP/actions/workflows/check_and_test_package.yml)
[![GitHub pages](https://github.com/KVSlab/VaSP/actions/workflows/build_docs.yml/badge.svg)](https://github.com/KVSlab/VaSP/actions/workflows/build_docs.yml)

<p align="center">
    <img src=docs/figures/functionality.png width="830 height="370" alt="Output pre processing"/>
</p>

## Description
The Vascular Fluid-Structure Interaction Simulation Pipeline (VaSP) is a toolkit for simulating fluid-structure interactions (FSI) in vascular systems. It streamlines the process from pre-processing to post-processing of vascular FSI simulations.
Starting with medical image-based surface meshes, VaSP uses extended version of [VMTK](http://www.vmtk.org) to generate volumetric FSI meshes. It then runs FSI simulations using [turtleFSI](https://github.com/KVSlab/turtleFSI). For post-processing, VaSP employs [FEniCS](https://fenicsproject.org/) and other Python packages to compute hemodynamic indices like wall shear stress and stress/strain. By integrating these tools, VaSP aims to simplify vascular FSI analyses.

## Installation
VaSP is a Python package for Python >= 3.10, with main dependencies to [VaMPy](https://github.com/KVSlab/VaMPy)  
and [turtleFSI](https://github.com/KVSlab/turtleFSI). VaSP and its dependencies can be installed with `conda` on Linux and  
macOS using the following command:

```
conda create -n your_environment -c conda-forge vasp
```

More details on installation via `conda` can be found [here](https://kvslab.github.io/VaSP/conda.html). The package can also be installed and run through its latest `Docker` image supported by Windows, Linux, and macOS, as explained [here](https://kvslab.github.io/VaSP/docker.html).


## Documentation
The documentation is hosted [here](https://kvslab.github.io/VaSP/).

## Citation
If you use VaSP in your research, please cite:

```bibtex
@article{Yamamoto2025VaSP,
  title   = {VaSP: Vascular Fluid–Structure Interaction Pipeline},
  author  = {Yamamoto, Kei and Bruneau, David A. and Ring, Johannes and Dokken, Jørgen S. and Valen-Sendstad, Kristian},
  journal = {SoftwareX},
  volume  = {32},
  pages   = {102392},
  year    = {2025},
  issn    = {2352-7110},
  doi     = {10.1016/j.softx.2025.102392},
  url     = {https://www.sciencedirect.com/science/article/pii/S2352711025003589}
}
```

The paper is available [here](https://doi.org/10.1016/j.softx.2025.102392).

## License
This software is licensed under the **GNU General Public License (GPL), version 3 or (at your option) any later version**.

For more details, see the [`LICENSE`](LICENSE) file in this repository.

## Authors
* [David Bruneau](https://github.com/dbruneau-mie)
* [Johannes Ring](https://github.com/johannesring)
* [Jørgen Schartum Dokken](https://github.com/jorgensd)
* [Kei Yamamoto](https://github.com/keiyamamo)

## Issues
Please report bugs and other issues through the issue tracker at:

https://github.com/KVSlab/VaSP/issues
