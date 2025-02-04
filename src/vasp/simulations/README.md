# Simulations folder

This folder contains five problems files 1. `cylinder.py`, 2. `offset_stenosis.py`, 3. `predeform.py`, 4. `aneurysm.py`, and 5. `avf.py` that can be used to run FSI simulations with `turtleFSI`. All problems file follow the general structure of a `turtleFSI` problem file, as described in the [turtleFSI documentation](https://turtlefsi2.readthedocs.io/en/latest/using_turtleFSI.html#create-your-own-problem-file). There is also one common file `simulation_common.py` that contains functions commonly used in the problem files. Additionally, there are two files, `FC_MCA_10` and `FC_Pressure` that contain the Fourier coefficients for the inflow and pressure boundary conditions, respectively.

Here is a brief description of each problem file:

1. `cylinder.py`: This is a simple problem to test the basic functionality of `turtleFSI` that can be run on a standard laptop.

2. `offset_stenosis.py`: This is a simple yet physiologically relevant problem that simulates blood flow through a stenosed artery. This problem is further described in the [tutorials](https://kvslab.github.io/VaSP/offset_stenosis.html#tutorial-offset-stenosis).

3. `predeform.py`: This problem is used to perform a pre-deformation simulation of a medical image-based surface mesh. The pre-deformation is necessary to generate a realistic initial configuration for the FSI simulation.

4. `aneurysm.py`: This problem simulates blood flow and its interaction with vascular wall through an artery with an aneurysm. This problem is further described in the [tutorials](https://kvslab.github.io/VaSP/aneurysm.html#tutorial-aneurysm).

5. `avf.py`: This problem  simulates blood flow and its interaction with vascular wall through arteriovenous fistula (AVF). This problem is further described in the [tutorials](https://kvslab.github.io/VaSP/avf.html#tutorial-avf).