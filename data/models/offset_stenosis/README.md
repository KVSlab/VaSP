# Offset stenosis model
TODO: provide analytical discription of the stenosis model.
The volune mesh can be created with the following command:

```
fsipy-meshing -i models/offset_stenosis/offset_stenosis.stl -cm False -f True -fli 0 -flo 4 -st constant -el 1 -ra -v -c 3.8 -nbf 1 -nbs 1
```