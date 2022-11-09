# !/usr/bin/python

# Local imports
from vmtkmeshgeneratorfsi import *

from os import path
from vmtk import vmtkscripts
import vtk
import numpy as np


#################
# USER INPUTS
# ################################################################################
file_name = "offset_stenosis"
ifile_surface = "surfaces/"+file_name+".stl"
are_caps = 1

# ################################################################################
# END OF USER INPUTS
########################


# Read vtp surface file (vtp) ##################################################
reader = vmtkscripts.vmtkSurfaceReader()
reader.InputFileName = ifile_surface
reader.Execute()
surface = reader.Surface
################################################################################

# Read vtp surface file (vtp) ##################################################
reader = vmtkscripts.vmtkSurfaceReader()
reader.InputFileName = ifile_surface
reader.Execute()
surface = reader.Surface
################################################################################

if are_caps == 1:
    clipp = vmtkscripts.vmtkSurfaceClipper()
    clipp.Surface = surface
    clipp.Execute()
    surface_no_cap = clipp.Surface
    
"""
viewer = vmtkscripts.vmtkSurfaceViewer()
viewer.Surface = surface_no_cap
viewer.Execute()
"""

writ = vmtkscripts.vmtkSurfaceWriter()
writ.Surface=surface_no_cap
writ.Format='stl'
writ.OutputFileName = "surfaces/"+file_name+"_nocap.stl"
writ.Execute()
