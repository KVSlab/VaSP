"""Top-level package for VaMPy."""
from importlib.metadata import metadata

# Imports from post-processing
#from .automatedPostprocessing import compute_flow_and_simulation_metrics
#from .automatedPostprocessing import postprocessing_common
# Imports from pre-processing
from .automatedPreprocessing import automated_preprocessing
from .automatedPreprocessing import preprocessing_common
from .automatedPreprocessing import predeform_mesh
# Imports from simulation scripts
#from .simulation import Aneurysm
#from .simulation import AVF
#from .simulation import simulation_common


meta = metadata("fsipy")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = [
#    "compute_flow_and_simulation_metrics",
#    "postprocessing_common",
    "automated_preprocessing",
    "preprocessing_common",
    "predeform_mesh",
#    "Aneurysm",
#    "AVF",
#    "Stenosis",
#    "simulation_common",
]
