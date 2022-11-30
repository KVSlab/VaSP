import pyvista as pv
from pyvista import examples
import numpy as np
import vtk
import postprocessing_common_pv
import sys
import os

pv.set_plot_theme("document")

CameraPosition = np.array(list(map(float, sys.argv[3].strip('[]').split(','))))
CameraFocalPoint = np.array(list(map(float, sys.argv[4].strip('[]').split(','))))
CameraViewUp = np.array(list(map(float, sys.argv[5].strip('[]').split(',')) ))
CameraParallelScale = float(sys.argv[6])

case_path = sys.argv[1]
mesh_name = sys.argv[2]
mesh_path = os.path.join(case_path,"mesh",mesh_name + ".h5")
mesh, surf = postprocessing_common_pv.assemble_mesh(mesh_path)


# This option will only work with onscreen rendering (needs to be run locally)
det_cam_position=0
contour_val=float(sys.argv[7])
time_step_idx=int(sys.argv[8])


print("read camera position:")
print(CameraPosition)
print(CameraFocalPoint)
print(CameraViewUp)

cpos = [CameraPosition,
        CameraFocalPoint,
        CameraViewUp]

# Define necessary folders
visualization_path = postprocessing_common_pv.get_visualization_path(case_path)
visualization_separate_domain_folder = os.path.join(visualization_path,"../Visualization_separate_domain")
visualization_hi_pass_folder = os.path.join(visualization_path,"../visualization_hi_pass")
visualization_sd1_folder = os.path.join(visualization_path,"../Visualization_sd1")
image_folder =  os.path.join(visualization_path,"../Images")
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

image_path = os.path.join(image_folder, 'pyvista_vel_contour_tbsmooth_'+str(contour_val)+'.png')
res_path = os.path.join(visualization_sd1_folder, "velocity_save_deg_1.h5")

# Read in vector at specific timestep
u_vec = postprocessing_common_pv.get_data_at_idx(res_path, time_step_idx)

# Compute normals (smooths the surface)
surf.compute_normals(inplace=True)
u_mag = np.linalg.norm(u_vec, axis=1) # take magnitude
mesh.point_arrays['u_mag'] = u_mag # Assign scalar to mesh
contour = mesh.contour([contour_val], scalars="u_mag")
if contour.n_points > 0:
    contour = postprocessing_common_pv.vtk_taubin_smooth(contour) # smooth the surface
    contour.compute_normals(inplace=True) # Compute normals


silhouette = dict(
    color='black',
    line_width=4.0,decimate=None
)

silhouette_outer = dict(
    color='black',
    line_width=3.0,decimate=None
)

if det_cam_position == 1:
    plotter= pv.Plotter(notebook=0)
else:
    plotter= pv.Plotter(off_screen=True)

plotter.add_mesh(surf, 
                color='snow', 
                silhouette=silhouette_outer, 
                show_scalar_bar=False,
                opacity=0.35,
                lighting=True, 
                smooth_shading=True, 
                specular=0.00, 
                diffuse=0.9,
                ambient=0.5, 
                name='surf',
                culling='front'
                )

if contour.n_points > 0:
    plotter.add_mesh(contour,
                    color='red', 
                    silhouette=silhouette, 
                    clim=[0.0, 2.0], # NOTE this is for u_mag only!
                    lighting=True, 
                    smooth_shading=True, 
                    specular=0.00, 
                    diffuse=0.9,
                    ambient=0.5, 
                    name='contour')

plotter.set_background('white')

plotter.show(auto_close=False)  

if det_cam_position == 1:
    plotter.show(auto_close=False) 
    print(plotter.camera_position)
else:
    plotter.camera_position = cpos
    plotter.show(screenshot=image_path)  