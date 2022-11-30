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

contour_val=float(sys.argv[7])
contour_val_hi=float(sys.argv[8])
time_step_idx = int(sys.argv[9])

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

image_path = os.path.join(image_folder, 'pyvista_q_contour_lo'+str(contour_val)+"_hi"+str(contour_val_hi)+'.png')
res_path = os.path.join(visualization_hi_pass_folder, "velocity_0_to_25.h5")
res_hi_path = os.path.join(visualization_hi_pass_folder, "velocity_25_to_100000.h5")


# Read hi and lo pass velocity
u_vec = postprocessing_common_pv.get_data_at_idx(res_path, time_step_idx)
u_hi_vec = postprocessing_common_pv.get_data_at_idx(res_hi_path, time_step_idx)

surf.compute_normals(inplace=True)
mesh_hi = mesh.copy()
mesh.point_arrays['u'] = u_vec
mesh_hi.point_arrays['u'] = u_hi_vec
mesh = postprocessing_common_pv.q_criterion_nd(mesh)
mesh_hi = postprocessing_common_pv.q_criterion_nd(mesh_hi)

contour = mesh.contour([contour_val], scalars="qcriterion")
if contour.n_points > 0:
    contour = contour.smooth(n_iter=100)
    contour.compute_normals(inplace=True)

contour_hi = mesh_hi.contour([contour_val_hi], scalars="qcriterion")
if contour_hi.n_points > 0:
    contour_hi = contour_hi.smooth(n_iter=100)
    contour_hi.compute_normals(inplace=True)

#q_lo = pv.read("c9_surf/isosurface_Q_lo.vtp")
#outer_surf = pv.read("c9_surf/Outer_surf_normals.vtp")
#
#q_lo.points =  q_lo.points*1000 # NEED to scale to mm for some vmtk scripts to work.
#outer_surf.points =  outer_surf.points*1000 # NEED to scale to mm for some vmtk scripts to work.
#
#
#surf = q_lo.extract_geometry()
#q_lo_smooth = surf.smooth(n_iter=100)
#surf = outer_surf.extract_geometry()
#outer_surf_smooth = surf.smooth(n_iter=1, relaxation_factor=0.1, convergence=0.0, edge_angle=0.1, feature_angle=0.1, boundary_smoothing=False)
#
silhouette_hi = dict(
    color='black',
    line_width=3.0,decimate=None
)
silhouette = dict(
    color='gray',
    line_width=3.0,decimate=None
)
silhouette_outer = dict(
    color='black',
    line_width=3.0,decimate=None
)

plotter= pv.Plotter(off_screen=True)

plotter.add_mesh(surf, 
                color='white', 
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
                    color='gray', 
                    opacity=0.5,
                    smooth_shading=True, 
                    silhouette=silhouette, 
                    specular=0.0, 
                    diffuse=1.0,
                    ambient=1.0,
                    name='contour_lo')

if contour_hi.n_points > 0:
    plotter.add_mesh(contour_hi,
                    color='purple', 
                    opacity=0.75,
                    smooth_shading=True, 
                    silhouette=silhouette_hi, 
                    specular=0.0, 
                    diffuse=1.0,
                    ambient=1.0,
                    name='contour_hi')
plotter.show(auto_close=False)  

plotter.camera_position = cpos
plotter.show(screenshot=image_path) 