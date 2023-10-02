import os
import h5py
from glob import glob
import numpy as np
import postprocessing_common_h5py
import postprocessing_common_pv

#import spectrograms as spec
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd 
import glob as glob
import re
import pyvista as pv

"""
This script takes the visualization files from TurtleFSI and outputs: 
(1) Domain-specific visualizations (wall displacement for the wall only, and velocity and pressure for the fluid only).
    This way the simualtion can be postprocessed as a CFD simulation and a Solid Mechanics Element simulation separately. 
(2) Reduced save-deg visualization if save_deg = 2 (Creates a lightweight file for faster postprocessing)
(3) Band-Pass filtered visualizations for d, v, p and strain

A "Transformed Matrix" is created as well, which stores the output data in a format that can be opened quickly when we want to create spectrograms.

Args:
    mesh_name: Name of the non-refined input mesh for the simulation. This function will find the refined mesh based on this name
    case_path (Path): Path to results from simulation
    stride: reduce output frequncy by this factor
    save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only). If we input save_deg = 1 for a simulation 
       that was run in TurtleFSI with save_deg = 2, the output from this script will be save_deg = 1, i.e only the corner nodes will be output
    dt (float): Actual time step of simulation
    start_t: Desired start time of the output files 
    end_t:  Desired end time of the output files 
    dvp: postprocess d, v, or p (displacement velocity or pressure)
    bands: list of bands for band-pass filtering displacement. 

Example: --dt 0.000679285714286 --mesh file_case16_el06 --end_t 0.951

"""
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_visualizations(case_path, mesh_name, save_deg, start_t, end_t, sampling_region, fluid_sampling_domain_ID, solid_sampling_domain_ID, r_sphere, x_sphere, y_sphere, z_sphere):

    case_name = os.path.basename(os.path.normpath(case_path)) # obtains only last folder in case_path
    visualization_path = postprocessing_common_h5py.get_visualization_path(case_path)
    

    # Get Mesh
    if save_deg == 1:
        mesh_path = case_path + "/mesh/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh
    else: 
        mesh_path = case_path + "/mesh/" + mesh_name +"_refined.h5" # Mesh path. Points to the visualization mesh with intermediate nodes 
    mesh_path_sd1 = case_path + "/mesh/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh
    mesh_path_fluid_sd1 = mesh_path_sd1.replace(".h5","_fluid_only.h5") # needed for mps
    mesh_path_solid_sd1 = mesh_path_sd1.replace(".h5","_solid_only.h5") # needed for mps
    
    visualization_separate_domain_folder = os.path.join(visualization_path,"../Visualization_separate_domain")
    visualization_hi_pass_folder = os.path.join(visualization_path,"../visualization_hi_pass")
    region_flow_metrics = [ ] 
    region_flow_metric_names = [ ]
    # 1 Read in WSS h5 files
    for WSS_filename in ["WSS.h5","RRT.h5","OSI.h5","TWSSG.h5"]:

        WSS_file = os.path.join(visualization_separate_domain_folder, WSS_filename)
        WSS_data = postprocessing_common_pv.get_data_at_idx(WSS_file,0)
        vectorData = h5py.File(WSS_file) 
        points = np.array(vectorData["Mesh/mesh/geometry"])
        element_connectivity = np.array(vectorData["Mesh/mesh/topology"])
        mesh, surf = postprocessing_common_pv.assemble_tri_mesh_arrays(points,element_connectivity)
        mesh.point_arrays['wss_quantity'] = WSS_data # Assign scalar to mesh


        points_poly = pv.PolyData(points)
        if sampling_region == "sphere":
            sphere = pv.Sphere(radius=r_sphere, center=(x_sphere, y_sphere, z_sphere))
            select = points_poly.select_enclosed_points(sphere)
        else:
            mesh_domain, surf, domains = postprocessing_common_pv.assemble_mesh_domains(mesh_path)
            mesh_domain.cell_arrays['domain_ID'] = domains # Assign scalar to mesh
            threshold = mesh_domain.threshold(value=1001,invert=True,scalars="domain_ID")
            thresh_surf = threshold.extract_surface()
            select = points_poly.select_enclosed_points(thresh_surf,check_surface=False)
    
    
        WSS_Region = mesh.point_arrays['wss_quantity'][select["SelectedPoints"]>0]
        WSS_perc99 = np.percentile(mesh.point_arrays['wss_quantity'], 99)
        WSS_max = np.max(mesh.point_arrays['wss_quantity'])
        WSS_perc99_region = np.percentile(WSS_Region, 99)
        WSS_max_region = np.percentile(WSS_Region,100)
        region_flow_metrics.append(WSS_max_region)
        region_flow_metric_names.append(WSS_filename.replace(".h5", "_max_region"))
        region_flow_metrics.append(WSS_perc99_region)
        region_flow_metric_names.append(WSS_filename.replace(".h5", "_99th_percentile_region"))
        print("Calculated regional maxima for ", WSS_filename)


        #plotter= pv.Plotter(notebook=0)
#
        #plotter.add_mesh(mesh,
        #                lighting=True, 
        #                smooth_shading=True, 
        #                specular=0.00, 
        #                diffuse=0.9,
        #                ambient=0.5, 
        #                name='contour')  
        #plotter.show(auto_close=False)  
    
    # 2 Read in Amp h5 files
    for Amp_filename in ["displacement_amplitude_25_to_1400.h5","pressure_amplitude_25_to_1400.h5","velocity_amplitude_25_to_1400.h5"]:
        
        Amp_file = os.path.join(visualization_hi_pass_folder, Amp_filename)
        Amp_csv = Amp_file.replace(".h5",".csv")
         
        # Find time index of maximum amplitude
        output_amplitudes = np.genfromtxt(Amp_csv, delimiter=',')
        #if "Pulsatile_Ramp_Cases_FC_CFD_undeformed" in visualization_hi_pass_folder:
        #    index_max = int(output_amplitudes[np.argmax(output_amplitudes[:,3]),1])
        #else:
        output_amplitudes[:,1] = range(len(output_amplitudes[:,1])) # Set indices as one column of array
        output_amplitudes=output_amplitudes[output_amplitudes[:,0]>=start_t]
        output_amplitudes=output_amplitudes[output_amplitudes[:,0]<=end_t]
    
        index_max = int(output_amplitudes[np.argmax(output_amplitudes[:,3]),1])
        
        print(index_max)

        # Get original amplitude data and build mesh
        Amp_data = postprocessing_common_pv.get_data_at_idx(Amp_file,index_max)
        vectorData = h5py.File(Amp_file) 
        points = np.array(vectorData["Mesh/0/mesh/geometry"])
        element_connectivity = np.array(vectorData["Mesh/0/mesh/topology"])
        mesh, surf = postprocessing_common_pv.assemble_mesh_arrays(points,element_connectivity)
        
        # For pressure, take only value. For vector, take magnitude
        if Amp_data.shape[1] ==1:
            mesh.point_arrays['Amp_quantity'] = Amp_data # Assign scalar to mesh
        else:
            mesh.point_arrays['Amp_quantity'] = (Amp_data[:,0]**2 + Amp_data[:,1]**2 + Amp_data[:,2]**2)**0.5
       
        # Get marked mesh or sphere and use this to select only points inside this region
        points_poly = pv.PolyData(points)
        if sampling_region == "sphere":
            sphere = pv.Sphere(radius=r_sphere, center=(x_sphere, y_sphere, z_sphere))
            select = points_poly.select_enclosed_points(sphere)
        else:
            mesh_domain, surf, domains = postprocessing_common_pv.assemble_mesh_domains(mesh_path)
            mesh_domain.cell_arrays['domain_ID'] = domains # Assign scalar to mesh
            threshold = mesh_domain.threshold(value=1001,invert=True,scalars="domain_ID")
            thresh_surf = threshold.extract_surface()
            select = points_poly.select_enclosed_points(thresh_surf,check_surface=False)
    
    
        Amp_Region = mesh.point_arrays['Amp_quantity'][select["SelectedPoints"]>0]
        Amp_perc99 = np.percentile(mesh.point_arrays['Amp_quantity'], 99)
        Amp_max = np.max(mesh.point_arrays['Amp_quantity'])
        Amp_perc99_region = np.percentile(Amp_Region, 99)
        Amp_max_region = np.percentile(Amp_Region,100)
        #print(Amp_perc99_region)
        #print(Amp_perc99)
        #print(np.argmax(mesh.point_arrays['Amp_quantity']))
#
        #print(len(mesh.point_arrays['Amp_quantity']))
        #print(len(Amp_data))
        region_flow_metrics.append(Amp_max_region)
        region_flow_metric_names.append(Amp_filename.replace(".h5", "_max_region"))
        region_flow_metrics.append(Amp_perc99_region)
        region_flow_metric_names.append(Amp_filename.replace(".h5", "_99th_percentile_region"))

        print("Calculated regional maxima for ", Amp_filename)


    # 3 Read in spi tecplot files

    tecplot_files = glob.glob(visualization_separate_domain_folder + "/*.tec")
    tecplot_files.sort(key=natural_keys)


    for file in tecplot_files:
        file_section = "points_vals"
        xyzval = []
        element_connectivity = []
        with open(file, 'r') as a:
            for line in a.readlines():
                A = re.match(r'TITLE = (.*$)', line, re.M | re.I)
                B = re.match(r'VARIABLES = (.*$)', line, re.M | re.I)
                C = re.match(r'ZONE (.*$)', line, re.M | re.I)
    
                if A or B or C:
                        continue
                elif line == '\n':
                    file_section = "connectivity"
                elif file_section == "points_vals":
                    xyzval.append([float(s) for s in line.split()])
                elif file_section == "connectivity":
                    element_connectivity.append([int(s) for s in line.split()])
                else:
                    print("unknown condition.....")
        #arrays = np.concatenate(arrays)
        xyzval = np.array(xyzval)
        points = xyzval[:,0:3] 
    
        element_connectivity = np.array(element_connectivity) - 1
                
        if element_connectivity.shape[1] == 4:
            mesh, surf = postprocessing_common_pv.assemble_mesh_arrays(points,element_connectivity)
        else:
            mesh, surf = postprocessing_common_pv.assemble_tri_mesh_arrays(points,element_connectivity)


        #mesh, surf = postprocessing_common_pv.assemble_mesh(mesh_path_sd1)
        mesh.point_arrays['spi'] = xyzval[:,3] # Assign scalar to mesh
    
    
        points_poly = pv.PolyData(points)
        if sampling_region == "sphere":
            sphere = pv.Sphere(radius=r_sphere, center=(x_sphere, y_sphere, z_sphere))
            select = points_poly.select_enclosed_points(sphere)
        else:
            mesh_domain, surf, domains = postprocessing_common_pv.assemble_mesh_domains(mesh_path)
            mesh_domain.cell_arrays['domain_ID'] = domains # Assign scalar to mesh
            threshold = mesh_domain.threshold(value=1001,invert=True,scalars="domain_ID")
            thresh_surf = threshold.extract_surface()
            select = points_poly.select_enclosed_points(thresh_surf,check_surface=False)
    
    
        SPI_Region = mesh.point_arrays['spi'][select["SelectedPoints"]>0]
        SPI_perc99 = np.percentile(mesh.point_arrays['spi'], 99)
        SPI_max = np.max(mesh.point_arrays['spi'])
        SPI_perc99_region = np.percentile(SPI_Region, 99)
        SPI_max_region = np.percentile(SPI_Region,100)
        SPI_filename = os.path.basename(file)
        if "_d." in SPI_filename:
            SPI_filename = "displacement_spi"
        elif "_wss." in SPI_filename:
            SPI_filename = "wss_spi"
        elif "_v." in SPI_filename:
            SPI_filename = "velocity_spi"
        
        region_flow_metrics.append(SPI_max_region)
        region_flow_metric_names.append(SPI_filename + "_max_region")
        region_flow_metrics.append(SPI_perc99_region)
        region_flow_metric_names.append(SPI_filename + "_99th_percentile_region")
        print("Calculated regional maxima for ", SPI_filename)

    region_flow_metrics_file = os.path.join(visualization_separate_domain_folder, "region_flow_metrics.csv")
    df = pd.DataFrame(np.array([region_flow_metrics]),columns=region_flow_metric_names)
    df.to_csv(region_flow_metrics_file)


    

if __name__ == '__main__':
    # Get Command Line Arguments and Input File Paths
    case_path, mesh_name, save_deg, _, _, start_t, end_t, _, _, sampling_region, fluid_sampling_domain_ID, solid_sampling_domain_ID, r_sphere, x_sphere, y_sphere, z_sphere = postprocessing_common_pv.read_command_line_region()
    create_visualizations(case_path, mesh_name, save_deg, start_t, end_t, sampling_region, fluid_sampling_domain_ID, solid_sampling_domain_ID, r_sphere, x_sphere, y_sphere, z_sphere)