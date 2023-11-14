import os
import time
from argparse import ArgumentParser
import configparser

import numpy as np
from scipy.signal import butter, filtfilt, spectrogram, periodogram
from scipy.interpolate import RectBivariateSpline
from scipy.stats import entropy
from scipy.spatial import cKDTree as KDTree

try:
    import pyvista as pv
    import postprocessing_common_pv
except ImportError:
    print("Could not import pyvista and/or vtk. Install these packages or use 'RandomPoint' sampling for spectrograms.")

import postprocessing_common_h5py
from chroma_filters import normalize, chroma_filterbank

"""
This library contains helper functions for creating spectrograms. All functions authored by David Bruneau unless otherwise specified.
"""



def read_spec_config(config_file,dvp):

    config = configparser.ConfigParser()
    with open(config_file) as stream:
        config.read_string("[Spectrogram_Parameters]\n" + stream.read()) 
    
    overlapFrac = float(config.get("Spectrogram_Parameters", "overlapFrac"))
    window = config.get("Spectrogram_Parameters", "window")
    n_samples = int(config.get("Spectrogram_Parameters", "n_samples"))
    nWindow_per_sec = int(config.get("Spectrogram_Parameters", "nWindow_per_sec"))
    lowcut = float(config.get("Spectrogram_Parameters", "lowcut"))
    # Thresholds and file names dependant on whether processing d, v or p
    if dvp == "d":
        thresh_val = int(config.get("Spectrogram_Parameters", "thresh_val_d"))# log(m**2)
        max_plot = int(config.get("Spectrogram_Parameters", "max_plot_d"))
        amplitude_file_name = str(config.get("Spectrogram_Parameters", "amplitude_file_d"))
    elif dvp == "wss":
        thresh_val = int(config.get("Spectrogram_Parameters", "thresh_val_wss"))# log(Pa**2) this hasnt been investigated for wss
        max_plot = int(config.get("Spectrogram_Parameters", "max_plot_wss"))
        amplitude_file_name = str(config.get("Spectrogram_Parameters", "amplitude_file_wss"))
    elif dvp == "v":
        thresh_val = int(config.get("Spectrogram_Parameters", "thresh_val_v")) # log((m/s)**2)
        max_plot = int(config.get("Spectrogram_Parameters", "max_plot_v"))
        amplitude_file_name = str(config.get("Spectrogram_Parameters", "amplitude_file_v"))
    else: 
        thresh_val = int(config.get("Spectrogram_Parameters", "thresh_val_p")) # log((Pa)**2)
        max_plot = int(config.get("Spectrogram_Parameters", "max_plot_p"))
        amplitude_file_name = str(config.get("Spectrogram_Parameters", "amplitude_file_p"))
    flow_rate_file_name = str(config.get("Spectrogram_Parameters", "flow_rate_file_name"))

    return overlapFrac, window, n_samples, nWindow_per_sec, lowcut, thresh_val, max_plot, amplitude_file_name, flow_rate_file_name

def read_command_line_spec():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    parser.add_argument('--case', type=str, default="cyl_test", help="Path to simulation results",
                        metavar="PATH")
    parser.add_argument('--mesh', type=str, default="artery_coarse_rescaled", help="Mesh File Name",
                        metavar="PATH")
    parser.add_argument('--save_deg', type=int, default=2, help="Input save_deg of simulation, i.e whether the intermediate P2 nodes were saved. Entering save_deg = 1 when the simulation was run with save_deg = 2 will result in only the corner nodes being used in postprocessing")
    parser.add_argument('--stride', type=int, default=1, help="Desired frequency of output data (i.e to output every second step, stride = 2)")    
    parser.add_argument('--start_t', type=float, default=0.0, help="Start time of simulation (s)")
    parser.add_argument('--end_t', type=float, default=0.05, help="End time of simulation (s)")
    parser.add_argument('--lowcut', type=float, default=25, help="High pass filter cutoff frequency (Hz)")
    parser.add_argument('--ylim', type=float, default=800, help="y limit of spectrogram graph")
    parser.add_argument('--sampling_region', type=str, default="sphere", help="sample within -sphere, or within specific -domain") 
    parser.add_argument('--fluid_sampling_domain_ID', type=int, default=1, help="Domain ID for fluid region to be sampled (need to input a labelled mesh with this ID)")   
    parser.add_argument('--solid_sampling_domain_ID', type=int, default=2, help="Domain ID for solid region to be sampled (need to input a labelled mesh with this ID)")    
    parser.add_argument('--r_sphere', type=float, default=1000000, help="Sphere in which to include points for spectrogram, this is the sphere radius")
    parser.add_argument('--x_sphere', type=float, default=0.0, help="Sphere in which to include points for spectrogram, this is the x coordinate of the center of the sphere (in m)")
    parser.add_argument('--y_sphere', type=float, default=0.0, help="Sphere in which to include points for spectrogram, this is the y coordinate of the center of the sphere (in m)")
    parser.add_argument('--z_sphere', type=float, default=0.0, help="Sphere in which to include points for spectrogram, this is the z coordinate of the center of the sphere (in m)")
    parser.add_argument('--dvp', type=str, default="v", help="Quantity to postprocess, input v for velocity, d for displacement, p for pressure, or wss for wall shear stress")
    parser.add_argument('--Re_a', type=float, default=0.0, help="Assuming linearly increasing Reynolds number: Re(t) = Re_a*t + Re_b . if both Re_a and Re_b are 0, don't plot against Re.")
    parser.add_argument('--Re_b', type=float, default=0.0, help="Assuming linearly increasing Reynolds number: Re(t) = Re_a*t + Re_b . if both Re_a and Re_b are 0, don't plot against Re.")
    parser.add_argument('--interface_only', type=bool, default=False, help="True gives you only spectrogram for the fluid-solid interface, 'False' gives you the volumetric spectrogram for all fluid in the sac or all the nodes thru the wall")
    parser.add_argument('--sampling_method', type=str, default="RandomPoint", help="'RandomPoint' (choose random nodes), 'SinglePoint' (choose Single Point with ID = 'point_id') or 'Spatial' (ensures uniform spatial sampling, e.g, in the case of fluid boundary layer the sampling will not bias towards the BL)")
    parser.add_argument('--component', type=str, default="mag", help="x, y, z or mag (magnitude)")
    parser.add_argument('--n_samples', type=int, default=10000, help="Number of samples for spectrogram (ignored for SinglePoint sampling)")
    parser.add_argument('--point_id', type=int, default=-1000000, help="Point ID for SinglePoint sampling")

    args = parser.parse_args()

    return args.case, args.mesh, args.save_deg, args.stride, args.start_t, args.end_t, args.lowcut, args.ylim, args.sampling_region, args.fluid_sampling_domain_ID, args.solid_sampling_domain_ID, args.r_sphere, args.x_sphere, args.y_sphere, args.z_sphere, args.dvp, args.Re_a, args.Re_b, args.interface_only, args.sampling_method, args.component, args.n_samples, args.point_id


def read_spectrogram_data(case_path, mesh_name, save_deg, stride, start_t, end_t, n_samples, ylim, sampling_region, fluid_sampling_domain_ID, solid_sampling_domain_ID,
                          r_sphere, x_sphere, y_sphere, z_sphere, dvp, interface_only,component,point_id,flow_rate_file_name=None,sampling_method="RandomPoint"):
 

    start = time.time()
    
    case_name = os.path.basename(os.path.normpath(case_path)) # obtains only last folder in case_path
    visualization_path = postprocessing_common_h5py.get_visualization_path(case_path)

    
    
    #--------------------------------------------------------
    # 1. Get names of relevant directories, files
    #--------------------------------------------------------
    
    if save_deg == 1:
        mesh_path = case_path + "/mesh/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh
    else: 
        mesh_path = case_path + "/mesh/" + mesh_name +"_refined.h5" # Mesh path. Points to the visualization mesh with intermediate nodes 
    mesh_path_fluid = mesh_path.replace(".h5","_fluid_only.h5") # needed for formatting SPI data
    
    formatted_data_folder_name = "res_"+case_name+'_stride_'+str(stride)+"t"+str(start_t)+"_to_"+str(end_t)+"save_deg_"+str(save_deg)
    formatted_data_folder = os.path.join(case_path,formatted_data_folder_name)
    visualization_separate_domain_folder = os.path.join(visualization_path,"../Visualization_separate_domain")
    visualization_hi_pass_folder = os.path.join(visualization_path,"../visualization_hi_pass")
    
    imageFolder = os.path.join(visualization_path,"../Spectrograms")
    if not os.path.exists(imageFolder):
        os.makedirs(imageFolder)
    
    #  output folder and filenames (if these exist already, they will not be re-generated)
    output_file_name = case_name+"_"+ dvp+"_"+component+".npz" 
    formatted_data_path = formatted_data_folder+"/"+output_file_name

    t = time.time()
    print("retrieved names")
    print(t - start)    

    #--------------------------------------------------------
    # 2. Prepare data
    #--------------------------------------------------------
    
    # If the output file exists, don't re-make it
    if os.path.exists(formatted_data_path):
        print('path found!')
    elif dvp == "wss":
        _ = postprocessing_common_h5py.create_transformed_matrix(visualization_separate_domain_folder, formatted_data_folder, mesh_path_fluid, case_name, start_t,end_t,dvp,stride)
    else: 
        # Make the output h5 files with dvp magnitudes
        _ = postprocessing_common_h5py.create_transformed_matrix(visualization_path, formatted_data_folder, mesh_path, case_name, start_t,end_t,dvp,stride)

    t = time.time()
    print("made matrix")
    print(t - start)  
    # For spectrograms, we only want the magnitude
    df = postprocessing_common_h5py.read_npz_files(formatted_data_path)
    t = time.time()
    print("read matrix")
    print(t - start)         

    
    # We want to find the points in the sac, so we use a sphere to roughly define the sac.
    sac_center = np.array([x_sphere, y_sphere, z_sphere])  
    if dvp == "wss":
        outFile = os.path.join(visualization_separate_domain_folder,"WSS_ts.h5")
        surfaceElements, coords = postprocessing_common_h5py.get_surface_topology_coords(outFile)
    else:
        coords = postprocessing_common_h5py.get_coords(mesh_path)

    if sampling_region == "sphere":
        # Get wall and fluid ids
        fluidIDs, wallIDs, allIDs = postprocessing_common_h5py.get_domain_ids(mesh_path)
        interfaceIDs = postprocessing_common_h5py.get_interface_ids(mesh_path)    
        sphereIDs = find_points_in_sphere(sac_center,r_sphere,coords)
        # Get nodes in sac only
        allIDs=list(set(sphereIDs).intersection(allIDs))
        fluidIDs=list(set(sphereIDs).intersection(fluidIDs))
        wallIDs=list(set(sphereIDs).intersection(wallIDs))
        interfaceIDs=list(set(sphereIDs).intersection(interfaceIDs))

    elif sampling_region == "domain": # To use this option, must input mesh with domain markers and indicate which domain represents the desired fluid region 
                                      # for the spectrogram (fluid_sampling_domain_ID) and which domain represents desired solid region (solid_sampling_domain_ID)
        fluidIDs, wallIDs, allIDs = postprocessing_common_h5py.get_domain_ids_specified_region(mesh_path,fluid_sampling_domain_ID,solid_sampling_domain_ID)
        interfaceIDs_set = set(fluidIDs) - (set(fluidIDs) - set(wallIDs))
        interfaceIDs = list(interfaceIDs_set)
    else:
        #print("Need to specify sampling method as -sphere or -domain, got " + sampling_region)
        raise Exception("Need to specify sampling method as -sphere or -domain, got " + sampling_region)

    if dvp == "wss":
        region_ids = sphereIDs  # for wss spectrogram, we use all the nodes within the sphere because the input df only includes the wall
    elif interface_only:
        region_ids = interfaceIDs   # Use only the interface IDs
        dvp=dvp+"_interface"
    elif dvp == "d":
        region_ids = wallIDs    # For displacement spectrogram, we need to take only the wall IDs
    else:
        region_ids = fluidIDs   # For pressure and velocity spectrogram, we need to take only the fluid IDs

    t = time.time()
    print("got IDs")
    print(t - start)  

    # Sample data (reduce compute time by random sampling)
    if sampling_method == "RandomPoint":
        idx_sampled = np.random.choice(region_ids, n_samples)
    elif sampling_method == "SinglePoint":
        idx_sampled = [point_id]
        case_name=case_name+"_"+sampling_method+"_"+str(point_id)
        print("Single Point spectrogram for point: "+str(point_id))
    elif sampling_method == "Spatial": # This method only works with a specified sphere
        mesh, surf = postprocessing_common_pv.assemble_mesh(mesh_path)

        bounds = [x_sphere-r_sphere, x_sphere+r_sphere, y_sphere-r_sphere, y_sphere+r_sphere, z_sphere-r_sphere, z_sphere+r_sphere]


        def generate_points(bounds, subdivisions=50):
            x_points=np.linspace(bounds[0], bounds[1],num=subdivisions)
            y_points=np.linspace(bounds[2], bounds[3],num=subdivisions)
            z_points=np.linspace(bounds[4], bounds[5],num=subdivisions)
            points = np.zeros((subdivisions**3,3))
            for i in range(subdivisions):
                for j in range(subdivisions):
                    for k in range(subdivisions):
                        ijk = (subdivisions**2)*i + subdivisions*j + k
                        points[ijk,:] = [x_points[i], y_points[j], z_points[k]]
            return points        

        t = time.time()
        print("got surf")
        print(t - start)  
        points = generate_points(bounds,subdivisions=50)

        t = time.time()
        print("generated points")
        print(t - start)  

        point_cloud=pv.PolyData(points)
        sphere = pv.Sphere(radius=r_sphere, center=(x_sphere, y_sphere, z_sphere))
        surf_sel = point_cloud.select_enclosed_points(surf, tolerance=0.01).threshold(value=0.5,scalars='SelectedPoints')
        sphere_sel = surf_sel.select_enclosed_points(sphere, tolerance=0.01).threshold(value=0.5,scalars='SelectedPoints')


        t = time.time()
        print("shaped point cloud")
        print(t - start)  
        tree = KDTree(coords)
        _, idx = tree.query(sphere_sel.points) #find closest node to the points in the equispaced points  


        t = time.time()
        print("queried kdtree")
        print(t - start)  

        
        # Make sure this is the correct order
        equispaced_fluid_ids = [x for x in idx if x in region_ids]
        idx_sampled = np.random.choice(equispaced_fluid_ids, n_samples)

        idx_sampled_set = set(idx_sampled)
        # compare the length and print if the list contains duplicates
        if len(idx_sampled) != len(idx_sampled_set):
            print("duplicates found in the list")
        else:
            print("No duplicates found in the list")
        
        #sampled_point_cloud=pv.PolyData(coords[idx_sampled])
        #plotter= pv.Plotter(off_screen=True)
        
        #plotter.add_mesh(surf, 
        #                color='red', 
        #                show_scalar_bar=False,
        #                opacity=0.05)
        #plotter.add_points(sampled_point_cloud, render_points_as_spheres=True, point_size=0.03)
        #plotter.show(auto_close=False)  
        #plotter.show(screenshot=imageFolder + "/points"+str(idx_sampled[0])+".png")

        case_name=case_name+"_"+sampling_method
    
    t = time.time()
    print("obtained sample points")
    print(t - start)  

    df = df.iloc[idx_sampled]
    dvp=dvp+"_"+component
    dvp=dvp+"_"+str(n_samples)
    t = time.time()
    print("sampled dataframe")
    print(t - start)  
       
    return dvp, df, case_name, case_path, imageFolder, visualization_hi_pass_folder

def get_location_from_pointID(probeID,polydata):
    nearestPointLoc = [10000,10000,10000]
    polydata.GetPoint(probeID,nearestPointLoc)
    return nearestPointLoc


def find_points_in_sphere(cent,rad,coords):
    
    # Calculate vector from  center to each node in the mesh
    x=coords[:,0]-cent[0]
    y=coords[:,1]-cent[1]
    z=coords[:,2]-cent[2]

    # Assemble into vector ((vectorPoint))
    vectorPoint=np.c_[x,y,z]

    # Calculate distance from each mesh node to center
    radius_nodes = np.sqrt(x**2+y**2+z**2)

    # get all points in sphere
    points_in_sphere_list=[index for index,value in enumerate(radius_nodes) if value < rad]
    points_in_sphere = np.array(points_in_sphere_list)

    return points_in_sphere


def shift_bit_length(x):
    '''
    Author: Dan MacDonald
    round up to nearest pwr of 2
    https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
    '''
    return 1<<(x-1).bit_length()

def get_psd(dfNearest,fsamp,scaling="density"):
    if dfNearest.shape[0] > 1:
        #print("> 1")
        for each in range(dfNearest.shape[0]):
            row = dfNearest.iloc[each]
            f, Pxx = periodogram(row,fs=fsamp,window='blackmanharris',scaling=scaling)
            if each == 0:
                Pxx_matrix = Pxx
            else:
                Pxx_matrix = Pxx_matrix + Pxx     
                # Pxx_matrix = np.dstack((Pxx_matrix,Pxx))
        Pxx_mean = Pxx_matrix/dfNearest.shape[0]     
    else:
        #print("<= 1")

        f,Pxx_mean = periodogram(dfNearest.iloc[0],fs=fsamp,window='blackmanharris')

    return Pxx_mean, f

def get_spectrogram(dfNearest,fsamp,nWindow,overlapFrac,window,start_t,end_t, scaling='spectrum', interpolate = False):
    ''' 
    Author: Daniel Macdonald
    Calculates spectrogram
    input dfNearest is a pandas df of shape (num_points, num_timesteps)
    fsamp is sampling frequency
    Use scaling = 'angle' for phase

    scaling{ ‘density’, ‘spectrum’ }, optional: power spectral density (‘density’) where Sxx has units of V**2/Hz or power spectrum (‘spectrum’) where Sxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density’.

    '''
    NFFT = shift_bit_length(int(dfNearest.shape[1]/nWindow)) # Could change to /5
    #print(dfNearest.shape[0])

    if dfNearest.shape[0] > 1:
        #print("> 1")
        for each in range(dfNearest.shape[0]):
            row = dfNearest.iloc[each]
            #freqs,bins,Pxx = spectrogram(row,\
            #    fs=fsamp)#,nperseg=NFFT,noverlap=int(overlapFrac*NFFT))#,nfft=2*NFFT,window=window)#,scaling=scaling) 
            freqs,bins,Pxx = spectrogram(row,\
                fs=fsamp,nperseg=NFFT,noverlap=int(overlapFrac*NFFT),nfft=2*NFFT,window=window,scaling=scaling) 
            #print(np.max(Pxx))
            if each == 0:
                Pxx_matrix = Pxx
            else:
                Pxx_matrix = Pxx_matrix + Pxx     
                # Pxx_matrix = np.dstack((Pxx_matrix,Pxx))
        Pxx_mean = Pxx_matrix/dfNearest.shape[0]     
    else:
        #print("<= 1")

        freqs,bins,Pxx_mean = spectrogram(dfNearest.iloc[0],\
            fs=fsamp,nperseg=NFFT,noverlap=int(overlapFrac*NFFT),nfft=2*NFFT,window=window,scaling=scaling) 

    if interpolate == True:
        interp_spline = RectBivariateSpline(freqs, bins, Pxx_mean, kx=3, ky=3)
        bins = np.linspace(start_t,end_t,100) #arange(-xmax, xmax, dx2)
        # freqs = np.linspace(0,freqs.max(),100) #np.arange(-ymax, ymax, dy2)
        Pxx_mean = interp_spline(freqs, bins)
        print('bins shape, freqs shape, pxx shape', bins.shape, freqs.shape, Pxx_mean.shape)
    
    Pxx_mean[Pxx_mean<0] = 1e-16
    return Pxx_mean, freqs, bins

def spectrogram_scaling(Pxx_mean,lower_thresh):
    #     Author: Daniel Macdonald
    Pxx_scaled = np.log(Pxx_mean)
    max_val = np.max(Pxx_scaled)
    min_val = np.min(Pxx_scaled)
    print('Pxx_scaled max', max_val)
    print('Pxx_scaled max', min_val)
    print('Pxx threshold', lower_thresh)
    Pxx_threshold_indices = Pxx_scaled < lower_thresh
    Pxx_scaled[Pxx_threshold_indices] = lower_thresh
    return Pxx_scaled, max_val, min_val, lower_thresh

def butter_bandpass(lowcut, highcut, fs, order=5, btype='band'):
    '''
    Note: if highcut selected, 'highcut' is not used
    lowcut = cutoff frequency for low cut
    highcut = cutoff frequency for high cut
    fs is samples per second
    returns filter coeff for butter_bandpass_filter function
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if btype == 'band':
        b, a = butter(order, [low, high], btype='band')
    elif btype == 'stop':
        b, a = butter(order, [low, high], btype='bandstop')
    elif btype == 'highpass':
        b, a = butter(order, low, btype='highpass')
    elif btype == 'lowpass':
        b, a = butter(order, high, btype='lowpass')
    elif 'pass' in btype:
        b, a = butter(order, [low, high], btype='bandpass')
    return b, a 

def butter_bandpass_filter(data, lowcut=25.0, highcut=15000.0, fs=2500.0, order=5, btype='band'):
    #     Author: Daniel Macdonald
    b, a = butter_bandpass(lowcut, highcut, fs, order=order,btype=btype)
    y = filtfilt(b, a, data)
    return y

def filter_time_data(df,fs,lowcut=25.0,highcut=15000.0,order=6,btype='highpass'):
    #     Author: Daniel Macdonald
    df_filtered = df.copy()
    for row in range(df.shape[0]):
        df_filtered.iloc[row] = butter_bandpass_filter(df.iloc[row],lowcut=lowcut,highcut=highcut,fs=fs,order=order,btype=btype)
    return df_filtered

def compute_average_spectrogram(df, fs, nWindow,overlapFrac,window,start_t,end_t,thresh, scaling="spectrum", filter_data=False,thresh_method="new"):
    #     Author: Daniel Macdonald
    if filter_data == True:
        df = filter_time_data(df,fs)

    Pxx_mean, freqs, bins = get_spectrogram(df,fs,nWindow,overlapFrac,window,start_t,end_t, scaling) # mode of the spectrogram
    if thresh_method == "old":
        Pxx_scaled, max_val, min_val, lower_thresh = spectrogram_scaling(Pxx_mean,thresh)
    elif thresh_method == "log_only":
        Pxx_scaled = np.log(Pxx_mean)
        max_val = np.max(Pxx_scaled)
        min_val = np.min(Pxx_scaled)
        lower_thresh = "None"
    else:
        Pxx_scaled = Pxx_mean
        max_val = np.max(Pxx_scaled)
        min_val = np.min(Pxx_scaled)
        lower_thresh = "None"

    #print('Pxx_scaled max', Pxx_scaled.max())
    return bins, freqs, Pxx_scaled, max_val, min_val, lower_thresh

def plot_spectrogram(fig1,ax1,bins,freqs,Pxx,ylim_,title=None,convert_a=0.0,convert_b=0.0,x_label=None,color_range=None):
    
    if color_range == None:
        im = ax1.pcolormesh(bins, freqs, Pxx, shading = 'gouraud')#, vmin = -30, vmax = -4) # look up in matplotlib to add colorbar
    else:
        im = ax1.pcolormesh(bins, freqs, Pxx, shading = 'gouraud', vmin = color_range[0], vmax = color_range[1]) # look up in matplotlib to add colorbar


    fig1.colorbar(im, ax=ax1)
    #fig1.set_size_inches(10, 7) #fig1.set_size_inches(10, 7)

    if title != None:
        ax1.set_title('{}'.format(title),y=1.08)
    if x_label != None:
        ax1.set_xlabel(x_label)
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_ylim([0,ylim_]) # new
    if convert_a > 0.000001 or convert_b > 0.000001:
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        def time_convert(x):
            return np.round(x*convert_a + convert_b,decimals=2)

        ax2.set_xticks( ax1.get_xticks() )
        ax2.set_xticklabels(time_convert(ax1.get_xticks()))
        ax2.set_xlabel(x_label)


def chromagram_from_spectrogram(Pxx,fs,n_fft,n_chroma=24,norm=True):
    #     Author: Daniel Macdonald
    # Calculate chroma filterbank
    chromafb = chroma_filterbank(
        sr=fs, 
        n_fft=n_fft, 
        tuning=0.0, 
        n_chroma=n_chroma,
        ctroct=5,
        octwidth=2,
        )
    # Calculate chroma 
    chroma = np.dot(chromafb,Pxx)
    # Normalize
    if norm == "max": # normalize chroma so that the maximum value is 1 in each column
        chroma = normalize(chroma, norm=np.inf, axis=0)
    elif norm == "sum": # normalize chroma so that each column sums to 1
        chroma = (chroma / np.sum(chroma,axis=0)) # Chroma must sum to one for entropy fuction to work
    else:
        print("Raw chroma selected")

    return chroma

def calc_chroma_entropy(chroma,n_chroma):
    #     Author: Daniel Macdonald
    chroma_entropy = -np.sum(chroma * np.log(chroma), axis=0) / np.log(n_chroma)
    #print(np.sum(chroma,axis=0))
    #chroma_entropy = entropy(normed_power) / np.log(n_chroma)
    #chroma_entropy = entropy(normed_power, axis=0) / np.log(n_chroma)
    return 1 - chroma_entropy

def plot_chromagram(fig1,ax1,bins,chroma,title=None,path=None,convert_a=1.0,convert_b=0.0,x_label=None,shading='gouraud',color_range=None):
    #plt.figure(figsize=(14,7)) #fig size same as before

    bins = bins*convert_a+convert_b
    #fig1, ax1 = plt.subplots()
    chroma_y = np.linspace(0,1,chroma.shape[0])
    if color_range == None:
        im = ax1.pcolormesh(bins, chroma_y, chroma, shading =shading)#, vmin = -30, vmax = -4) # look up in matplotlib to add colorbar
    else:
        im = ax1.pcolormesh(bins, chroma_y, chroma, shading =shading, vmin = color_range[0], vmax = color_range[1]) # look up in matplotlib to add colorbar

    fig1.colorbar(im, ax=ax1)
    #fig1.set_size_inches(8, 4) #fig1.set_size_inches(10, 7)

    ax1.set_ylabel('Chroma')
    if title != None:
        ax1.set_title('{}'.format(title))
    if x_label != None:
        ax1.set_xlabel(x_label)

    if path != None:
        fig1.savefig(path)
        path_csv = path.replace(".png",".csv")
        np.savetxt(path_csv, chroma, delimiter=",")


def get_sampling_constants(df,start_t,end_t):
    #     Author: Daniel Macdonald
    '''
    T = period, in seconds, 
    nsamples = samples per cycle
    fs = sample rate
    '''
    T = end_t - start_t
    nsamples = df.shape[1]
    fs = nsamples/T 
    return T, nsamples, fs 
