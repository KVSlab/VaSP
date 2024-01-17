from numpy.fft import fftfreq, fft, ifft
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl

def filter_SPI(U: np.ndarray, W_low_cut: np.ndarray, tag: str) -> float:
    """
    Calculate the Spectral Power Index (SPI) for a given signal.

    Args:
        U (np.ndarray): Input signal.
        W_low_cut (np.ndarray): Array indicating frequency components to be cut.
        tag (str): Tag specifying whether to include mean in the FFT calculation ("withmean" or "withoutmean").

    Returns:
        float: Spectral Power Index.

    Author:
        Mehdi Najafi
    """
    if tag == "withmean":
        U_fft = fft(U)
    else:
        U_fft = fft(U - np.mean(U))

    # Filter any amplitude corresponding frequency equal to 0Hz
    U_fft[W_low_cut[0]] = 0

    # Filter any amplitude corresponding frequency lower to 25Hz
    U_fft_25Hz = U_fft.copy()
    U_fft_25Hz[W_low_cut[1]] = 0

    # Compute the absolute values
    power_25Hz = np.sum(np.power(np.absolute(U_fft_25Hz), 2))
    power_0Hz = np.sum(np.power(np.absolute(U_fft), 2))

    if power_0Hz < 1e-8:
        return 0

    return power_25Hz / power_0Hz

def calculate_spi(case_name: str, df: pd.DataFrame, output_folder: Union[str, Path], mesh_path: Union[str, Path],
                  start_t: float, end_t: float, low_cut: float, high_cut: float, dvp: str) -> None:
    """
    Calculate SPI (Spectral Power Index) and save results in a Tecplot file.

    Args:
        case_name (str): Name of the case.
        df (pd.DataFrame): Input DataFrame containing relevant data.
        output_folder (Union[str, Path]): Output folder path.
        mesh_path (Union[str, Path]): Path to the mesh file.
        start_t (float): Start time for SPI calculation.
        end_t (float): End time for SPI calculation.
        low_cut (float): Lower frequency cutoff for SPI calculation.
        high_cut (float): Higher frequency cutoff for SPI calculation.
        dvp (str): Type of data to be processed ("v", "d", "p", or "wss").

    Returns:
        None: Saves SPI results in a Tecplot file.
    """
    # Get wall and fluid ids
    fluid_ids, wall_ids, all_ids = get_domain_ids(mesh_path)
    fluid_elements, wall_elements, all_elements = get_domain_topology(mesh_path)

    # For displacement spectrogram, we need to take only the wall IDs, filter the data and scale it.
    if dvp == "wss":
        output_file = Path(output_folder) / f"{case_name}_WSS_ts.h5"
        surface_elements, coord = get_surface_topology_coords(output_file)
        ids = np.arange(len(coord))
    elif dvp == "d":
        ids = wall_ids
        elems = np.squeeze(wall_elements)
        coords = get_coords(mesh_path)
        coord = coords[ids, :]
    else:
        ids = fluid_ids
        elems = np.squeeze(fluid_elements)
        coords = get_coords(mesh_path)
        coord = coords[ids, :]

    df_spec = df.iloc[ids]

    T, num_ts, fs = spec.get_sampling_constants(df_spec, start_t, end_t)
    time_between_files = 1 / fs
    W = fftfreq(num_ts, d=time_between_files)

    # Cut low and high frequencies
    mask = np.logical_or(np.abs(W) < low_cut, np.abs(W) > high_cut)
    W_cut = np.where(np.abs(W) == 0) + np.where(mask)

    number_of_points = len(ids)
    SPI = np.zeros([number_of_points])

    for i in range(len(ids)):
        SPI[i] = filter_SPI(df_spec.iloc[i], W_cut, "withoutmean")

    output_filename = Path(output_folder) / f'{case_name}_spi_{low_cut}_to_{high_cut}_t{start_t}_to_{end_t}_{dvp}.tec'

    for j in range(len(ids)):
        elems[elems == ids[j]] = j

    with open(output_filename, 'w') as outfile:
        var_type = 'TRIANGLE' if dvp == "wss" else 'TETRAHEDRON'
        outfile.write(f'VARIABLES = X,Y,Z,SPI\nZONE N={coord.shape[0]},E={elems.shape[0]},F=FEPOINT,ET={var_type}\n')
        for i in range(coord.shape[0]):
            outfile.write(f'{coord[i, 0]: 16.12f} {coord[i, 1]: 16.12f} {coord[i, 2]: 16.12f} {SPI[i]: 16.12f}\n')
        for i in range(elems.shape[0]):
            c = elems[i]
            if dvp == "wss":
                outfile.write(f'\n{c[0] + 1} {c[1] + 1} {c[2] + 1}')
            else:
                outfile.write(f'\n{c[0] + 1} {c[1] + 1} {c[2] + 1} {c[3] + 1}')


def create_domain_specific_viz(formatted_data_folder, output_folder, meshFile,save_deg,time_between_files,start_t,dvp,overwrite=False):

    # Get input data
    components_data = []
    component_names = ["mag","x","y","z"]
    for i in range(len(component_names)):
        if dvp == "p" and i>0:
            break
        file_str = dvp+"_"+component_names[i]+".npz"
        print(file_str)
        component_file = [file for file in os.listdir(formatted_data_folder) if file_str in file]
        component_data = np.load(formatted_data_folder+"/"+component_file[0])['component']
        components_data.append(component_data)


    # Create name for output file, define output path

    if dvp == "v":
        viz_type = 'velocity'
    elif dvp == "d":
        viz_type = 'displacement'
    elif dvp == "p":
        viz_type = 'pressure'
    else:
        print("Input d, v or p for dvp")

    viz_type = viz_type+"_save_deg_"+str(save_deg)
    output_file_name = viz_type+'.h5'
    output_path = os.path.join(output_folder, output_file_name)

    # Create output directory
    if os.path.exists(output_folder):
        print('Path exists!')
    if not os.path.exists(output_folder):
        print("creating output folder")
        os.makedirs(output_folder)

    #read in the fsi mesh:
    fsi_mesh = h5py.File(meshFile,'r')

    # Count fluid and total nodes
    coordArrayFSI= fsi_mesh['mesh/coordinates'][:,:]
    topoArrayFSI= fsi_mesh['mesh/topology'][:,:]
    nNodesFSI = coordArrayFSI.shape[0]
    nElementsFSI = topoArrayFSI.shape[0]

    # Get fluid only topology
    fluidTopology, wallTopology, allTopology = get_domain_topology(meshFile)
    fluid_ids, wall_ids, all_ids = get_domain_ids(meshFile)
    coordArrayFluid= fsi_mesh['mesh/coordinates'][fluid_ids,:]
    nNodesFluid = len(fluid_ids)
    nElementsFluid = fluidTopology.shape[1]

    coordArraySolid= fsi_mesh['mesh/coordinates'][wall_ids,:]
    nNodesSolid = len(wall_ids)
    nElementsSolid = wallTopology.shape[1]
    # Get number of timesteps
    num_ts = components_data[0].shape[1]

    if os.path.exists(output_path) and overwrite == False:
            print('File path {} exists; not overwriting. set overwrite = True to overwrite this file.'.format(output_path))

    else:
        # Remove old file path
        if os.path.exists(output_path):
            print('File path exists; rewriting')
            os.remove(output_path)

        # Create H5 file
        vectorData = h5py.File(output_path,'a')

        # Create mesh arrays
        # 1. update so that the fluid only nodes are used
        # Easiest way is just inputting the fluid-only mesh
        # harder way is modifying the topology of the mesh.. if an element contains a node that is in the solid, then don't include it?
        # for save_deg = 2, maybe we can use fenics to create refined mesh with the fluid and solid elements noted?
        # hopefully that approach will yield the same node numbering as turtleFSI


        if dvp == "d":
            geoArray = vectorData.create_dataset("Mesh/0/mesh/geometry", (nNodesSolid,3))
            geoArray[...] = coordArraySolid
            topoArray = vectorData.create_dataset("Mesh/0/mesh/topology", (nElementsSolid,4), dtype='i')

            # Fix Wall topology (need to renumber nodes consecutively so that dolfin can read the mesh)
            for node_id in range(nNodesSolid):
                wallTopology = np.where(wallTopology == wall_ids[node_id], node_id, wallTopology)
            topoArray[...] = wallTopology
            #print(wallTopology)

        else:
            geoArray = vectorData.create_dataset("Mesh/0/mesh/geometry", (nNodesFluid,3))
            geoArray[...] = coordArrayFluid
            topoArray = vectorData.create_dataset("Mesh/0/mesh/topology", (nElementsFluid,4), dtype='i')

            # Fix Fluid topology
            for node_id in range(len(fluid_ids)):
                fluidTopology = np.where(fluidTopology == fluid_ids[node_id], node_id, fluidTopology)
            topoArray[...] = fluidTopology

        # 2. loop through elements and load in the df
        for idx in range(num_ts):
            ArrayName = 'VisualisationVector/' + str(idx)
            if dvp == "p":
                v_array = vectorData.create_dataset(ArrayName, (nNodesFluid,1))
                v_array[:,0] = components_data[0][fluid_ids,idx]
                attType = "Scalar"

            elif dvp == "v":
                v_array = vectorData.create_dataset(ArrayName, (nNodesFluid,3))
                v_array[:,0] = components_data[1][fluid_ids,idx]
                v_array[:,1] = components_data[2][fluid_ids,idx]
                v_array[:,2] = components_data[3][fluid_ids,idx]
                attType = "Vector"

            elif dvp == "d":
                v_array = vectorData.create_dataset(ArrayName, (nNodesSolid,3))
                v_array[:,0] = components_data[1][wall_ids,idx]
                v_array[:,1] = components_data[2][wall_ids,idx]
                v_array[:,2] = components_data[3][wall_ids,idx]
                attType = "Vector"

            else:
                print("ERROR, input dvp")

        vectorData.close()

        # 3 create xdmf so that we can visualize
        if dvp == "d":
            create_xdmf_file(num_ts,time_between_files,start_t,nElementsSolid,nNodesSolid,attType,viz_type,output_folder)

        else:
            create_xdmf_file(num_ts,time_between_files,start_t,nElementsFluid,nNodesFluid,attType,viz_type,output_folder)

def reduce_save_deg_viz(formatted_data_folder, output_folder, meshFile,save_deg,time_between_files,start_t,dvp,overwrite=False):

    # Get input data
    components_data = []
    component_names = ["mag","x","y","z"]
    for i in range(len(component_names)):
        if dvp == "p" and i>0:
            break
        file_str = dvp+"_"+component_names[i]+".npz"
        print(file_str)
        component_file = [file for file in os.listdir(formatted_data_folder) if file_str in file]
        component_data = np.load(formatted_data_folder+"/"+component_file[0])['component']
        components_data.append(component_data)


    # Create name for output file, define output path

    if dvp == "v":
        viz_type = 'velocity'
    elif dvp == "d":
        viz_type = 'displacement'
    elif dvp == "p":
        viz_type = 'pressure'
    else:
        print("Input d, v or p for dvp")

    viz_type = viz_type+"_save_deg_"+str(save_deg)
    output_file_name = viz_type+'.h5'
    output_path = os.path.join(output_folder, output_file_name)

    # Create output directory
    if os.path.exists(output_folder):
        print('Path exists!')
    if not os.path.exists(output_folder):
        print("creating output folder")
        os.makedirs(output_folder)

    #read in the fsi mesh:
    fsi_mesh = h5py.File(meshFile,'r')

    # Count fluid and total nodes
    coordArrayFSI= fsi_mesh['mesh/coordinates'][:,:]
    topoArrayFSI= fsi_mesh['mesh/topology'][:,:]
    nNodesFSI = coordArrayFSI.shape[0]
    nElementsFSI = topoArrayFSI.shape[0]

    # Get fluid only topology
    fluid_ids, wall_ids, all_ids = get_domain_ids(meshFile)

    # Get number of timesteps
    num_ts = components_data[0].shape[1]

    if os.path.exists(output_path) and overwrite == False:
            print('File path {} exists; not overwriting. set overwrite = True to overwrite this file.'.format(output_path))

    else:
        # Remove old file path
        if os.path.exists(output_path):
            print('File path exists; rewriting')
            os.remove(output_path)
        # Create H5 file
        vectorData = h5py.File(output_path,'a')

        # Create mesh arrays
        geoArray = vectorData.create_dataset("Mesh/0/mesh/geometry", (nNodesFSI,3))
        geoArray[...] = coordArrayFSI
        topoArray = vectorData.create_dataset("Mesh/0/mesh/topology", (nElementsFSI,4), dtype='i')
        topoArray[...] = topoArrayFSI


        # 2. loop through elements and load in the df
        for idx in range(num_ts):
            ArrayName = 'VisualisationVector/' + str(idx)
            if dvp == "p":
                v_array = vectorData.create_dataset(ArrayName, (nNodesFSI,1))
                v_array[:,0] = components_data[0][all_ids,idx]
                attType = "Scalar"

            else:
                v_array = vectorData.create_dataset(ArrayName, (nNodesFSI,3))
                v_array[:,0] = components_data[1][all_ids,idx]
                v_array[:,1] = components_data[2][all_ids,idx]
                v_array[:,2] = components_data[3][all_ids,idx]
                attType = "Vector"


        vectorData.close()

        # 3 create xdmf so that we can visualize
        create_xdmf_file(num_ts,time_between_files,start_t,nElementsFSI,nNodesFSI,attType,viz_type,output_folder)



def create_fixed_xdmf_file(time_values,nElements,nNodes,attType,viz_type,h5_file_list,output_folder):

    # Create strings
    num_el = str(nElements)
    num_nodes = str(nNodes)
    nDim = '1' if attType == "Scalar" else '3'

    # Write lines of xdmf file
    lines = []
    lines.append('<?xml version="1.0"?>\n')
    lines.append('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
    lines.append('<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
    lines.append('  <Domain>\n')
    lines.append('    <Grid Name="TimeSeries_'+viz_type+'" GridType="Collection" CollectionType="Temporal">\n')
    lines.append('      <Grid Name="mesh" GridType="Uniform">\n')
    lines.append('        <Topology NumberOfElements="'+num_el+'" TopologyType="Tetrahedron" NodesPerElement="4">\n')
    lines.append('          <DataItem Dimensions="'+num_el+' 4" NumberType="UInt" Format="HDF">'+h5_file_list[0]+'.h5:/Mesh/0/mesh/topology</DataItem>\n')
    lines.append('        </Topology>\n')
    lines.append('        <Geometry GeometryType="XYZ">\n')
    lines.append('          <DataItem Dimensions="'+num_nodes+' 3" Format="HDF">'+h5_file_list[0]+'.h5:/Mesh/0/mesh/geometry</DataItem>\n')
    lines.append('        </Geometry>\n')

    h5_array_index = 0
    for idx, time_value in enumerate(time_values):
        # Zero the h5 array index if a timesteps come from the next h5 file
        if h5_file_list[idx] != h5_file_list[idx-1]:
            h5_array_index = 0
        lines.append('        <Time Value="'+str(time_value)+'" />\n')
        lines.append('        <Attribute Name="'+viz_type+'" AttributeType="'+attType+'" Center="Node">\n')
        lines.append('          <DataItem Dimensions="'+num_nodes+' '+nDim+'" Format="HDF">'+h5_file_list[idx]+'.h5:/VisualisationVector/'+str(h5_array_index)+'</DataItem>\n')
        lines.append('        </Attribute>\n')
        lines.append('      </Grid>\n')
        if idx == len(time_values)-1:
            break
        lines.append('      <Grid> \n')
        if attType == "Scalar":
            #lines.append('        <ns0:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_'+viz_type+'&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />\n')
            lines.append('        <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_'+viz_type+'&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />\n')
        else:
            lines.append('        <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_'+viz_type+'&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />\n')
        h5_array_index += 1

    lines.append('    </Grid>\n')
    lines.append('  </Domain>\n')
    lines.append('</Xdmf>\n')

    # writing lines to file
    xdmf_path = output_folder+'/'+viz_type.lower()+'_fixed.xdmf'

    # Remove old file path
    if os.path.exists(xdmf_path):
        print('File path exists; rewriting')
        os.remove(xdmf_path)

    xdmf_file = open(xdmf_path, 'w')
    xdmf_file.writelines(lines)
    xdmf_file.close()


def get_time_between_files(input_path, output_folder,mesh_path, case_name, dvp,stride=1):
    # Create name for case, define output path
    print('Creating matrix for case {}...'.format(case_name))
    output_folder = output_folder

    # Create output directory
    if os.path.exists(output_folder):
        print('Path exists')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get node ids from input mesh. If save_deg = 2, you can supply the original mesh to get the data for the
    # corner nodes, or supply a refined mesh to get the data for all nodes (very computationally intensive)
    if dvp == "d" or dvp == "v" or dvp == "p":
        fluid_ids, wall_ids, all_ids = get_domain_ids(mesh_path)
        ids = all_ids

    # Get name of xdmf file
    if dvp == 'd':
        xdmf_file = input_path + '/displacement.xdmf' # Change
    elif dvp == 'v':
        xdmf_file = input_path + '/velocity.xdmf' # Change
    elif dvp == 'p':
        xdmf_file = input_path + '/pressure.xdmf' # Change
    elif dvp == 'wss':
        xdmf_file = input_path + '/WSS_ts.xdmf' # Change
    elif dvp == 'mps':
        xdmf_file = input_path + '/MaxPrincipalStrain.xdmf' # Change
    elif dvp == 'strain':
        xdmf_file = input_path + '/InfinitesimalStrain.xdmf' # Change
    else:
        print('input d, v, p, mps, strain or wss for dvp')

    # If the simulation has been restarted, the output is stored in multiple files and may not have even temporal spacing
    # This loop determines the file names from the xdmf output file
    file1 = open(xdmf_file, 'r')
    Lines = file1.readlines()
    h5_ts=[]
    time_ts=[]
    index_ts=[]

    # This loop goes through the xdmf output file and gets the time value (time_ts), associated
    # .h5 file (h5_ts) and index of each timestep inthe corresponding h5 file (index_ts)
    for line in Lines:
        if '<Time Value' in line:
            time_pattern = '<Time Value="(.+?)"'
            time_str = re.findall(time_pattern, line)
            time = float(time_str[0])
            time_ts.append(time)

        elif 'VisualisationVector' in line:
            #print(line)
            h5_pattern = '"HDF">(.+?):/'
            h5_str = re.findall(h5_pattern, line)
            h5_ts.append(h5_str[0])

            index_pattern = "VisualisationVector/(.+?)</DataItem>"
            index_str = re.findall(index_pattern, line)
            index = int(index_str[0])
            index_ts.append(index)
    time_between_files = time_ts[2] - time_ts[1] # Calculate the time between files from xdmf file

    return time_between_files

