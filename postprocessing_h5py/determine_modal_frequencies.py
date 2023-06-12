import os
import numpy as np
import spectrograms as spec
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import wavfile
from scipy.signal import find_peaks_cwt
import glob
import postprocessing_common_h5py
"""
This script creates a combined spectrogram from three component-wise spectrograms by averaging each component. This avoids artifacts associated 
with taking the spectrogram of the  magnitude when the direction reverses

Args:
    mesh_name: Name of the non-refined input mesh for the simulation. This function will find the refined mesh based on this name
    case_path (Path): Path to results from simulation
    stride: reduce output frequncy by this factor
    save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only). If we input save_deg = 1 for a simulation 
       that was run in TurtleFSI with save_deg = 2, the output from this script will be save_deg = 1, i.e only the corner nodes will be output
    start_t: Desired start time of the output files 
    end_t:  Desired end time of the output files 
    lowcut: High pass filter cutoff frequency (Hz)
    ylim: y limit of spectrogram graph")
    r_sphere: Sphere in which to include points for spectrogram, this is the sphere radius
    x_sphere: Sphere in which to include points for spectrogram, this is the x coordinate of the center of the sphere (in m)
    y_sphere: Sphere in which to include points for spectrogram, this is the y coordinate of the center of the sphere (in m)
    z_sphere: Sphere in which to include points for spectrogram, this is the z coordinate of the center of the sphere (in m)
    dvp: "d", "v", "p", or "wss", parameter to postprocess
    interface_only: uses nodes at the interface only. Used for wall pressure spectrogram primarily

"""

def determine_modal_frequencies(case_path, dvp, n_samples, thresh_val, max_plot):
    
    idy_time=12 # time index for determining peak freqs
    thresh_peak=-40 # Power threshold in db for determining peaks
    peak_idx_min_gap=4 # minimum spacing between peaks, in terms of frequency index (4 for surgical cases)
    # Get viz path
    visualization_path = postprocessing_common_h5py.get_visualization_path(case_path)    
    imageFolder = os.path.join(visualization_path,"../Spectrograms")

    # Get all csv files (make sure there is only one for each component)
    x_csv_files = glob.glob(imageFolder + "/**/"+dvp+"_combined_**spectrogram.csv", recursive = True)

    # create spec name
    fullname=os.path.basename(x_csv_files[0]).replace("d_combined_","d_overlayLines_").replace(".csv",".png")

    # get bins from x csv file header
    bins_file = open(x_csv_files[0], "r")
    bins_txt = bins_file.readline()
    bins_txt = bins_txt.replace(" ","").replace("#","")
    bins_list = bins_txt.split(",")
    num_bins = int(len(bins_list))
    bins = np.zeros(num_bins)
    for i in range(num_bins):
        bins[i] = float(bins_list[i])
    
    # Read data
    csv_x_data = np.loadtxt(x_csv_files[0], delimiter=",")

    # Frequencies are the first column of the data
    freqs=csv_x_data[:,0]

    # Average the component10
    Pxx = csv_x_data[:,1:]
    peaks_indices = find_peaks_cwt(Pxx[:,idy_time], np.arange(3,6) ,noise_perc=40) # gap_thresh=10
    print(peaks_indices)
    #print(Pxx[peaks_indices,idy_time])
    #print(peaks_indices)
    peak_idx_prev = 0
    peak_idx_filtered = []
    for peak_idx in peaks_indices:
        if Pxx[peak_idx,idy_time] > thresh_peak:
            if (peak_idx - peak_idx_prev >= peak_idx_min_gap):
                peak_idx_filtered.append(peak_idx)
            else:
                if Pxx[peak_idx,idy_time] >  Pxx[peak_idx_prev,idy_time]:
                    del peak_idx_filtered[-1]
                    peak_idx_filtered.append(peak_idx)
                #else:
                #    peak_idx_filtered.append(peak_idx_prev)

        peak_idx_prev = peak_idx

    modal_freqs = freqs[peak_idx_filtered]

    # create separate spectrogram figure
    fig2, ax2_1 = plt.subplots()
    fig2.set_size_inches(7.5, 5) #fig1.set_size_inches(10, 7)
    title = "threshold Pxx = {}".format(thresh_val)
    path_to_fig = os.path.join(imageFolder, fullname)

    spec.plot_spectrogram(fig2,ax2_1,bins,freqs,Pxx,ylim,title=title,path=path_to_fig,x_label="Time (s)",color_range=[thresh_val,max_plot])
    for i in range(0,4):
        plt.axhline(y = modal_freqs[i], color = 'r', linestyle = '-', label = "{:.2f} Hz".format(modal_freqs[i]))

    print(os.path.basename(case_path) + " Mode 0: {:.2f} Hz, Mode 1: {:.2f} Hz, Mode 2: {:.2f} Hz, Mode 3: {:.2f} Hz,".format(modal_freqs[0],modal_freqs[1],modal_freqs[2],modal_freqs[3]))
  
    plt.axvline(x = bins[idy_time], color = 'b', linestyle = '-')
    plt.legend()
    fig2.savefig(path_to_fig)


if __name__ == '__main__':
    # Load in case-specific parameters
    case_path, mesh_name, save_deg, stride,  start_t, end_t, lowcut, ylim, r_sphere, x_sphere, y_sphere, z_sphere, dvp, _, _, interface_only, sampling_method, component, _, point_id = spec.read_command_line_spec()

    # Read fixed spectrogram parameters from config file
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Spectrogram.config")
    overlapFrac, window, n_samples, nWindow_per_sec, lowcut, thresh_val, max_plot, amplitude_file_name, flow_rate_file_name = spec.read_spec_config(config_file,dvp)

    
    # Create spectrograms
    determine_modal_frequencies(case_path, dvp, 
                                 n_samples, 
                                 thresh_val, 
                                 max_plot)