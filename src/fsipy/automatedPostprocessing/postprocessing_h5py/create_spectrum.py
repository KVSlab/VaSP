import os
import numpy as np
import spectrograms as spec
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import wavfile

"""
This script creates spectrograms, power spectral density and chromagrams from formatted matrices (.npz files)"

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

def create_spectrum(case_name, dvp, df, start_t, end_t, 
                                 nWindow_per_sec, overlapFrac, 
                                 window, lowcut, thresh_val, max_plot, imageFolder, 
                                 flow_rate_file=None, amplitude_file=None,
                                 power_scaled=False):


    # Calculate number of windows (you can adjust this equation to fit your temporal/frequency resolution needs)
    nWindow = np.round(nWindow_per_sec*(end_t-start_t))+3

    # Get sampling constants
    T, _, fs = spec.get_sampling_constants(df,start_t,end_t)


    ## High-pass filter dataframe for spectrogram
    #df_filtered = spec.filter_time_data(df,fs,
    #                                    lowcut=lowcut,
    #                                    highcut=15000.0,
    #                                    order=6,
    #                                    btype='highpass')

    

    #length = end_t - start_t
    #t = np.linspace(0, length, fs * length)  #  Produces a 5 second Audio-File

    #y2 = df_filtered.iloc[3]/np.max(df_filtered.iloc[3])
#
    #fullname2 = dvp+ '_sound_'+str(y2.name)+"_"+case_name
    #path_to_fig2 = os.path.join(imageFolder, fullname2 + '.wav')
    #from scipy.io import wavfile
    #wavfile.write(path_to_fig2, int(fs), y2)
#
    #y2 = df_filtered.iloc[10]/np.max(df_filtered.iloc[3])
    #fullname2 = dvp+ '_sound_'+str(y2.name)+"_"+case_name
    #path_to_fig2 = os.path.join(imageFolder, fullname2 + '.wav')
    #from scipy.io import wavfile
    #wavfile.write(path_to_fig2, int(fs), y2)

    # PSD calculation
    Pxx_array, freq_array = spec.get_psd(df,fs,scaling="spectrum")
    Pxx_log = np.log(Pxx_array)

    # Plot PSD
    plt.plot(freq_array, Pxx_log)
    plt.xlabel('Freq. (Hz)')
    plt.ylabel('input units^2/Hz')
    #plt.ylim([0,ylim_])
    fullname = dvp+ '_psd_no_filter_'+case_name
    path_to_fig = os.path.join(imageFolder, fullname + '.png')
    plt.savefig(path_to_fig)
    path_csv = os.path.join(imageFolder, fullname + '.csv')
    #print(freq_array.shape)
    #print(Pxx_log.shape)

    #data_csv = np.concatenate((freq_array,Pxx_log),axis=0)
    #np.savetxt(path_csv, data_csv,header="Freqs(Hz),spectrum", delimiter=",")
    data_csv = np.stack((freq_array,Pxx_log),axis=1)
    path_csv = os.path.join(imageFolder, fullname + '.csv')
    np.savetxt(path_csv, data_csv,header="Freqs(Hz),spectrum", delimiter=",")

if __name__ == '__main__':
    # Load in case-specific parameters
    case_path, mesh_name, save_deg, stride,  start_t, end_t, lowcut, ylim, r_sphere, x_sphere, y_sphere, z_sphere, dvp, _, _, interface_only, sampling_method, component, _, point_id = spec.read_command_line_spec()

    # Read fixed spectrogram parameters from config file
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Spectrogram.config")
    overlapFrac, window, n_samples, nWindow_per_sec, lowcut, thresh_val, max_plot, amplitude_file_name, flow_rate_file_name = spec.read_spec_config(config_file,dvp)

    # Create or read in spectrogram dataframe
    dvp, df, case_name, case_path, imageFolder, visualization_hi_pass_folder  = spec.read_spectrogram_data(case_path, 
                                                                                                           mesh_name, 
                                                                                                           save_deg, 
                                                                                                           stride, 
                                                                                                           start_t, 
                                                                                                           end_t, 
                                                                                                           n_samples, 
                                                                                                           ylim, 
                                                                                                           r_sphere, 
                                                                                                           x_sphere, 
                                                                                                           y_sphere, 
                                                                                                           z_sphere, 
                                                                                                           dvp, 
                                                                                                           interface_only, 
                                                                                                           component,
                                                                                                           point_id,
                                                                                                           flow_rate_file_name='MCA_10',
                                                                                                           sampling_method=sampling_method)
    

    amplitude_file = os.path.join(visualization_hi_pass_folder,amplitude_file_name)    
    flow_rate_file = os.path.join(case_path, flow_rate_file_name) 
    # Create spectrograms
    create_spectrum(case_name, 
                                 dvp, 
                                 df,
                                 start_t, 
                                 end_t, 
                                 nWindow_per_sec, 
                                 overlapFrac, 
                                 window, 
                                 lowcut,
                                 thresh_val, 
                                 max_plot, 
                                 imageFolder, 
                                 flow_rate_file=None,
                                 amplitude_file=None,
                                 power_scaled=False)
    if sampling_method=="SinglePoint":
        sonify_point(case_name, dvp, df, start_t, end_t, overlapFrac, lowcut, imageFolder)
