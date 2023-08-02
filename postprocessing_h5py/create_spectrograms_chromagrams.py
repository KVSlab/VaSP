import os
import numpy as np
import spectrograms as spec
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

def create_spectrogram_composite(case_name, dvp, df, start_t, end_t, 
                                 nWindow_per_sec, overlapFrac, 
                                 window, lowcut, thresh_val, max_plot, imageFolder, 
                                 flow_rate_file=None, amplitude_file=None,
                                 power_scaled=False):


    # Calculate number of windows (you can adjust this equation to fit your temporal/frequency resolution needs)
    nWindow = np.round(nWindow_per_sec*(end_t-start_t))+3

    # Get sampling constants
    T, _, fs = spec.get_sampling_constants(df,start_t,end_t)


    # High-pass filter dataframe for spectrogram
    df_filtered = spec.filter_time_data(df,fs,
                                        lowcut=lowcut,
                                        highcut=15000.0,
                                        order=6,
                                        btype='highpass')

    

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
    Pxx_array, freq_array = spec.get_psd(df_filtered,fs)
    # Plot PSD
    plt.plot(freq_array, Pxx_array)
    plt.xlabel('Freq. (Hz)')
    plt.ylabel('input units^2/Hz')
    #plt.ylim([0,ylim_])
    fullname = dvp+ '_psd_'+case_name
    path_to_fig = os.path.join(imageFolder, fullname + '.png')
    plt.savefig(path_to_fig)
    
    
    # Create composite figure
    if amplitude_file and flow_rate_file:
        fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True,  gridspec_kw={'height_ratios': [1,3,1,1,1]})
    elif flow_rate_file:
        fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True,  gridspec_kw={'height_ratios': [1,3,1,1]})
    elif amplitude_file:
        fig1, (ax2, ax3, ax4, ax5) = plt.subplots(4, sharex=True,  gridspec_kw={'height_ratios': [3,1,1,1]})
    else:
        fig1, (ax2, ax3, ax4) = plt.subplots(3, sharex=True,  gridspec_kw={'height_ratios': [3,1,1]})
    # Spectrogram--------------------------------------------------------------
    
    fig1.set_size_inches(7.5, 9) #fig1.set_size_inches(10, 7)
    # Specs with Reyynolds number
    bins, freqs, Pxx, max_val, min_val, lower_thresh = spec.compute_average_spectrogram(df_filtered, 
                                                                                        fs, 
                                                                                        nWindow,
                                                                                        overlapFrac,
                                                                                        window,
                                                                                        start_t,
                                                                                        end_t,
                                                                                        thresh_val,
                                                                                        scaling="spectrum",
                                                                                        filter_data=False,
                                                                                        thresh_method="old")
    bins = bins+start_t # Need to shift bins so that spectrogram timing is correct
    spec.plot_spectrogram(fig1,ax2,bins,freqs,Pxx,ylim,color_range=[thresh_val,max_plot])
    
    
    # Chromagram ------------------------------------------------------------
    n_fft = spec.shift_bit_length(int(df.shape[1]/nWindow))*2 
    n_chroma=24
    # recalculate spectrogram without filtering the data
    bins_raw, freqs_raw, Pxx_raw, max_val_raw, min_val_raw, lower_thresh_raw = spec.compute_average_spectrogram(df, 
                                                                                                                fs, 
                                                                                                                nWindow,
                                                                                                                overlapFrac,
                                                                                                                window,
                                                                                                                start_t,
                                                                                                                end_t,
                                                                                                                thresh_val,
                                                                                                                scaling="spectrum",
                                                                                                                filter_data=False,
                                                                                                                thresh_method="old")
    bins_raw = bins_raw+start_t # Need to shift bins so that spectrogram timing is correct
    # Reverse the log of the data
    Pxx_raw=np.exp(Pxx_raw)
    
    # Calculate chromagram
    norm="sum" # normalize so that all chroma in column sum to 1 (other option is "max", which sets the max value in each column to 1)
    chroma = spec.chromagram_from_spectrogram(Pxx_raw,fs,n_fft,n_chroma=n_chroma,norm=norm)
    if power_scaled == True:
        chroma_power = (chroma)*(Pxx.max(axis=0)-thresh_val)
        # Plot chromagram
        spec.plot_chromagram(fig1,ax3,bins_raw,chroma_power)
    else:
        # Plot chromagram
        spec.plot_chromagram(fig1,ax3,bins_raw,chroma)
    # Hack to make all the x axes of the subplots align
    divider2 = make_axes_locatable(ax4)
    cax2 = divider2.append_axes("right", size="5%", pad=0.9)
    cax2.remove()
    
    # Calculate SBI
    chroma_entropy = spec.calc_chroma_entropy(chroma,n_chroma)
    # Plot SBI
    if power_scaled == True:
        chroma_entropy_power = (chroma_entropy)*(Pxx.max(axis=0)-thresh_val)
        # Plot chromagram
        ax4.plot(bins,chroma_entropy_power)
    else:
        # Plot chromagram
        ax4.plot(bins,chroma_entropy)
    ax4.set_ylabel('SBI')
    
    # Plot Flow Rate or inlet velocity from input file
    if flow_rate_file:
        flow_rates = np.loadtxt(flow_rate_file)
        flow_rates = flow_rates[np.where((flow_rates[:,0]>start_t) & (flow_rates[:,0]<end_t))]
        ax1.plot(flow_rates[:,0],flow_rates[:,1])
        ax1.set_ylabel('Flow Rate (normalized)')
        # Hack to make all the x axes of the subplots align
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.9)
        cax.remove()
    
    
    if amplitude_file:
        # Hack to make all the x axes of the subplots align
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes("right", size="5%", pad=0.9)
        cax.remove()
        output_amplitudes = np.genfromtxt(amplitude_file, delimiter=',')
        output_amplitudes=output_amplitudes[output_amplitudes[:,0]>=start_t]
        output_amplitudes=output_amplitudes[output_amplitudes[:,0]<=end_t]
        ax5.plot(output_amplitudes[:,0],output_amplitudes[:,10],label=case_name)
        ax5.set_ylabel("Amplitude")
        ax5.set_xlabel('Time (s)')
    else: 
        ax4.set_xlabel('Time (s)')
      
    # Name composite Figure and save
    composite_figure_name = dvp+"_"+case_name + '_'+str(nWindow)+'_windows_'+'_'+"thresh"+str(thresh_val)+"_composite_figure"
    if power_scaled == True:
        composite_figure_name = composite_figure_name + "_power_scaled"
    path_to_fig = os.path.join(imageFolder, composite_figure_name + '.png')
    fig1.savefig(path_to_fig)
    
    # create separate spectrogram figure
    fig2, ax2_1 = plt.subplots()
    fig2.set_size_inches(7.5, 5) #fig1.set_size_inches(10, 7)
    title = "Pxx max = {:.2e}, Pxx min = {:.2e}, threshold Pxx = {}".format(max_val, min_val, lower_thresh)
    fullname = dvp+"_"+case_name + '_'+str(nWindow)+'_windows_'+'_'+"thresh"+str(thresh_val)+"_spectrogram"
    path_to_fig = os.path.join(imageFolder, fullname + '.png')
    spec.plot_spectrogram(fig2,ax2_1,bins,freqs,Pxx,ylim,title=title,path=path_to_fig,x_label="Time (s)",color_range=[thresh_val,max_plot])
    fig2.savefig(path_to_fig)


def sonify_point(case_name, dvp, df, start_t, end_t, overlapFrac, lowcut, imageFolder):


    # Get sampling constants
    T, _, fs = spec.get_sampling_constants(df,start_t,end_t)

    # High-pass filter dataframe for spectrogram
    df_filtered = spec.filter_time_data(df,fs,
                                        lowcut=lowcut,
                                        highcut=15000.0,
                                        order=6,
                                        btype='highpass')


    length = end_t - start_t
    t = np.linspace(0, length, int(fs * length))  #  Produces a 5 second Audio-File
    y2 = df_filtered.iloc[0]/np.max(df_filtered.iloc[0])
    fullname2 = dvp+ '_sound_'+str(y2.name)+"_"+case_name
    path_to_fig2 = os.path.join(imageFolder, fullname2 + '.wav')
    wavfile.write(path_to_fig2, int(fs), y2)


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
    create_spectrogram_composite(case_name, 
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
