import os
import numpy as np
import spectrograms as spec
#import matplotlib.pyplot as plt
import postprocessing_common_h5py
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

def sigmoid_impulse_filter(x_in, midpoint_increase, midpoint_falloff, k=0.25):
    Sig_increase = 1/(1+np.exp(-k*(x_in-midpoint_increase)))
    Sig_decrease = 1-1/(1+np.exp(-k*(x_in-midpoint_falloff)))
    Sig_impulse = np.minimum(Sig_increase, Sig_decrease)
    return Sig_impulse


def create_harmonic_concentration_index(case_name, case_path, dvp, df, start_t, end_t, 
                                 nWindow_per_sec, overlapFrac, 
                                 window, lowcut, thresh_val, max_plot, imageFolder, 
                                 flow_rate_file=None, amplitude_file=None,
                                 power_scaled=False):
    mode = "sigmoid"
    visualization_path = postprocessing_common_h5py.get_visualization_path(case_path)    
    case_path_modal = case_path.replace("Pulsatile_Ramp_Cases_FC_CFD_undeformed","Modal_Excitation_All_Cases")
    case_path_modal = case_path_modal.replace("Pulsatile_Ramp_Cases_FC","Modal_Excitation_All_Cases")
    case_path_modal = case_path_modal.replace("Pulsatile_CFD","Modal")
    case_path_modal = case_path_modal.replace("Pulsatile","Modal")

    visualization_path_modal = postprocessing_common_h5py.get_visualization_path(case_path_modal)    

    imageFolderModal = os.path.join(visualization_path_modal,"../Spectrograms")
    #imageFolderModal = imageFolder.replace("Pulsatile_Ramp_Cases_FC","Modal_Excitation_All_Cases")
    #imageFolderModal = imageFolder.replace("Pulsatile_Ramp_Cases_FC_CFD_undeformed","Modal_Excitation_All_Cases")
    #imageFolderModal = imageFolder.replace("Pulsatile","Modal")
    #imageFolderModal = imageFolder.replace("Pulsatile_CFD","Modal")

    print(imageFolderModal)
    out_csv_modal_freqs = os.path.join(imageFolderModal,"modal_freqencies.csv")
    out_csv_harmonic_concentration = os.path.join(imageFolder,"harmonic_concentration_"+mode+".csv") 
    out_csv_Pxx = os.path.join(imageFolder,"out_csv_Pxx.csv") # 
    modal_freqs = np.loadtxt(out_csv_modal_freqs)
    print(modal_freqs)
    # only works with pressure so far. 
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
    
    
    #mode1_freq = modal_freqs[1]
    #mode1_freq_half = modal_freqs[1]/2
    #mode1_freq_third = modal_freqs[1]/3
    #mode1_freq_quarter = modal_freqs[1]/4
    #mode1_freq_fifth = modal_freqs[1]/5
    total_power = np.sum(Pxx_array)
    print("total power (before subtracting modal freqs)= ", total_power)
    np.savetxt(out_csv_Pxx,np.array([freq_array,Pxx_array]).T,delimiter=",")
    #print(freq_array[np.where((freq_array>200) & (freq_array<300))])
    harmonic_concentration_index = []

    # Loop through modes 1 to 3
    for i in range(1,4):

        if modal_freqs[i] < 170:
            upper_bnd = 18
            lower_bnd = 9      
        elif modal_freqs[i] < 300:
            upper_bnd = 22
            lower_bnd = 12 
        else:
            upper_bnd = 25
            lower_bnd = 15   
        buffer = 5
        upper_bound = modal_freqs[i]+upper_bnd
        lower_bound = modal_freqs[i]-lower_bnd
    
        if mode == "bins":
            total_power -= np.sum(Pxx_array[np.where((freq_array>lower_bound) & (freq_array<upper_bound))])
        else: 
             # for sigmoid filter
            sigmoid_filter = sigmoid_impulse_filter(freq_array,lower_bound-buffer, upper_bound+buffer)
            total_power -= np.sum(Pxx_array*sigmoid_filter)
        print("total power = ", total_power)

    for i in range(1,4):
        print(total_power)
        if modal_freqs[i] < 170:
            upper_bnd = 18
            lower_bnd = 9      
        elif modal_freqs[i] < 300:
            upper_bnd = 22
            lower_bnd = 12 
        else:
            upper_bnd = 25
            lower_bnd = 15   
        buffer = 5
        upper_bound = modal_freqs[i]+upper_bnd
        lower_bound = modal_freqs[i]-lower_bnd
        

        num_subharmonics = 4
        # Plot PSD
        plt.plot(freq_array, Pxx_array)
        plt.xlabel('Freq. (Hz)')
        plt.ylabel('input units^2/Hz')
        plt.xlim([25,800]) # new
        plt.xscale('log')
        subharmonic_power=np.zeros(num_subharmonics)
        for j in range(1,num_subharmonics+1):

            plt.axvline(x = upper_bound/j, color = 'r', linestyle = '-', label = "{:.2f} Hz".format(upper_bound/j))
            plt.axvline(x = lower_bound/j, color = 'g', linestyle = '-', label = "{:.2f} Hz".format(lower_bound/j))
            if mode == "bins":
                print('power array for {} subharmonic (using bins): {}'.format(j,Pxx_array[np.where((freq_array>lower_bound/j) & (freq_array<upper_bound/j))]))
                subharmonic_power[j-1] = np.sum(Pxx_array[np.where((freq_array>lower_bound/j) & (freq_array<upper_bound/j))])
            else:
                sigmoid_filter = sigmoid_impulse_filter(freq_array,(lower_bound-buffer)/j, (upper_bound+buffer)/j,k=0.25*j)
                print('power array for {} subharmonic (using sigmoid filter): {}'.format(j,Pxx_array*sigmoid_filter))
                subharmonic_power[j-1] = np.sum(Pxx_array*sigmoid_filter)

        subharmonic_concentration = subharmonic_power/total_power
        #plt.ylim([0,ylim_])
        fullname = dvp+ '_psd_'+case_name+"_overlay_Mode_{}".format(i)
        path_to_fig = os.path.join(imageFolder, fullname + '.png')
        plt.title(str(subharmonic_concentration))
        plt.savefig(path_to_fig)
        plt.clf()
        harmonic_concentration_index.append(subharmonic_concentration)
    
    np.savetxt(out_csv_harmonic_concentration,np.array(harmonic_concentration_index),delimiter=",",header="freq/mode #, freq/2, freq/3, freq/4")

    
    '''
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
    spec.plot_spectrogram(fig2,ax2_1,bins,freqs,Pxx,ylim,title=title,x_label="Time (s)",color_range=[thresh_val,max_plot])
    # Save data to files (spectrogram, chromagram, SBI)

    fullname = dvp+"_"+case_name + '_'+str(nWindow)+'_windows_'+'_'+"thresh"+str(thresh_val)+"_spectrogram"
    path_to_spec = os.path.join(imageFolder, fullname + '.png')
    fig2.savefig(path_to_spec)
    path_csv = path_to_spec.replace(".png",".csv")
    #freqs_txt = np.array2string(freqs, precision=2, separator=',',)
    data_csv = np.append(freqs[np.newaxis].T,Pxx, axis=1)
    bins_txt =  np.array2string(bins, max_line_width=10000, precision=2, separator=',',).replace("[","").replace("]","")
    np.savetxt(path_csv, data_csv,header=bins_txt, delimiter=",")

    # Save data to files (spectrogram, chromagram, SBI)
    fullname = dvp+"_"+case_name + '_'+str(nWindow)+'_windows_'+'_chromagram'
    path_to_chroma = os.path.join(imageFolder, fullname + '.png')
    path_csv = path_to_chroma.replace(".png",".csv")
    chroma_y = np.linspace(0,1,chroma.shape[0])
    #freqs_txt = np.array2string(freqs, precision=2, separator=',',)
    data_csv = np.append(chroma_y[np.newaxis].T,chroma, axis=1)
    bins_txt =  np.array2string(bins_raw, max_line_width=10000, precision=2, separator=',',).replace("[","").replace("]","")
    np.savetxt(path_csv, data_csv,header=bins_txt, delimiter=",")

    fullname = dvp+"_"+case_name + '_'+str(nWindow)+'_windows_'+'_SBI'
    path_to_SBI = os.path.join(imageFolder, fullname + '.png')
    path_csv = path_to_SBI.replace(".png",".csv")
    #freqs_txt = np.array2string(freqs, precision=2, separator=',',)
    print(bins)
    print(chroma_entropy)
    data_csv = np.array([bins,chroma_entropy]).T
    np.savetxt(path_csv, data_csv,header="t (s), SBI", delimiter=",")
    #bins_raw,chroma
    '''

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
    case_path, mesh_name, save_deg, stride,  start_t, end_t, lowcut, ylim, sampling_region, fluid_sampling_domain_ID, solid_sampling_domain_ID, r_sphere, x_sphere, y_sphere, z_sphere, dvp, _, _, interface_only, sampling_method, component, _, point_id = spec.read_command_line_spec()

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
                                                                                                           sampling_region,
                                                                                                           fluid_sampling_domain_ID,
                                                                                                           solid_sampling_domain_ID,
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
    create_harmonic_concentration_index(case_name, 
                                 case_path,
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
