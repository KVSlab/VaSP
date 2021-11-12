import matplotlib as mpl
mpl.use('Agg')
import numpy 
import pandas as pd 
#import vtk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import spectrogram
import random
from scipy.interpolate import RectBivariateSpline

from tempfile import mkdtemp
import os

"""
This script contains helper functions for creating spectrograms.
"""

def get_location_from_pointID(probeID,polydata):
	nearestPointLoc = [10000,10000,10000]
	polydata.GetPoint(probeID,nearestPointLoc)
	return nearestPointLoc


#def find_nearest_points_in_radius(probe_pos,pl,num_points,polyData,radius):
#	"""
#	probeID is actual ID of nearest point
#	nearestPointLoc is xyz location of probeID
#	nearestNeighbours is all the points within the radius
#	nearest_points is sample of nearest_Neighbours
#	NOTE: nearest_points is likely unneccesary after implementing masking 
#	function. 
#	"""
#	probeID = pl.FindClosestPoint(probe_pos) 
#	nearestPointLoc = [10000,10000,10000] 
#	polyData.GetPoint(probeID,nearestPointLoc) 
#	nearestNeighbours = vtk.vtkIdList() 
#	pl.FindPointsWithinRadius(radius,probe_pos,nearestNeighbours) 
#	num_Ids = nearestNeighbours.GetNumberOfIds()
#	print('There are',num_Ids,'in radius',radius)
#	idList = []
#	if num_Ids >= 1:
#		if num_points > num_Ids: 
#			num_points = num_Ids 
#		random_index = random.sample(range(num_Ids), num_points-1)
#		idList = [nearestNeighbours.GetId(idx) for idx in random_index]
#	nearest_points = [probeID] + idList
#	return nearest_points


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



#def vtk_read_unstructured_grid(FILE):
#	'''
#	Reads vtu file. Create a more general reader, particularly for h5 mesh
#	going forward.
#	'''
#	reader = vtk.vtkXMLUnstructuredGridReader()
#	reader.SetFileName(FILE)
#	reader.Update()
#	return reader.GetOutput()

#def vtk_mask_points(polyData, max_points=1000):
#	'''
#	Randomly mask points.
#	SetRandomModeType(2) uses stratified spatial sampling.
#	'''
#	ptMask = vtk.vtkMaskPoints()
#	ptMask.SetInputData(polyData)
#	ptMask.RandomModeOn()
#	ptMask.SetRandomModeType(2)
#	ptMask.SetMaximumNumberOfPoints(max_points)
#	ptMask.Update()
#	return ptMask.GetOutput()

#def get_original_ids(masked_polydata, polyData):
#	n = masked_polydata.GetPoints().GetNumberOfPoints()
#	original_ids = np.zeros(n)
#	pll = vtk.vtkPointLocator()
#	pll.SetDataSet(polyData)
#	for pt in range(n):
#		location = masked_polydata.GetPoints().GetPoint(pt)
#		original_ids[pt] = pll.FindClosestPoint(location)
#	return original_ids

def shift_bit_length(x):
	'''
	round up to nearest pwr of 2
	https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
	'''
	return 1<<(x-1).bit_length()

def get_spectrogram(dfNearest,fsamp,nWindow,overlapFrac,window,start_t,end_t, scaling='spectrum', interpolate = False):
	''' 
	Calculates spectrogram
	input dfNearest is a pandas df of shape (num_points, num_timesteps)
	fsamp is sampling frequency
	Use scaling = 'angle' for phase
	'''
	NFFT = shift_bit_length(int(dfNearest.shape[1]/nWindow)) # Could change to /5
	print(dfNearest.shape[0])

	if dfNearest.shape[0] > 1:
		#print("> 1")
		for each in range(dfNearest.shape[0]):
			row = dfNearest.iloc[each]
			#freqs,bins,Pxx = spectrogram(row,\
			#	fs=fsamp)#,nperseg=NFFT,noverlap=int(overlapFrac*NFFT))#,nfft=2*NFFT,window=window)#,scaling=scaling) 
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

def spectrogram_scaling(Pxx_mean,thresh_percent):
	Pxx_scaled = np.log(Pxx_mean)
	max_val = np.max(Pxx_scaled)
	min_val = np.min(Pxx_scaled)
	print(max_val)
	print(min_val)
	thresh = max_val-(max_val-min_val)*thresh_percent


	Pxx_threshold_indices = Pxx_scaled < thresh
	Pxx_scaled[Pxx_threshold_indices] = thresh
	return Pxx_scaled

def spectrogram_scaling_old(Pxx_mean,lower_thresh):
	Pxx_scaled = np.log(Pxx_mean)
	Pxx_threshold_indices = Pxx_scaled < lower_thresh
	Pxx_scaled[Pxx_threshold_indices] = lower_thresh
	return Pxx_scaled

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
	elif btype == 'highpass':
		b, a = butter(order, low, btype='highpass')
	elif btype == 'lowpass':
		b, a = butter(order, high, btype='lowpass')
	return b, a 

def butter_bandpass_filter(data, lowcut=25.0, highcut=15000.0, fs=2500.0, order=5, btype='band'):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order,btype=btype)
	y = filtfilt(b, a, data)
	return y

def filter_time_data(df,fs,lowcut=25.0,highcut=15000.0,order=6,btype='highpass'):
	df_filtered = df.copy()
	for row in range(df.shape[0]):
		df_filtered.iloc[row] = butter_bandpass_filter(df.iloc[row],lowcut=lowcut,highcut=highcut,fs=fs,order=order,btype=btype)
	return df_filtered

def compute_average_spectrogram(df, fs, nWindow,overlapFrac,window,start_t,end_t,thresh, filter_data=False,thresh_method="new"):
	if filter_data == True:
		df = filter_time_data(df,fs)

	Pxx_mean, freqs, bins = get_spectrogram(df,fs,nWindow,overlapFrac,window,start_t,end_t, 'spectrum') # mode of the spectrogram
	if thresh_method == "new":
		Pxx_scaled =spectrogram_scaling(Pxx_mean,thresh)
	else:
		Pxx_scaled =spectrogram_scaling_old(Pxx_mean,thresh)

	print('Pxx_scaled max', Pxx_scaled.max())
	return bins, freqs, Pxx_scaled

def plot_spectrogram(bins,freqs,Pxx,case_name,start_t,end_t,ylim_,path=None,Re=False,customCmap=None):
	#plt.figure(figsize=(14,7)) #fig size same as before

	fig1, ax1 = plt.subplots()
	if customCmap == None:
		print('no custom colormap')
		ax1.pcolormesh(bins, freqs, Pxx, shading = 'gouraud')#, vmin = -30, vmax = -4) # look up in matplotlib to add colorbar
	else:
		print('custom colormap')
		cmap = plt.get_cmap('Greys')
		ax1.pcolormesh(bins, freqs, Pxx, shading = 'gouraud',cmap=cmap)#, vmin = -30, vmax = -4) # look up in matplotlib to add colorbar

	if Re == True: 
		ax1.set_xlabel('Re')
	else:
		ax1.set_xlabel('Time (s)')

	ax1.set_ylabel('Frequency [Hz]')
	ax1.set_xlim([start_t,end_t]) # new

	ax1.set_ylim([0,ylim_]) # new
	ax1.set_title('{}'.format(case_name))

	if path != None:
		fig1.savefig(path)


def get_sampling_constants(df,start_t,end_t):
	'''
	T = period, in seconds, 
	nsamples = samples per cycle
	fs = sample rate
	'''
	T = end_t - start_t
	nsamples = df.shape[1]
	fs = nsamples/T 
	return T, nsamples, fs 

#def sample_polydata(df, polydata_geo, num_points=2000):
#	polydata_masked = vtk_mask_points(polydata_geo, num_points)
#	#print(polydata_masked)
#	''' 
#	Find the indices of the df that we'll use, filter df
#	'''
#	sampled_points_og_ids = get_original_ids(polydata_masked, polydata_geo) 
#	df_sampled = df.filter(sampled_points_og_ids.astype(int), axis = 0)
#	'''
#	Reset index here allows us to easily search the masked polydata
#	'''
#	df_sampled.reset_index(level=0, inplace=True)
#	df_sampled.index.names = ['NewIds']
#	df_sampled.drop('Ids', 1, inplace = True)
#	return df_sampled, polydata_masked


