#Perform wavelength calibration

import numpy as np
import pickle
from scipy import interpolate
from scipy.optimize import curve_fit
import pdb

def stretched(wl, shift, stretch):
	'''
	Given a raw wavelength array, calculates the shifted and stretched
	version given shift and stretch nuisance parameters. Then calculates
	the new flux array by spline interpolating the data (global variable) onto the
	new stretched wavelength array
	'''
	wl1=shift+wl*stretch
	data_int=interpolate.splev(wl1,cs_data,der=0)
	
	return data_int


def correct(wl_arr, data_arr, skyorder, output=False):
	'''
	Perform wavelength calibration by shifting/stretching each frame to match the frame closest in time to the wavelength standard.
	'''
	data_corrected = np.zeros(data_arr.shape)
	num_orders, num_files, num_pixels = data_arr.shape
	for order in range(num_orders):
		if skyorder==1:
			control = data_arr[order,0,:] #spectrum assumed to have "true" wavelengths, file1
			rcontrol = control/control.max() #renormalize
		elif skyorder==2:
			control = data_arr[order,-1,:] #spectrum assumed to have "true" wavelengths, file1
			rcontrol = control/control.max() #renormalize
		wl_raw = wl_arr[order,]
		for frame in range(num_files):
			data_to_correct = data_arr[order,frame,]
			data_to_correct = data_to_correct/data_to_correct.max() #normalize
			global cs_data
			cs_data = interpolate.splrep(wl_raw, data_to_correct,s=0.0)
			popt, pconv = curve_fit(stretched, wl_raw, rcontrol, p0=np.array([0,1.]))
			data_stretched = stretched(wl_raw, *popt)
			data_corrected[order, frame,] = data_stretched #normalized

	if output==True:
		pickle.dump([wl_data,wavecorrect],open('wavelengthcalibrated.pic','wb'),protocol=2)

	return data_corrected

	

