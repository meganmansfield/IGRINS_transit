#FINDME: come up with better names for all the variables
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import rc
rc('axes',linewidth=2)

### DEFINE FUNCTIONS ###
def PCA(cube, PCs = 6):
	'''
	Inputs: cube (arg) - a numpy array of dimension 3, data cube output from make_cube function
			PCs (kwarg) - integer, number of principal components to be removed from data

	Outputs: pca_data - a numpy array of dimension 3, input data cube with N principal components removed
		     pca_scale - a numpy array of dimension 3, cube containing the first N principal components only

	Perform PCA on a data cube. Returns two numpy arrays
	of the same shape as the input cube. The first will
	be the input cube with the first N PCs removed. The
	second will be the input cube with the first N PCs 
	remaining and all others removed.
	'''
	num_orders, num_files, num_pixels = cube.shape
	pca_data =  np.zeros(cube.shape)
	pca_scale = np.zeros(cube.shape)
	for order in range(num_orders):
		u,s,vh = np.linalg.svd(cube[order,],full_matrices=False) #decompose
		s1 = s.copy()
		s1[PCs:] = 0. 
		W1 = np.diag(s1)
		A1 = np.dot(u, np.dot(W1, vh)) #recompose
		pca_scale[order,] = A1 

		s[0:PCs] = 0. #remove first N PCs
		W = np.diag(s)
		A = np.dot(u, np.dot(W,vh))
		#sigma clip
		sig = np.std(A)
		med = np.median(A)
		loc = (A > 3. * sig+med)
		A[loc] = 0. 
		loc = (A < -3. * sig+med)
		A[loc] = 0.
		pca_data[order,] = A

	return pca_data, pca_scale

def do_pca(wl_data,normalized,nPCAs,test_pca=True,plot=False,output=False,test_order=5):
	# wl_data,normalized=pickle.load(open(wavecal,'rb'))

	num_orders,num_files,num_pixels=normalized.shape
	sub_pca_matrix=np.zeros((num_files,num_pixels,nPCAs))
	aspect=0.01/num_files
	if test_pca==True:
		for numpcs in range(1,nPCAs+1):
			pca_clean_data,pca_noplanet=PCA(normalized, numpcs) #working with normalized data
			sub_pca_matrix[:,:,numpcs-1]=pca_clean_data[test_order,:,:]
			plt.figure()
			plt.imshow(sub_pca_matrix[:,:,numpcs-1],extent=(np.min(wl_data[test_order,:]),np.max(wl_data[test_order,:]),1,num_files),aspect=aspect)
			plt.ylabel('Exposure',fontsize=20)
			plt.xlabel('Wavelength',fontsize=20)
			plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
			plt.colorbar()
			plt.title(str(numpcs))
			plt.show()

	else:
		pca_clean_data,pca_noplanet=PCA(normalized,nPCAs) #working with normalized data

	if plot==True:
		one_component,one_noplanet=PCA(normalized,1)
		vmin=np.min((np.min(normalized),np.min(pca_clean_data)))
		vmax=np.max((np.max(normalized),np.max(pca_clean_data)))
		fig,(ax1,ax2,ax3)=plt.subplots(3,1,sharex=True,figsize=(7.5,8))
		ax1.imshow(normalized[test_order,:,:],extent=(np.min(wl_data[test_order,:]),np.max(wl_data[test_order,:]),1,num_files),aspect=aspect)
		ax1.set_title('Before PCA',fontsize=15)
		ax1.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
		ax2.imshow(one_component[test_order,:,:],extent=(np.min(wl_data[test_order,:]),np.max(wl_data[test_order,:]),1,num_files),aspect=aspect)
		ax2.set_title('1 Component Removed',fontsize=15)
		ax2.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
		ax3.imshow(pca_clean_data[test_order,:,:],extent=(np.min(wl_data[test_order,:]),np.max(wl_data[test_order,:]),1,num_files),aspect=aspect)
		ax3.set_xlabel('Wavelength [$\mu$m]',fontsize=20)
		ax3.set_title(str(nPCAs)+' Components Removed',fontsize=15)
		ax3.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
		fig.add_subplot(111, frame_on=False)
		plt.tick_params(labelcolor='none',bottom=False,left=False)
		plt.ylabel('Exposure',fontsize=20)
		plt.show()

	if output==True:
		pickle.dump([wl_data,pca_clean_data],open('PCA_'+str(nPCAs)+'_clean_data.pic','wb'),protocol=2)
		pickle.dump([wl_data,pca_noplanet],open('PCA_'+str(nPCAs)+'_noise.pic','wb'),protocol=2)

	return wl_data,pca_clean_data,pca_noplanet





