#FINDME: come up with better names for all the variables
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import interpolate

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

def do_pca(wl_data,normalized,nPCAs,test_pca=True,output=False,test_order=5):
	# wl_data,normalized=pickle.load(open(wavecal,'rb'))

	num_orders,num_files,num_pixels=normalized.shape
	sub_order_1=5
	sub_order_2=25	#for plotting purposes
	sub_pca_matrix=np.zeros((num_files,num_pixels,nPCAs))
	if test_pca==True:
		for numpcs in range(1,nPCAs+1):
			pca_clean_data,pca_noplanet=PCA(normalized, numpcs) #working with normalized data
			sub_pca_matrix[:,:,numpcs-1]=pca_clean_data[test_order,:,:]
			plt.figure()
			plt.imshow(sub_pca_matrix[:,:,numpcs-1],aspect=10)
			plt.colorbar()
			plt.title(str(numpcs))
			plt.show()

	else:
		pca_clean_data,pca_noplanet=PCA(normalized,nPCAs) #working with normalized data

	if output==True:
		pickle.dump([wl_data,pca_clean_data],open('PCA_'+str(nPCAs)+'_clean_data.pic','wb'),protocol=2)
		pickle.dump([wl_data,pca_noplanet],open('PCA_'+str(nPCAs)+'_noise.pic','wb'),protocol=2)

	return wl_data,pca_clean_data,pca_noplanet





