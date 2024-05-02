import numpy as np 
import matplotlib.pyplot as plt 
import pickle

def phasecrop(data,ph,Rvel,Per,T14):
	edgephase=T14/(Per*24.)/2.
	loc=np.where((ph>-edgephase)&(ph<edgephase))[0]
	ph=ph[loc]
	Rvel=Rvel[loc]
	data=data[:,loc,:]
	pickle.dump([ph],open('./phi_phstrip.pic','wb'),protocol=2)
	pickle.dump([Rvel],open('./Vbary_phstrip.pic','wb'),protocol=2)
	return data,ph,Rvel


