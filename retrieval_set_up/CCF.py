import time
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.pyplot import *
import pickle
from scipy import constants
import numpy as np
import pdb
from numba import jit
import math
import numpy as np
import scipy as sp
from array import *
from scipy import interpolate
from scipy import signal
from scipy import special
from scipy import interp
from scipy import ndimage
from astropy.io import fits
import datetime

def get_rot_ker(vsini, wStar):
    nx, = wStar.shape
    dRV = np.mean(2.0*(wStar[1:]-wStar[0:-1])/(wStar[1:]+wStar[0:-1]))*2.998E5
    nker = 401
    hnker = (nker-1)//2
    rker = np.zeros(nker)
    for ii in range(nker):
        ik = ii - hnker
        x = ik*dRV / vsini
        if np.abs(x) < 1.0:
            y = np.sqrt(1-x**2)
            rker[ii] = y
    rker /= rker.sum()
  
    return rker


def log_likelihood_PCA(Vsys, Kp, scale, cs_p, wlgrid, data_arr,data_scale,ph):


    Ndet, Nphi, Npix = data_arr.shape
 

    I = np.ones(Npix)
    N = Npix#np.array([Npix])
    
    # Time-resolved total radial velocity
    RV = Vsys + Rvel + Kp*np.sin(2.*np.pi*ph)  # Vsys is an additive term around zero   
    dl_l = RV*1E3 / constants.c

    # Initializing log-likelihoods and CCFs
    logL_Matteo = 0.  
    logL_Zuck = 0.
    CCF = 0.
    # Looping through each phase and computing total log-L by summing logLs for each obvservation/phase
    for j in range(Ndet):
        wCut = wlgrid[j,].copy() # Cropped wavelengths    
        gTemp=np.zeros((Nphi,Npix))  #"shifted" model spectra array at each phase
        for i in range(Nphi):
            wShift = wCut * (1.0 - dl_l[i])
            Depth_p = interpolate.splev(wShift, cs_p, der=0) * scale
            gTemp[i,] = Depth_p

        fData=(1.-gTemp)*data_scale[j,]  #1??+fp/fstar is same as (fstar+fp)/fstar..tell "stretches" by transmittance

        #faster SVD
        u,ss,vh=np.linalg.svd(fData,full_matrices=False)  #decompose
        ss[0:4]=0.
        W=np.diag(ss)
        A=np.dot(u,np.dot(W,vh))
        gTemp=A#-data_arr[j,]  #??? If do 0+gTemp then don't need to subtract. What is 1+gTemp anyway??
        #'''
        #gTemp+=1.
        
        # pdb.set_trace()
        for i in range(Nphi):	
            gVec=gTemp[i,].copy()
            gVec-=(gVec.dot(I))/float(Npix)  #mean subtracting here...
            sg2=(gVec.dot(gVec))/float(Npix)
            fVec=data_arr[j,i,].copy() # already mean-subtracted
            sf2=(fVec.dot(fVec))/Npix
            R=(fVec.dot(gVec))/Npix # cross-covariance
            CC=R/np.sqrt(sf2*sg2) # cross-correlation
            if (ph[i]>-0.008)&(ph[i]<0.017):
                logL_Matteo+=0.
                CCF+=0.
            else:
                if not np.isnan(CC):
                    CCF+=CC
                logL_Matteo+=(-0.5*N * np.log(sf2+sg2-2.0*R))
                logL_Zuck+=(-0.5*N * np.log (1.0 - CC**2.0))	


    return logL_Matteo, logL_Zuck, CCF

#################

wl_data, data_arr=pickle.load(open('./PCA_4_clean_data_phstrip.pic','rb'))
wl_data, data_scale=pickle.load(open('./PCA_4_noise_phstrip.pic','rb'))
ph = pickle.load(open('./phi_phstrip.pic','rb'))[0]      # Time-resolved phases
Rvel =pickle.load(open('./Vbary_phstrip.pic','rb'))[0]#+30.  # Time-resolved Earth-star velocity
# pdb.set_trace()
Ndet, Nphi, Npix = data_arr.shape
Kp=198.401  #nominal/published/expected orbital velocity of planet


#'''
## from making CCF plot on multiple CPUs
import sys
ind=int(sys.argv[1])
# ind=0
nKp=200#100 #number of KP points. Will have to change if change velocity resolution
dRV=1 #velocity resolution (note, if you change it you will have to change array stuff below)

#creating KP/Vsys arrays (does all Kp's along a "Vsys Slice")
Kparr=(np.arange(nKp) - (nKp-1)//2) * dRV +Kp #builds "symetric" array about some nominal specified Kp
Kparr=np.linspace(-300.599,300.401,602) #going from -Kp to +Kp with a bit of extra
start=-100#-50 #starting Vsys value
delta=2 #how many "Vsys slices" to do on one CPU
Vsysarr=np.arange(start+delta*ind,start+delta*(ind+1),dRV)
name='part_'+str(ind)
##
#'''

logLarr=np.zeros((len(Kparr),len(Vsysarr)))
CCFarr=np.zeros((len(Kparr),len(Vsysarr)))
scale=1.


#cropping phases if have RM shit from star
'''
loc1=np.where((ph < -1E-3))[0]
loc2=np.where((ph > 1E-3))[0]
loc=np.concatenate([loc1,loc2])
ph=ph[loc]
Rvel=Rvel[loc]
data_arr=data_arr[:,loc,]
data_scale=data_scale[:,loc,]
'''
######


#loading model from make_trans_spec_1DRC.py here
wl_model, Depth=np.loadtxt('./MODELS/gridcomp/H2O/W76_H2O_-3.txt').T

##rotational kernel
vsini=3.3
#ker_rot=get_rot_ker(vsini, wl_model)
#Fp_conv_rot = np.convolve(Depth,ker_rot,mode='same')


xker = np.arange(41)-20
sigma = 5.5/(2.* np.sqrt(2.0*np.log(2.0)))  #nominal

yker = np.exp(-0.5 * (xker / sigma)**2.0)
yker /= yker.sum()
Fp_conv = np.convolve(Depth,yker,mode='same')
cs_p = interpolate.splrep(wl_model,Fp_conv,s=0.0) 


for i in range(len(Kparr)):
    for j in range(len(Vsysarr)):
        t1=time.time()
        logL_M, logL_Z, CCF1=log_likelihood_PCA(Vsysarr[j], Kparr[i],scale, cs_p, wl_data, data_arr,data_scale,ph)
        logLarr[i,j]=logL_M
        CCFarr[i,j]=CCF1
        print i, j, CCF1, logL_M
        t2=time.time()
        print(t2-t1)

if ind < 10:name='part_0'+str(ind)
pickle.dump([Vsysarr, Kparr,CCFarr,logLarr],open('./KP_VSYS/gridcomp/H2O/10e-3/PC4/test_PCA'+name+'.pic','wb'))



