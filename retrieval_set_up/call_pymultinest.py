import pymultinest
import math, os
import pdb
import numpy as np
import pickle
from fm import *
import pickle
import time
from scipy import constants
from numba import jit
from astropy.io import fits
import sys
from joblib import Parallel, delayed
from scipy.stats import norm

'''
#does the retreival using pymultinest
Currently the retrieved parameters are:

logH2O,  logCO, logOH,   logHCN,logISO=log10 of the constant w/ altitude mixing ratios (ISO is C13/C12 relative terrestrial )

Tiso=isothermal "limb" temperature (can be made more complex using the TP proifle parameterizations in fm.py) 
xRp=scaling to the 10 bar radius (to account for the Rp(P) degeneracy) 
logPc=log10 of the cloud top pressure
Kp=planet orbital velocity
Vsys=system velocity
loga=spectral "strecth" factor (from Brogi & Line)-->note, this is kind of degenrate with xRp...not sure if need both
dphi=phase offset parameter to accounting for ephemeris timing errors



'''

'''
interactive -n 12 -p mrlinegpu1 -q mrlinegpu1 --time=12:00:00 --gres=gpu:1



#cluster specific ommands to load modules needed for GPU. Install these own your own cluster?
module purge
module load tensorflow/1.8-agave-gpu
module unload python/2.7.14
module load multinest/3.10
module load anaconda2/4.4.0


'''


#*********************************************
#defining liklihood function. Takes in the raw Fp/Fstar and orbital parameters and computes CCF and logLike
#Vsys=sysemic velocity (about 0 assuming a velocity array "rvel" is read in already)
#Kp=planet orbit semi-amplitude
#scale=model scale factor
#F_model= data wavelength grid pre-convolved model Fp/Fstar array (ndetectors x npixels)
#wlgrid=data wavelength grid (ndetectors x npixels)
#data_arr=telluric subtracted data array--NOT MEAN SUBTRACTED YET (ndetectors x nphase x npixels)

def LOOP(j,Ndet, Nphi, Npix, data_scale,data_arr,wlgrid, dl_l, cs_p, I):
    # Initializing log-likelihoods and CCFs
    logL_Matteo = 0.  
    logL_Zuck = 0.
    CCF = 0.
    ###start job-lib here????
    wCut = wlgrid[j,].copy() # Cropped wavelengths    
    gTemp=np.zeros((Nphi,Npix))  #"shifted" model spectra array at each phase
    for i in range(Nphi):
        wShift = wCut * (1.0 - dl_l[i])
        Depth_p = interpolate.splev(wShift, cs_p, der=0)
        gTemp[i,] = Depth_p

    fData=(1.-gTemp)*data_scale[j,]  #1??+fp/fstar is same as (fstar+fp)/fstar..tell "stretches" by transmittance

    #faster SVD
    u,ss,vh=np.linalg.svd(fData.T,full_matrices=False)  #decompose
    ss[0:4]=0.
    W=np.diag(ss)
    A=np.dot(u,np.dot(W,vh))
    gTemp=A.T #taking transpose of A   #-data_arr[j,]  #??? If do 0+gTemp then don't need to subtract. What is 1+gTemp anyway??
    #gTemp+=1.

    #'''
    #pdb.set_trace()
    for i in range(Nphi):   
        gVec=gTemp[i,].copy()
        gVec-=(gVec.dot(I))/float(Npix)  #mean subtracting here...
        sg2=(gVec.dot(gVec))/float(Npix)
        fVec=data_arr[j,i,].copy() # already mean-subtracted
        sf2=(fVec.dot(fVec))/Npix
        R=(fVec.dot(gVec))/Npix # cross-covariance
        #CC=R/np.sqrt(sf2*sg2) # cross-correlation   
        #CCF+=CC
        logL_Matteo+=(-0.5*Npix * np.log(sf2+sg2-2.0*R))
        #logL_Zuck+=(-0.5*N * np.log (1.0 - CC**2.0)) 

    return logL_Matteo   


#compacting likelhood function...
def log_likelihood_PCA(Vsys, Kp, dphi, cs_p, wlgrid, data_arr,data_scale):


    Ndet, Nphi, Npix = data_arr.shape
 
    I = np.ones(Npix)
    
    # Time-resolved total radial velocity
    RV = Vsys + Rvel + Kp*np.sin(2.*np.pi*(ph+dphi)) 
    dl_l = RV*1E3 / constants.c #doppler shift in wavelength units

    ##
    #job lib parallelizes the for-loop over the spectral order. The "CC analysis/logL" is computed seperately for each order
    #then summed. Each order is like a "data-point", kind of...
    #n_jobs can be whatever you want (however many cpus on the gpu node), but note, there is diminishing returns
    #and also no benfiti if you use more cpus than there are orders....max should be N_orders
    logL_i=Parallel(n_jobs=42)(delayed(LOOP)(j,Ndet, Nphi, Npix, data_scale,data_arr,wlgrid, dl_l, cs_p, I) for j in range(Ndet))
    logL=np.sum(logL_i)
    
    return logL

#################


#making prior cube
def prior(cube, ndim, nparams):
        ##gas free parameters
        cube[0]=-12+11.*cube[0] #logH2O
        cube[1]=-12+11*cube[1] #logCO
        cube[2]=-12+11.*cube[2] #logOH
        cube[3]=-12+11.*cube[3] #logHmFF
        cube[4]=-12+11.*cube[4] #logHmBF
        # cube[4]=-3+6.*cube[4] #log(C13/C12)

        ##other "bulk" atmosphere parameters
        cube[5]=500+2500*cube[5] #isothermal T
        cube[6]=0.5+1.*cube[6] #x-Rp
        cube[7]=-6+6*cube[7] #logPc

        #HRCCS specific parameters
        cube[8]=norm(180.,20.).ppf(cube[8]) #Kp
        cube[9]=-20+40*cube[9] #Vsys
        # cube[11]=-0.01+0.02*cube[11] #dphi




#putting all the likelihood stuff togather
def loglike(cube, ndim, nparams):
    print('****************')
    t0=time.time()
    #unpacking parameters from "cube"
    logH2O,  logCO, logOH, logHmFF, logHmBF=cube[0],cube[1],cube[2],cube[3],cube[4]
    Tiso, xRp, logPc,Kp,Vsys=cube[5],cube[6],cube[7],cube[8],cube[9]#dphi,cube[10]
    dphi=0.
    scale=1. #spectral scale/strecth
  
    ##all values required by forward model go here--even if they are fixed--this goes into xx (see fx_trans in fm.py)
    xx=np.array([logH2O, logCO,  logOH,  logHmFF, logHmBF, Tiso,   xRp, logPc]) #the -15's are for unused gases--can add
 
    #calling f(x)
    wno,Depth=fx_trans(xx)
    Depth=Depth[::-1] #flipping arrays
    wl_model=1E4/wno[::-1] #converting to microns from wavenumber
    t1=time.time()

    ker_rot=get_rot_ker(3.3, wl_model) #rotation kernel ###FINDME: vsin(i) here needs to be changed for other planets
    Depth_conv_rot = np.convolve(Depth,ker_rot,mode='same') #doing convolution

    #making IP and doing IP convolution
    xker = np.arange(41)-20
    sigma = 5.5/(2.* np.sqrt(2.0*np.log(2.0)))  #nominal
    yker = np.exp(-0.5 * (xker / sigma)**2.0)
    yker /= yker.sum()
    Depth_conv = np.convolve(Depth_conv_rot,yker,mode='same')*scale
    cs_p = interpolate.splrep(wl_model,Depth_conv,s=0.0) 
    t2=time.time()

    logL_M=log_likelihood_PCA(Vsys, Kp,dphi, cs_p, wl_data, data_arr,data_scale)


    print(logH2O, logCO,logOH,logHmFF,logHmBF, Kp,Vsys, scale)
     
    loglikelihood=logL_M

    t3=time.time()
    print('TOTAL: ',t3-t0)
    print('logL ',t3-t2)
    print('Conv/Interp: ',t2-t1)
    print('GPU RT: ',t1-t0)
    return loglikelihood


########################
#reading in the data from text file
########################

#data arrays--these don't need to be passed...they are "global"--bad programing but who cares
wl_data, data_arr=pickle.load(open('PCA_4_clean_data_new.pic','rb')) #PCA's matrix
wl_data, data_scale=pickle.load(open('PCA_4_noise_new.pic','rb')) #crap that was removed for model re-injection in logL
ph = pickle.load(open('phi_new.pic','rb'))[0]      # phases array
Rvel =pickle.load(open('Vbary_new.pic','rb'))[0]## barycenter velocity array
Ndet, Nphi, Npix = data_arr.shape

'''
#uncomment if want to make a plot of a single spectrum
###making a single forward model spectrum given the atmospheric parameters for testing... ------------------------
####
#          logH2O, logCO, logCH4 , logH2S,  logOH,  logHCN, logC2H2,  logISO,   Tiso,   xRp, logPc   ]
x=np.array([-5.,     -3., -10.,     -10.,    -4.,    -10.,    -10,      0.,     2000.,    1.,  0. ])
scale=1
wno,Depth=fx_trans(x)
Depth=Depth[::-1]
wl_model=1E4/wno[::-1]
from matplotlib.pyplot import*
plot(wl_model, Depth)
show()
pdf.set_trace()
######
'''

#Setting up pymiultinest input and running (kind of like emcee...)
outfile='PMN_mmw_1.3.pic' 
n_params=11 #number of params (count the number of indecies in cube..it's too dumb to figure it out on its own)
Nlive=500 #number of live points (100-1000 is good....the more the better, but the "slower")
res=False #if the job crashes or you get the boot, you can change this to "true" to restart it from where it left off
pymultinest.run(loglike, prior, n_params, outputfiles_basename='./chains/mmw13/H2O_CO_OH_Hm_',resume=res, verbose=True,n_live_points=Nlive)
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='./chains/mmw13/H2O_CO_OH_Hm_')
s = a.get_stats()
output=a.get_equal_weighted_posterior()
pickle.dump(output,open(outfile,"wb"))





