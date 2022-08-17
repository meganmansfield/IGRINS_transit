import math as mth
import math
import numpy as np
import scipy as sp
from array import *
from scipy import interpolate
from scipy import signal
from scipy import special
from scipy import interp
from scipy import ndimage
import pdb
import datetime
import pickle
from scipy import constants
from numba import jit
from numba import jit, vectorize, guvectorize, float64, complex64, int32,float64,cuda
import time
import h5py
import matplotlib as mpl
# mpl.use('TkAgg')
from matplotlib.pyplot import *


#GENERAL COLLECTION OF NECESSARY RADIATIVE TRANSFER FUNCTIONS. SOMEWHAT COMMENTED...


###BEGIN TRANSMISSION SPECTRA GPU ROUTINES##################################
@guvectorize([(float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:,:])], '(m),(n),(o),(o),(n,m,q)->(o,q)',target='cuda',nopython=True)
def xsec_interp_gpu_trans(logPgrid, logTgrid, Patm, Tatm, xsec, xsec_int):

        Ng=xsec.shape[-1]
        Natm=Patm.shape[0]  #number of midpoint atmosphere points

        for i in range(Natm-1):  #looping through atmospheric layers
            Pavg=0.5*(Patm[i]+Patm[i+1])
            Tavg=0.5*(Tatm[i]+Tatm[i+1])

            y=mth.log10(Patm[i])
            x=mth.log10(Tatm[i])
            if y > logPgrid[-1]: y=logPgrid[-1]
            if y < logPgrid[0]: y=logPgrid[0]
            if x > logTgrid[-1]: x=logTgrid[-1]
            if x < logTgrid[0]: x=logTgrid[0]


            #foo=logPgrid[logPgrid < -2.0]
            p_ind_hi=0
            while logPgrid[p_ind_hi] <= y: p_ind_hi=p_ind_hi+1
            p_ind_low=p_ind_hi-1
            T_ind_hi=0
            while logTgrid[T_ind_hi] <= x: T_ind_hi=T_ind_hi+1
            T_ind_low=T_ind_hi-1

            y2=logPgrid[p_ind_hi]
            y1=logPgrid[p_ind_low]
            x2=logTgrid[T_ind_hi]
            x1=logTgrid[T_ind_low]
            w11=(x2-x)/(x2-x1)
            w21=(x-x1)/(x2-x1)
            yy1=(y2-y)/(y2-y1)
            yy2=(y-y1)/(y2-y1)
            ww11=yy1*w11
            ww21=yy1*w21
            ww12=yy2*w11
            ww22=yy2*w21

            for j in range(Ng): #looping through gases
                Q11=xsec[T_ind_low,p_ind_low,j]
                Q12=xsec[T_ind_low,p_ind_hi,j]
                Q22=xsec[T_ind_hi,p_ind_hi,j]
                Q21=xsec[T_ind_hi,p_ind_low,j]
                xsec_int[i,j]=10**(ww11*Q11+ww21*Q21+ww12*Q12+ww22*Q22)

#########


#COMPUTES THE LIMB TRANSMITTANCE t AS A FUNCTION OF ALTITUDE/LEVEL, Z
#newest version, pulls abundance loop out seperately to make faster
@guvectorize([(float64[:,:], float64[:,:],float64[:,:], float64[:])], '(o,q), (m,l), (o,o) -> (o) ',target='cuda',nopython=True)
def CalcTran(xsecs, Fractions,uarr, trans):
    ngas=len(Fractions)
    nlevels=len(uarr)
    nwno=xsecs.shape[0]
    #ncont=xsecContinuum.shape[-1]
    kb=1.38E-23
    nlev=113 ###FINDME###---this is hard coded (its same value as nelvels but python gpu throws a fit if it's dynamically typed...)
    wt_xsec=cuda.local.array(shape=(nlev,),dtype=float64) #stupid memory BS ot define local array
    #trans=np.zeros(nlevels)
    for i in range(nlevels):
        wt_xsec[i]=0.
        for k in range(ngas):
            wt_xsec[i]+=Fractions[k,i]*xsecs[i,k]

    for i in range(nlevels-2):
        tautmp=0.E0
        for j in range(i):
            curlevel=i-j-1
            tautmp+=2.*wt_xsec[curlevel]*uarr[i,j]

        trans[i]=mth.exp(-tautmp)

######


##this is just a stupid function to take the dot-product so as to not have to copy the big ass 
#t[wno, Z] array from GPU to CPU.  
@guvectorize([(float64[:], float64[:],float64[:],float64, int32, float64[:])],'(o), (o), (o), (), () -> ()',target='cuda',nopython=True)
def CalcAnnulus(trans, Z, dZ, r0, locPc, depth):
    nlevels=len(Z)
    FF=0.
    for i in range(nlevels):
        if i >= locPc: trans[i]=0.
        FF+=(1-trans[i])*(r0+Z[i])*dZ[i]

    depth[0]=FF

#####


#compute path langths and path mass for each segment of each tangent height
@jit(nopython=True)
def compute_paths(Z, Pavg, Tavg,r0):
    uarr=np.zeros((len(Z),len(Z)))
    nlevels=len(Z)
    kb=1.38E-23
    for i in range(nlevels-2):
        for j in range(i):
            curlevel=i-j-1
            r1=r0+Z[i]
            r2=r0+Z[i-j]
            r3=r0+Z[curlevel]
            uarr[i,j]=(((r3**2-r1**2)**2)**0.25-((r2**2-r1**2)**2)**0.25)*(Pavg[curlevel]*1.0E5)/(kb*Tavg[curlevel])
    return uarr


######END TRANSIT GPU ROUTINES##################################


#"wrapper script" for GPU transmission spectrum routines (what is called in make_trans_spec_1DRC and call_pymultinest)
def tran(T, P, mmw,Ps,Pc,H2O,CO,OH, HmFF,HmBF, H2,He, amp, power,M,Rstar,Rp):
    
   
    #putting abundance arrays into 2D array for shipping off to GPU--same order as in xsects() loading routine
    Fractions=np.array([H2*H2, He*H2, H2O, CO,OH,HmFF,HmBF ])

    #loading cross-sections (kind of a dumb way of doing this)
    logPgrid = restore.xsects[0] #pressure grid that xsecs are pre-computed on
    logTgrid = restore.xsects[1] #temperature grid that xsecs are pre-computed on
    wno = restore.xsects[2] #wavenumber array for xsecs
    d_xsecarr = restore.xsects[3] #this is a memory pointer that puts the xses on GPU

 
    #Computing hydrostatic grid 
    n = len(P)
    Z=np.zeros(n)  #level altitudes
    dZ=np.zeros(n)  #layer thickness array
    r0=Rp*69911.*1.E3  #converting planet radius to meters
    mmw=mmw*1.660539E-27  #converting mmw to Kg
    kb=1.38E-23
    G=6.67384E-11
    M=1.898E27*M #jupiter masses to kg
    
    #Compute avg Temperature at each grid mid-point
    Tavg = np.array([0.0]*(n-1))
    Pavg = np.array([0.0]*(n-1))
    for z in range(n-1):
        Pavg[z] = np.sqrt(P[z]*P[z+1])
        Tavg[z] = interp(np.log10(Pavg[z]),sp.log10(P),T)

    #create hydrostatic altitutde grid from P and T
    Phigh=P.compress((P>Ps).flat)  #deeper than reference pressure
    Plow=P.compress((P<=Ps).flat)   #shallower than reference pressure
    for i in range(Phigh.shape[0]):  #looping over levels above ref pressure
        i=i+Plow.shape[0]-1
        g=G*M/(r0+Z[i])**2 #value of gravity at each index/pressure layer
        H=kb*Tavg[i]/(mmw[i]*g)  #scale height
        dZ[i]=H*np.log(P[i+1]/P[i]) #layer thickness in altitude units (m), dZ is negative below reference pressure
        Z[i+1]=Z[i]-dZ[i]   #level altitude
    for i in range(Plow.shape[0]-1):  #looping over levels below ref pressure
        i=Plow.shape[0]-i-1
        g=G*M/(r0+Z[i])**2
        H=kb*Tavg[i]/(mmw[i]*g)
        dZ[i]=H*np.log(P[i+1]/P[i])
        Z[i-1]=Z[i]+dZ[i]

    #Xsec interpolation on P-T grid
    xsec_inter=xsec_interp_gpu_trans(logPgrid, logTgrid, P, T, d_xsecarr)  

    #compute transmission spectrum
    path_arr=compute_paths(Z, Pavg, Tavg, r0) #path segments "ds" along each tangent beam
    t=CalcTran(xsec_inter, Fractions, path_arr) #limb transmittance (t[wno, Z])
    #t5=time.time()
    #print 'TRANSMITTANCE', t5-t4
    locPc=np.where(P >= Pc)[0][0] #finding cloud top pressure array index
    annulus=CalcAnnulus(t, Z, dZ, r0, locPc) #computing the annlus integral
    #t6=time.time()
    #print 'DOT PRODUCT', t6-t5
    annulus=annulus.copy_to_host() #copy annulus integral from GPU to CPU memory
    #t7=time.time()
    #print 'COPY', t7-t6
    F=((r0+np.min(Z[:-1]))/(Rstar*6.955E8))**2+2./(Rstar*6.955E8)**2.*annulus #the usual transmission equation
    #t8=time.time()
    #print 'FINAL', t8-t7
 

    return wno, F, Z




#*******************************************************************
# FILE: xsects.py
#
# DESCRIPTION: This function loads the cross-sections
#
#*******************************************************************


def xsects():

    xsecpath='/data/mrline2/ABSCOEFF/SAMPLED_XSEC/' #location on agave where cross-sections live
    ### Read in x-section arrays
    # H2-H2
    file=xsecpath+'xsecarrH2H2_FREED_samp_3800_10000_R500000.h5'
    hf=h5py.File(file, 'r')
    wno=np.array(hf['wno'])
    T=np.array(hf['T'])
    P=np.array(hf['P'])
    xsecarrH2=np.array(hf['xsec'])
    hf.close()
    print 'H2'

    #define mega xsecarr 
    Ngas=7
    xsecarr=(np.ones((len(wno), len(T), len(P), Ngas))*(-50))

    ####
    xsecarr[:,:,:,0]=xsecarrH2.T
    del xsecarrH2

    # H2-He
    file=xsecpath+'xsecarrH2He_FREED_samp_3800_10000_R500000.h5'
    hf=h5py.File(file, 'r')
    wno=np.array(hf['wno'])
    T=np.array(hf['T'])
    P=np.array(hf['P'])
    xsecarr[:,:,:,1]=np.array(hf['xsec']).T
    hf.close()
    print 'He'

    # H2O
    file=xsecpath+'xsecarrH2O_POK_samp_3800_10000_R500000.h5'
    hf=h5py.File(file, 'r')
    wno=np.array(hf['wno'])
    #T=np.array(hf['T'])
    #P=np.array(hf['P'])
    xsecarr[:,:,:,2]=np.array(hf['xsec']).T#np.array(hf['xsec']).T[:,5:,:-1]
    for i in range(16): xsecarr[:,i,:,2]=xsecarr[:,16,:,2] #Ehsan's pokozatel xsecs stop at 500K, so forcing to 500K xsecs below 500K
    hf.close()
    print 'H2O'

    
    #CO
    file=xsecpath+'xsecarrCO_HITEMP_HELIOS_samp_3800_10000_R500000.h5'
    hf=h5py.File(file, 'r')
    wno=np.array(hf['wno'])
    #T=np.array(hf['T'])
    #P=np.array(hf['P'])
    xsecarr[:,:,:,3]=np.array(hf['xsec']).T
    hf.close()
    print 'CO'

    #'''
    # OH
    file=xsecpath+'xsecarrOH_HITEMP_HELIOS_samp_3800_10000_R500000.h5'
    hf=h5py.File(file, 'r')
    wno=np.array(hf['wno'])
    #T=np.array(hf['T'])
    #P=np.array(hf['P'])
    xsecarr[:,:,:,4]=np.array(hf['xsec']).T
    hf.close()
    print 'OH'

    #HmFF
    file=xsecpath+'xsecarrHMFF_samp_3800_10000_R500000.h5'
    hf=h5py.File(file, 'r')
    wno=np.array(hf['wno'])
    #T=np.array(hf['T'])
    #P=np.array(hf['P'])
    xsecarr[:,:,:,5]=np.array(hf['xsec']).T
    hf.close()
    print 'HmFF'

    #HmBF
    file=xsecpath+'xsecarrHMBF_samp_3800_10000_R500000.h5'
    hf=h5py.File(file, 'r')
    wno=np.array(hf['wno'])
    #T=np.array(hf['T'])
    #P=np.array(hf['P'])
    xsecarr[:,:,:,6]=np.array(hf['xsec']).T
    hf.close()
    print 'HmBF'

    # # HCN
    # file=xsecpath+'xsecarrHCN_EXOMOL_HELIOS_samp_3800_10000_R500000.h5'
    # hf=h5py.File(file, 'r')
    # wno=np.array(hf['wno'])
    # #T=np.array(hf['T'])
    # #P=np.array(hf['P'])
    # xsecarr[:,:,:,5]=np.array(hf['xsec']).T
    # hf.close()
    # print 'HCN'
    # #'''

    # # HCN
    # file=xsecpath+'xsecarrSiO_EBJT_EHSAN_samp_3800_10000_R500000.h5'
    # hf=h5py.File(file, 'r')
    # wno=np.array(hf['wno'])
    # #T=np.array(hf['T'])
    # #P=np.array(hf['P'])
    # xsecarr[:,:,:,6]=np.array(hf['xsec']).T
    # hf.close()
    # print 'SiO'
    # #'''

    # # HCN
    # file=xsecpath+'xsecarrTiO_TOTO_EHSAN_samp_3800_10000_R500000.h5'
    # hf=h5py.File(file, 'r')
    # wno=np.array(hf['wno'])
    # #T=np.array(hf['T'])
    # #P=np.array(hf['P'])
    # xsecarr[:,:,:,7]=np.array(hf['xsec']).T
    # hf.close()
    # print 'TiO'
    # #'''

    # # HCN
    # file=xsecpath+'xsecarrVO_VOMYT_EHSAN_samp_3800_10000_R500000.h5'
    # hf=h5py.File(file, 'r')
    # wno=np.array(hf['wno'])
    # #T=np.array(hf['T'])
    # #P=np.array(hf['P'])
    # xsecarr[:,:,:,7]=np.array(hf['xsec']).T
    # hf.close()
    # print 'VO'
    # #'''

    # # FeH
    # file=xsecpath+'xsecarrFeH_MOLLIST_EHSAN_samp_3800_10000_R500000.h5'
    # hf=h5py.File(file, 'r')
    # wno=np.array(hf['wno'])
    # #T=np.array(hf['T'])
    # #P=np.array(hf['P'])
    # xsecarr[:,:,:,5]=np.array(hf['xsec']).T
    # hf.close()
    # print 'FeH'

    # # C2H2
    # file=xsecpath+'xsecarrC2H2_ExoMol_HELIOS_SUPER_samp_3800_10000_R500000.h5'
    # hf=h5py.File(file, 'r')
    # wno=np.array(hf['wno'])
    # #T=np.array(hf['T'])
    # #P=np.array(hf['P'])
    # xsecarr[:,:,:,5]=np.array(hf['xsec']).T
    # hf.close()
    # print 'C2H2'

    # # CH4
    # file=xsecpath+'xsecarrCH4_HITEMP_HELIOS_samp_3800_10000_R500000.h5'
    # hf=h5py.File(file, 'r')
    # wno=np.array(hf['wno'])
    # #T=np.array(hf['T'])
    # #P=np.array(hf['P'])
    # xsecarr[:,:,:,6]=np.array(hf['xsec']).T
    # hf.close()
    # print 'CH4'

    # # 13CO
    # file=xsecpath+'xsecarrCO_ISOTOPE_HITEMP_HELIOS_samp_3800_10000_R500000.h5'
    # hf=h5py.File(file, 'r')
    # wno=np.array(hf['wno'])
    # #T=np.array(hf['T'])
    # #P=np.array(hf['P'])
    # xsecarr[:,:,:,8]=np.array(hf['xsec']).T
    # hf.close()
    # print 'C13O16'


    #cropping the wavenumber grid over selected range wnomin to wnomax (current full span is 3800 - 10000 cm-1 (1 - 2.63 um), native R=500K)
    wnomin =3500#minimum wno cut
    wnomax =9500#maximum wno cut
    loc=np.where((wno <= wnomax) & (wno >= wnomin))
    loc=loc[0]
    wno=wno[loc[::2]]  #sampling down: doing every-<number>-wavenumber-point (so R=250K instead of R=500K)--works ok for final R < 60K
    ###note, can probably crop in T and P as well to save memory to add more xsecs at once...
    xx=np.ascontiguousarray(xsecarr[loc[::2],:,:,:]) #putting as a "c-formatted" continuous array for GPU 
   
    del xsecarr
 
    print 'DONE READ'
    return np.log10(P),np.log10(T),wno,cuda.to_device(xx) #returing arrays, punting mastr xsecarry onto GPU--this eats the memory it's many GB big
  



#********************************************************************************


# FILE: TP.py
#
# DESCRIPTION: This function takes stellar, planetary, and atmospheric parameters 
# and returns the temperature-pressure profile. -- This is the Guillot 2010 (rather the Parmenteir 2014 modification) profile
#
# CALLING SEQUENCE: >>> tp = TP(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])
#
# NOTE: run '$ module load python' before running '$ python'
#
# Test: >>> x = [0.93,0.598,4400,0.01424,100,10.**3.7,10.**(-2.-2.),10.**(-1-2),10.**(-2),1.]
#       >>> from TP import TP
#       >>> tp = TP(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])
#       >>> T = tp[0]
#       >>> P = tp[1]
#********************************************************************************

def TP(Teq, Teeff, g00, kv1, kv2, kth, alpha):


    Teff = Teeff
    f = 1.0  # solar re-radiation factor
    A = 0.0  # planetary albedo
    g0 = g00

    # Compute equilibrium temperature and set up gamma's
    T0 = Teq
    gamma1 = kv1/kth
    gamma2 = kv2/kth

    # Initialize arrays
    logtau =np.arange(-10,20,.1)
    tau =10**logtau


    #computing temperature
    T4ir = 0.75*(Teff**(4.))*(tau+(2.0/3.0))
    f1 = 2.0/3.0 + 2.0/(3.0*gamma1)*(1.+(gamma1*tau/2.0-1.0)*sp.exp(-gamma1*tau))+2.0*gamma1/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma1*tau)
    f2 = 2.0/3.0 + 2.0/(3.0*gamma2)*(1.+(gamma2*tau/2.0-1.0)*sp.exp(-gamma2*tau))+2.0*gamma2/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma2*tau)
    T4v1=f*0.75*T0**4.0*(1.0-alpha)*f1
    T4v2=f*0.75*T0**4.0*alpha*f2
    T=(T4ir+T4v1+T4v2)**(0.25)
    P=tau*g0/(kth*0.1)/1.E5


    # Return TP profile
    return T, P


"""
  TP_MS: Temperature-Pressure profile from Madhu & Seager (2009), Equations (1,2)

  INPUTS:
  P: array of pressure points
  T0: T at the lowest pressure (highest altitude)
  P1,P2: highest pressure in lvl 1, turn-over pressure in level 2
  a1, a2: alpha_x as in paper
  P3: top of level 3, below which is assumed isothermal

  OUTPUTS:
  Returned array of temperature values that are are mapped onto the input pressure array P

"""
def TP_MS(P,T0,P1,P2,P3,a1,a2):

        P=P[::-1] #reverse array to be low -> high pressure
        n=P.shape[0]
        beta=0.5 #set beta as in paper
        Tarr=np.zeros(n)
        P0=P[-1] #first pressure point

        Tarr1=(np.log(P/P0)/a1)**(1./beta)+T0 #Layer 1 (Equation 1)
        T2=(np.log(P1/P0)/a1)**(1./beta)+T0-(np.log(P1/P2)/a2)**(1./beta)
        Tarr2=(np.log(P/P2)/a2)**(1./beta)+T2 #Layer 2 (Equation 2)

        Tarr[P<P1]=Tarr1[P<P1] #Load in Layer 1 equation into return array
        Tarr[P>=P1]=Tarr2[P>=P1] #Load in Layer 2 equation into return array
        loc3=np.where(P>=P3)[0]

        Tarr[loc3]=Tarr2[loc3][-1]

        Tarr[Tarr<100]=100
        #pdb.set_trace()
        return Tarr[::-1]



#**************************************************************
# FILE: restore.py
#
# DESCRIPTION: This class calls the function xsects(), thus 
# loading the x-sections as global variables.
#
# USAGE: >>> from restore import restore
#        >>> Pgrid = restore.xsects[0]
#
#**************************************************************

class restore():
    xsects = xsects()


#**************************************************************************
# FILE:get_rot_ker.py
# 
# DESCRIPTION: Computes rotation kernal for convolution
#  vsini is in km/s, wStar is the model wavelength grid (constant R)
#**************************************************************************
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


#**************************************************************************
# FILE:fx.py
# 
# DESCRIPTION: Fowrad model...this takes in "input parameters (x)"
# and sets it up to pass into pymultinest
# 
# USAGE: 
#**************************************************************************
#needs to take in data wavelength grid here
def fx_trans(x):
    #print(x)
    logH2O, logCO, logOH,  logHmFF, logHmBF, Tiso,   xRp, logPc =x

    npars=np.array(x).shape[0]
    fH2He = 1.-(10.**logH2O+10.**logCO+10**logOH+10**logHmFF+10**logHmBF)
    if fH2He < 0.0:
        fH2He = 0.0
    frac=0.176471
    fH2=fH2He/(1.+frac)
    fHe=frac*fH2
    # mmw = 2.*fH2 + 4.*fHe + (18.*10.**logH2O + 28.*10.**logCO + 17.*10.**logOH + 1.*10.**logHmFF + 1.*10.**logHmBF)
    mmw=1.3
    #planet params
    Rp=1.83
    Rstar=1.73
    Mp=0.92

    logP = np.arange(-8.,1.,0.08)+0.08 #FINDME--If you change this also change in 
    P = 10.0**logP
    T=P*0.+Tiso #making a constant temperature array
    #call the guillot TP function here, then interpolate in log(P) to the pressures here
    #np.interp
    Ps=0.1
  
    wnocrop, Depth, Z=tran(T, P,mmw+P[1:]*0.,Ps,10**logPc,10**logH2O+P[1:]*0., 10**logCO+P[1:]*0.,10**logOH+P[1:]*0.,10**logHmFF+P[1:]*0.,10**logHmBF+P[1:]*0.,fH2+P[1:]*0.,fHe+P[1:]*0., 0, 0,Mp,Rstar,Rp*xRp)
    #gas arrays are one element smaller because they're calculated at the midpoint.

    return wnocrop, Depth


