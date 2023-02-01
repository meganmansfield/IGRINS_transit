from scipy import constants
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import rc

#rotation kernel
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

def trail(wl_data,pca_clean_data,pca_noplanet,model,phi,Vbary,Kp,Vsys,scale,Per,T14,vsini,verbose=False):

    print("Calculating trail...")
    print("Note: trail may take a while to compute. Turn on verbose to get updates on the status of the calculation.")
    
    #Calculate phases corresponding to edge of transit
    edgephase=T14/(Per*24.)/2.

    num_orders, num_files, num_pixels = pca_clean_data.shape
    
    #calculate predicted planetary radial velocity
    RV = Vsys + Vbary + Kp*np.sin(2.*np.pi*phi) 
    #calculate predicted stellar radial velocity
    starRV= Vsys + Vbary

    #convolution of model to correct for system rotation
    #essentially broadens the lines to account for the planetary rotation
    wl_model=model[:,0]
    rprs_model=model[:,1]
    ker_rot=get_rot_ker(vsini, wl_model)
    model_conv_rot = np.convolve(rprs_model,ker_rot,mode='same')

    #convolution of model to add instrument broadening
    xker = np.arange(41)-20
    modelres=np.mean(wl_model[1:]/np.diff(wl_model))
    scalefactor=modelres/45000. #resolution of IGRINS is ~45k. Would need to change for other instruments
    sigma = scalefactor/(2.* np.sqrt(2.0*np.log(2.0)))

    yker = np.exp(-0.5 * (xker / sigma)**2.0)
    yker /= yker.sum()
    rprs_conv_final = np.convolve(model_conv_rot,yker,mode='same')

    #spline coefficients for interpolating model to data wavelength grid
    coeff_spline = interpolate.splrep(wl_model,rprs_conv_final,s=0.0) 

    #Setting up velocity grid for making trail
    maxnum=int(np.max((abs(np.min(RV)),np.max(RV))))+15
    Vel=np.linspace(-maxnum,maxnum,4*maxnum+1) #set up velocity grid with step size of 0.5 km/s
    Nvel=len(Vel)

    I = np.ones(num_pixels)
    N = num_pixels

    # Initializing log-likelihoods and CCFs
    logL_Matteo = 0.  
    CCF = 0.
    CCFarr=np.zeros((Nvel,num_orders, num_files))
    logL_Matteo=np.zeros((Nvel,num_orders, num_files))

    # Looping through each phase and computing total log-L by summing logLs for each obvservation/phase
    for kk in range(Nvel):
        if verbose:
            print(kk) 
        for j in range(num_orders):
            wCut = wl_data[j,].copy() # Cropped wavelengths    
            gTemp=np.zeros((num_files,num_pixels))  #"shifted" model spectra array at each phase
            dl_l=Vel[kk]*1E3/constants.c
            wShift = wCut * (1.0 - dl_l)
            rpshift=interpolate.splev(wShift,coeff_spline,der=0)*scale

            for i in range(num_files):
                gTemp[i,:]=rpshift  
                gVec=gTemp[i,:].copy()
                gVec-=(gVec.dot(I))/float(num_pixels)  #mean subtracting the model here
                sg2=(gVec.dot(gVec))/float(num_pixels)
                fVec=pca_clean_data[j,i,:].copy() # already mean-subtracted
                sf2=(fVec.dot(fVec))/num_pixels
                R=(fVec.dot(gVec))/num_pixels # cross-covariance
                CC=R/np.sqrt(sf2*sg2) # cross-correlation   
                CCFarr[kk,j,i]=CC
                logL_Matteo[kk,j,i]=(-0.5*num_pixels * np.log(sf2+sg2-2.0*R))

    CCF=np.sum(CCFarr,axis=1).T
    logL=np.sum(logL_Matteo,axis=1)

    for i in range(num_files): CCF[i,]=CCF[i,]-np.mean(CCF[i,:])
    for i in range(num_files): logL[i,:]=logL[i,:]-np.mean(logL[i,:])

    rc('axes',linewidth=2)
    fig,ax=plt.subplots()
    cax=ax.imshow(CCF,origin='lower', extent=[Vel.min(),Vel.max(), phi.min(),phi.max()],aspect='auto',interpolation='none')
    cbar=plt.colorbar(cax)
    cbar.ax.tick_params(labelsize=15,width=2,length=6)
    cbar.set_label('Cross-correlation coefficient',fontsize=15)
    ax.plot(RV[abs(phi)<edgephase], phi[abs(phi)<edgephase],color='red',linewidth=2,ls='--',alpha=0.5)
    ax.plot(starRV[abs(phi)<edgephase], phi[abs(phi)<edgephase], color='blue',linewidth=2,ls='--',alpha=0.5)
    plt.axhline(y=edgephase,color='white',linestyle='--')
    plt.axhline(y=-edgephase,color='white',linestyle='--')
    plt.xlabel('Radial velocity [km/s]',fontsize=20)
    plt.ylabel('Phase',fontsize=20)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.3)
    plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
    plt.tight_layout()
    plt.savefig('trail.pdf',fmt='pdf')
    plt.show()

    return CCF




