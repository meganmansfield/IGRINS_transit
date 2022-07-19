import pickle
from scipy import constants
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
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


def log_likelihood_PCA(Vsys, Kp, scale, cs_p, wl_data, pca_clean_data,pca_noplanet,phi,Vbary,numPCs):
    
    num_orders, num_files, num_pixels = pca_clean_data.shape

    I = np.ones(num_pixels)
    N = num_pixels
    
    #calculate predicted planetary radial velocity
    RV = Vsys + Vbary + Kp*np.sin(2.*np.pi*phi) 
    dl_l = RV*1E3 / constants.c

    # Initializing log-likelihoods and CCFs
    logL_Matteo = 0.  
    CCF = 0.

    # Looping through each phase and computing total log-L by summing logLs for each obvservation/phase
    for j in range(num_orders):
        wCut = wl_data[j,].copy() # Cropped wavelengths    
        gTemp=np.zeros((num_files,num_pixels))  #"shifted" model spectra array at each phase
        for i in range(num_files):
            wShift = wCut * (1.0 - dl_l[i])
            Depth_p = interpolate.splev(wShift, cs_p, der=0) * scale
            gTemp[i,] = Depth_p

        fData=(1.-gTemp)*pca_noplanet[j,]

        #faster SVD
        u,ss,vh=np.linalg.svd(fData,full_matrices=False)  #decompose
        ss[0:int(numPCs)]=0.
        W=np.diag(ss)
        A=np.dot(u,np.dot(W,vh))
        gTemp=A

        for i in range(num_files):	
            gVec=gTemp[i,].copy()
            gVec-=(gVec.dot(I))/float(num_pixels)  #mean subtracting the model here
            sg2=(gVec.dot(gVec))/float(num_pixels)
            fVec=pca_clean_data[j,i,].copy() # already mean-subtracted
            sf2=(fVec.dot(fVec))/num_pixels
            R=(fVec.dot(gVec))/num_pixels # cross-covariance
            CC=R/np.sqrt(sf2*sg2) # cross-correlation	
            CCF+=CC
            logL_Matteo+=(-0.5*N * np.log(sf2+sg2-2.0*R))	


    return logL_Matteo, CCF

#################
def run_CCF(wl_data,pca_clean_data,pca_noplanet,model,phi,Vbary,Kp,Vsys,scale,numPCs,Per,T14,vsini,name='test',output=True,verbose=False):

    #discard data taken outside of transit (doesn't contribute to planetary signal)
    print("Cropping out-of-transit data...")
    edgephase=T14/(Per*24.)/2.
    loc=np.where((phi>-edgephase)&(phi<edgephase))[0]
    phi=phi[loc]
    Vbary=Vbary[loc]
    pca_clean_data=pca_clean_data[:,loc,:]
    pca_noplanet=pca_noplanet[:,loc,:]

    num_orders, num_files, num_pixels = pca_clean_data.shape

    #set up array of Kp and Vsys values to perform cross-correlation at
    Kparr=np.linspace(-250,250,501)
    Vsysarr=np.linspace(-50,50,101)

    logLarr=np.zeros((len(Kparr),len(Vsysarr)))
    CCFarr=np.zeros((len(Kparr),len(Vsysarr)))
    
    wl_model=model[:,0]
    rprs_model=model[:,1]

    #convolution of model to correct for system rotation
    #essentially broadens the lines to account for the stellar rotation
    ker_rot=get_rot_ker(vsini, wl_model)
    model_conv_rot = np.convolve(rprs_model,ker_rot,mode='same')

    #convolution of model to add instrument broadening
    xker = np.arange(41)-20
    modelres=np.mean(wl_model[1:]/np.diff(wl_model))
    scalefactor=modelres/45000.
    sigma = scalefactor/(2.* np.sqrt(2.0*np.log(2.0)))

    yker = np.exp(-0.5 * (xker / sigma)**2.0)
    yker /= yker.sum()
    rprs_conv_final = np.convolve(model_conv_rot,yker,mode='same')
    coeff_spline = interpolate.splrep(wl_model,rprs_conv_final,s=0.0)

    print("Calculating cross-correlation...")
    print("Note: this calculation may take several hours to complete on a normal computer.")
    for i in range(len(Kparr)):
        if verbose:
            print("Kp step:"+str(i))
        for j in range(len(Vsysarr)):
            logL_M, CCF1=log_likelihood_PCA(Vsysarr[j], Kparr[i],scale, coeff_spline, wl_data, pca_clean_data,pca_noplanet,phi,Vbary,numPCs)
            logLarr[i,j]=logL_M
            CCFarr[i,j]=CCF1
            # if verbose:
            #     print(i, j, CCF1, logL_M)

    if output==True:
        pickle.dump([Vsysarr, Kparr,CCFarr,logLarr],open('PCA_'+name+'.pic','wb'))

    print("Creating cross-correlation plot...")

    #Calculate significance of detection after clipping any points 3-sigma or more away from the mean (to avoid diluting significance with high-sigma area near the peak)
    mean, med, stdev = sigma_clipped_stats(CCFarr,sigma=3.0)
    sigmaarr = (CCFarr-med)/stdev
    maxindices=np.unravel_index(sigmaarr.argmax(),sigmaarr.shape)
    if verbose:
        print("Maximum detection significance: "+float(sigmaarr.max())+". Maximum detection at Vsys="+float(Vsysarr[maxindices[1]])+", Kp="+float(Kparr[maxindices[0]]))

    fig,ax=plt.subplots()
    cax=ax.imshow(CCFarr,origin='lower', extent=[Vsysarr.min(),Vsysarr.max(), Kparr.min(),Kparr.max()],aspect="auto",interpolation='none',zorder=0)
    plt.scatter(Vsysarr[maxindices1[1]],Kparr[maxindices1[0]],s=80,color='k',marker='x',zorder=4)
    cbar=plt.colorbar(cax)
    plt.axvline(x=Vsys,color='white',ls='--',lw=2,zorder=1)
    plt.axhline(y=Kp,color='white',ls='--',lw=2,zorder=1)
    plt.xlabel('$\Delta$V$_{sys}$ [km/s]',fontsize=20)
    plt.ylabel('K$_{p}$ [km/s]',fontsize=20)
    cbar.set_label('Cross-correlation coefficient',fontsize=15)
    cbar.ax.tick_params(labelsize=15,width=2,length=6)
    plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
    plt.tight_layout()
    plt.savefig('CCF'+name+'.pdf',fmt='pdf')
    plt.show()

    rc('axes',linewidth=2)
    fig,ax=plt.subplots()
    cax=ax.imshow(sigmaarr,origin='lower', extent=[Vsysarr.min(),Vsysarr.max(), Kparr.min(),Kparr.max()],aspect="auto",interpolation='none',zorder=0)
    plt.scatter(Vsysarr[maxindices1[1]],Kparr[maxindices1[0]],s=80,color='k',marker='x',zorder=4)
    cbar=plt.colorbar(cax)
    plt.axvline(x=Vsys,color='white',ls='--',lw=2,zorder=1)
    plt.axhline(y=Kp,color='white',ls='--',lw=2,zorder=1)
    plt.xlabel('$\Delta$V$_{sys}$ [km/s]',fontsize=20)
    plt.ylabel('K$_{p}$ [km/s]',fontsize=20)
    cbar.set_label('Significance ($\sigma$)',fontsize=15)
    cbar.ax.tick_params(labelsize=15,width=2,length=6)
    plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
    plt.tight_layout()
    plt.savefig('CCF_SNR'+name+'.pdf',fmt='pdf')
    plt.show()

    return CCFarr




