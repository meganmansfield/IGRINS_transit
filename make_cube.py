"""
This script reads in IGRINS data which has been reduced using the IGRINS Pipeline Package (PLP)


"""

#import packages
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import glob
import os
import matplotlib.pyplot as plt
import pickle
from matplotlib import rc

def make_cube(path,date,Tprimary_UT,Per,radeg,decdeg,skyorder,exptime,badorders,trimedges,plot=True,output=False,testorders=False):
	#make list of observed files
	filearr_specH=sorted(glob.glob(path+'*SDCH*spec.fits'))
	filearr_specK=sorted(glob.glob(path+'*SDCK*spec.fits'))

	#use one file to get shape
	firstH=fits.open(filearr_specH[0])
	firstK=fits.open(filearr_specK[0])

	num_files=len(filearr_specH) #number of observed spectra / different phases observed

	num_orders=firstH[0].data.shape[0]+firstK[0].data.shape[0] #number of orders
	num_pixels=firstH[0].data.shape[1] #number of pixels per order
	time_start=np.zeros(num_files)

	print('Making data cube...')
	data_RAW=np.zeros((num_orders,num_files,num_pixels))
	wlgrid=np.zeros((num_orders,num_pixels))
	for i in range(len(filearr_specH)):

		#H
		hdu_list = fits.open(filearr_specH[i])
		image_dataH = hdu_list[0].data
		hdr=hdu_list[0].header
		date_beginH=hdr['DATE-OBS']
		date_endH=hdr['DATE-END']
		t1=Time(date_beginH,format='isot',scale='utc') #date of observation, in TimeISOT format from astropy
		time_start[i]=t1.mjd

		#K
		hdu_list = fits.open(filearr_specK[i])
		image_dataK = hdu_list[0].data
		hdr=hdu_list[0].header
		date_beginK=hdr['DATE-OBS']
		date_endK=hdr['DATE-END']
		if date_beginK!=date_beginH:
			print('ERROR: H and K files are misaligned')
		data=np.concatenate([image_dataK,image_dataH])
		data_RAW[:,i,:]=data

	#use the file that is closest in time to the wavelength calibration as the template wavelength solution
	if skyorder==1:
		wavefileH=fits.open(filearr_specH[0])
		wavefileK=fits.open(filearr_specK[0])
		wlgrid=np.concatenate([wavefileK[1].data,wavefileH[1].data])
	elif skyorder==2:
		wavefileH=fits.open(filearr_specH[-1])
		wavefileK=fits.open(filearr_specK[-1])
		wlgrid=np.concatenate([wavefileK[1].data,wavefileH[1].data])

	#calculating observed phases
	print('Calculating observed phases...')
	gemini = EarthLocation.from_geodetic(lat=-30.2407*u.deg, lon=-70.7366*u.deg, height=2722*u.m)
	tprimary=Time(Tprimary_UT, format='isot', scale='tdb', location=gemini).mjd
	phi=np.zeros(num_files)
	for i in range(num_files):
		phi[i]=(time_start[i]+.5*exptime/3600./24.-tprimary)/Per

	#barycentric velocity correction
	print('Calculating barycentric velocity...')
	sc = SkyCoord(ra=radeg*u.deg, dec=decdeg*u.deg)

	Vbary=np.zeros(len(time_start))
	for i in range(len(time_start)):
		barycorr = sc.radial_velocity_correction(obstime=Time(time_start[i],format='mjd'), location=gemini)  
		Vbary[i]=-barycorr.to(u.km/u.s).value

	#Create output files
	if output==True:
		pickle.dump(phi,open('phi.pic','wb'),protocol=2)
		pickle.dump(Vbary,open('Vbary.pic','wb'),protocol=2)
		pickle.dump([wlgrid,data_RAW,skyorder],open('data_RAW_'+date+'.pic','wb'),protocol=2)

	print('Mean barycentric velocity during observation period is '+str("{:.3f}".format(np.mean(Vbary)))+' km/s')

	#for plotting SNR

	
	filearr_snrH=sorted(glob.glob(path+'*SDCH*sn.fits'))
	filearr_snrK=sorted(glob.glob(path+'*SDCK*sn.fits'))
	snr_RAW=np.zeros((num_orders,num_files,num_pixels))
	for i in range(len(filearr_snrH)):
		hdu_list = fits.open(filearr_snrH[i])
		image_snrH = hdu_list[0].data
		hdu_list = fits.open(filearr_snrK[i])
		image_snrK = hdu_list[0].data
		snr_RAW[:,i,:]=np.concatenate([image_snrK,image_snrH])

	if len(badorders)>0:
		cwlgrid=np.delete(wlgrid,badorders,axis=0)
	else:
		cwlgrid=wlgrid
	cwlgrid=cwlgrid[:,trimedges[0]:trimedges[1]]
	if len(badorders)>0:
		csnr_RAW=np.delete(snr_RAW,badorders,axis=0)
	else:
		csnr_RAW=snr_RAW
	csnr_RAW=csnr_RAW[:,:,trimedges[0]:trimedges[1]]
	csnr_RAW[np.isnan(csnr_RAW)]=0. #remove NaNs
	csnr_RAW[csnr_RAW <0.]=0. #remove negative flux values
	num_orders, num_pixels=cwlgrid.shape

	if plot==True:
		#calculate median over phases
		med1=np.median(csnr_RAW,axis=1)

		rc('axes',linewidth=2)
		plt.figure(figsize=(8,5))
		for i in range(num_orders): plt.plot(cwlgrid[i,:],med1[i,],color='red')

		#median of each order
		med2=np.median(med1, axis=1)
		compare=np.max(med2)
		if testorders==True:
			below100=[]
			below200=[]
			below75per=[]
			for i in range(num_orders):
				if med2[i]<100.:
					below100.append(i)
				elif med2[i]<200.:
					below200.append(i)
				if med2[i]<0.7*compare:
					below75per.append(i)
			print('Orders with SNR<100: ',below100)
			print('Orders with 100<SNR<200: ',below200)
			print('Orders with <70\% transmittance: ',below75per)
		medwl=np.median(cwlgrid,axis=1)
		plt.plot(medwl, med2,'ob')
		plt.xlabel('Wavelength [$\mu$m]',fontsize=20)
		plt.ylabel('Signal-to-Noise',fontsize=20)
		plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
		plt.tight_layout()
		plt.savefig('SNR_per_order.png')
		plt.show()

	#Clean the NaNs and negative flux values from the data, and remove unwanted orders
	print('Cleaning data and removing unwanted orders...')
	if len(badorders)>0:
		cdata=np.delete(data_RAW,badorders,axis=0)
	else:
		cdata=data_RAW
	cdata=cdata[:,:,trimedges[0]:trimedges[1]]
	cdata[np.isnan(cdata)]=0. #purging NaNs
	cdata[cdata <0.]=0. #purging negative values
	
	return phi,Vbary,wlgrid,data_RAW,cwlgrid,cdata



