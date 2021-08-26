#6/10/21 cool, everything is working in this file. Use this version of make_cube
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import pdb
import glob
import os
import matplotlib.pyplot as plt

path='./WASP107/20210418/reduced/' #path to reduced data
date='20210418' #date of observations
Tprimary_UT='2021-04-19T03:55:00.000' #time of primary transit midpoint for this epoch
Per=5.721492 #period
radeg=188.3864167 #RA of target in degrees
decdeg=-10.1462139 #Dec of target in degrees
skyorder=1 #1 if sky frame was taken first in the night, 2 if sky frame was taken second

filearr_specH=sorted(glob.glob(path+'*SDCH*spec.fits'))
filearr_specK=sorted(glob.glob(path+'*SDCK*spec.fits'))

#for getting shape
firstH=fits.open(filearr_specH[0])
firstK=fits.open(filearr_specK[0])

Nphi=len(filearr_specH)
#Ndet, Npix=wlgrid.shape
Ndet=firstH[0].data.shape[0]+firstK[0].data.shape[0]
Npix=firstH[0].data.shape[1]
time_MJD=np.zeros(Nphi)

data_RAW=np.zeros((Ndet, Nphi,Npix))
wlgrid=np.zeros((Ndet,Npix))
for i in range(len(filearr_specH)):

	#H
	hdu_list = fits.open(filearr_specH[i])
	image_dataH = hdu_list[0].data
	# wavelengthH = hdu_list[1].data 
	hdr=hdu_list[0].header
	date_beginH=hdr['DATE-OBS'] #in UTC
	date_endH=hdr['DATE-END']
	#time_MJD=hdr['MJD-OBS']
	t1=Time(date_beginH,format='isot',scale='utc')
	t2=Time(date_endH,format='isot',scale='utc')

	time_MJD[i]=float(0.5*(t1.mjd+t2.mjd))
	print(date_beginH, date_endH, time_MJD[i])


	#print('H',date_begin, date_end)
	#plot(wlgridH[-10,],image_dataH[-10,])
	#K
	hdu_list = fits.open(filearr_specK[i])
	image_dataK = hdu_list[0].data
	# wavelengthK = hdu_list[1].data 
	hdr=hdu_list[0].header
	date_beginK=hdr['DATE-OBS'] #in UTC
	date_endK=hdr['DATE-END']
	if date_beginK!=date_beginH:
		print('Error: H and K files are misaligned')
	#print(date_begin,date_end)
	#plot(wlgridK[10,],image_dataK[10,])
	data=np.concatenate([image_dataK,image_dataH])
	data_RAW[:,i,:]=data #Ndet, Nphi, Npix
	# wlgrid[:,i,:]=np.concatenate([wavelengthK,wavelengthH])
	#pdb.set_trace()

if skyorder==1:
	wavefileH=fits.open(filearr_specH[0]) #file that is closest to the wavelength calibration in time
	wavefileK=fits.open(filearr_specK[0])
	wlgrid=np.concatenate([wavefileK[1].data,wavefileH[1].data])
elif skyorder==2:
	wavefileH=fits.open(filearr_specH[-1]) #file that is closest to the wavelength calibration in time
	wavefileK=fits.open(filearr_specK[-1])
	wlgrid=np.concatenate([wavefileK[1].data,wavefileH[1].data])

masked=np.isnan(data_RAW)
#pdb.set_trace()
#comuting phase array
tprimary=Time(Tprimary_UT,format='isot',scale='utc')
t0=tprimary.mjd #in days
#Per=1.360031 #period days
phi=(time_MJD-t0)/Per



#barycorrection
gemini = EarthLocation.from_geodetic(lat=-30.2407*u.deg, lon=-70.7366*u.deg, height=2722*u.m)
sc = SkyCoord(ra=radeg*u.deg, dec=decdeg*u.deg)

#gemini = EarthLocation.from_geodetic(lat=-24.6274*u.deg, lon=-70.4039*u.deg, height=2715*u.m)
#sc = SkyCoord(ra=37.155108*u.deg, dec=-7.060664*u.deg)

Vbary=np.zeros(len(time_MJD))
for i in range(len(time_MJD)):
	barycorr = sc.radial_velocity_correction(obstime=Time(time_MJD[i],format='mjd'), location=gemini)  
	Vbary[i]=-barycorr.to(u.km/u.s).value


import pickle
pickle.dump(phi,open('phi.pic','wb'),protocol=2)
pickle.dump(Vbary,open('Vbary.pic','wb'),protocol=2)
pickle.dump([wlgrid,data_RAW,skyorder],open('data_RAW_'+date+'.pic','wb'),protocol=2)
pickle.dump(masked,open('masked.pic','wb'),protocol=2)

for i in range(len(time_MJD)):
	print(time_MJD[i], Vbary[i])


#for plotting SNR
filearr_snrH=sorted(glob.glob(path+'*SDCH*sn.fits'))
filearr_snrK=sorted(glob.glob(path+'*SDCK*sn.fits'))
snr_RAW=np.zeros((Ndet, Nphi,Npix))
for i in range(len(filearr_snrH)):
	hdu_list = fits.open(filearr_snrH[i])
	image_snrH = hdu_list[0].data
	hdu_list = fits.open(filearr_snrK[i])
	image_snrK = hdu_list[0].data
	snr_RAW[:,i,:]=np.concatenate([image_snrK,image_snrH])

whichorders=[  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,17, 28, 29, 30, 31, 32, 33,34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
wlgrid=wlgrid[whichorders,100:-100]  #cropping edge of orders, converting to um

snr_RAW=snr_RAW[whichorders,:,100:-100] #cropping edge of orders
snr_RAW[np.isnan(snr_RAW)]=0. #pruging NaN's
snr_RAW[snr_RAW <0.]=0. #purging negative values
Ndet, Npix=wlgrid.shape

#median over phases
med1=np.median(snr_RAW,axis=1)
plt.figure()
for i in range(Ndet): plt.plot(wlgrid[i,:],med1[i,],color='red')

#median of each order
med2=np.median(med1, axis=1)
medwl=np.median(wlgrid,axis=1)
plt.plot(medwl, med2,'ob')

plt.show()

# data_RAW=data_RAW[whichorders,:,100:-100] #cropping edge of orders
# data_RAW[np.isnan(data_RAW)]=0. #pruging NaN's
# data_RAW[data_RAW <0.]=0. #purging negative values
# Ndet, Npix=wlgrid.shape

# #median over phases
# med1=np.median(data_RAW,axis=1)
# plt.figure()
# for i in range(Ndet): plt.plot(wlgrid[i,:],med1[i,],color='red')

# #median of each order
# med2=np.median(med1, axis=1)
# medwl=np.median(wlgrid,axis=1)
# plt.plot(medwl, med2,'ob')

# plt.show()



