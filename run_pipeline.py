'''
Run full IGRINS pipeline; code shows places where parameters can be changed
'''

import make_cube
import numpy as np
import wavecal
import do_pca
import trail

#current params for W76
path='/Users/megan/Documents/Projects/GeminiTransitSurvey/WASP76/20211029/reduced/' #path to reduced data
date='20211029' #date of observations
Tprimary_UT='2021-10-30T04:20:00.000' #time of primary transit midpoint for this epoch
Per=1.809886 #period
radeg=26.6329167 #RA of target in degrees
decdeg=2.7003889 #Dec of target in degrees
skyorder=1 #1 if sky frame was taken first in the night, 2 if sky frame was taken second
badorders=np.array([0,21,22,23,24,25,26,27,52,53]) #set which orders will get ignored. Lower numbers are at the red end of the spectrum, and IGRINS has 54 orders
trimedges=np.array([100,-100]) #set how many points will get trimmed off the edges of each order: fist number is blue edge, second number is red edge
Vsys=-1.109 #Known systemic velocity from GAIA DR2
scale=1.

#Make data cube and calculate barycentric velocity
phi,Vbary,grid_RAW,data_RAW,wlgrid,data=make_cube.make_cube(path,date,Tprimary_UT,Per,radeg,decdeg,skyorder,badorders,trimedges,plot=False,output=False,testorders=False)

#Perform wavelength calibration
wlgrid,wavecorrect=wavecal.correct(wlgrid,data,skyorder,plot=False,output=False)

#Perform PCA
wlgrid,pca_clean_data,pca_noplanet=do_pca.do_pca(wlgrid,wavecorrect,4,test_pca=False,output=True)

#Make trail plot
#FINDME: strip out-of-transit stuff from trail plot and CCF
#Model must be in 2-column format: 1st column wavelengths, 2nd column transit depth; Lines 35-37
#Figure out how I calculated Kp and dphi so that I can code those in for real. Lines 38-39
#How did I get vsini for convoluting model with stellar rotation? Lines 46-48
#Lines 50-56 change so you can feed in models at any resolution.
trailCCF=trail.trail(wlgrid,pca_clean_data,pca_noplanet,model,phi,Vbary,Vsys,scale)

#Perform CCF


