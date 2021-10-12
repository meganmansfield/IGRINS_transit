'''
Run full IGRINS pipeline; code shows places where parameters can be changed
'''

import make_cube
import numpy as np

path='/Users/megan/Documents/Projects/GeminiTransitSurvey/WASP107/20210418/reduced/' #path to reduced data
date='20210418' #date of observations
Tprimary_UT='2021-04-19T03:55:00.000' #time of primary transit midpoint for this epoch
Per=5.721492 #period
radeg=188.3864167 #RA of target in degrees
decdeg=-10.1462139 #Dec of target in degrees
skyorder=1 #1 if sky frame was taken first in the night, 2 if sky frame was taken second
badorders=np.array([0,21,22,23,24,25,26,52,53]) #set which orders will get ignored. Lower numbers are at the red end of the spectrum, and IGRINS has 54 orders
trimedges=np.array([100,-100]) #set how many points will get trimmed off the edges of each order: fist number is blue edge, second number is red edge

#hmmmm...need to figure out in make_cube how to select all but the badorders.
phi,Vbary,wlgrid,data_RAW=make_cube.make_cube(path,date,Tprimary_UT,Per,radeg,decdeg,skyorder,badorders,trimedges,plot=True,output=False)