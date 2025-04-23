#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: debanjansarkar
"""
import numpy as np
from astropy import units as u
from astropy.cosmology.units import littleh
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum
#import py21cmsense as p21
#print(p21.__version__)

#%%

class chord_sense(object):
    
    def __init__(
        self,
        spacing=[6,6],
        n_side=[6,11],
        orientation=None,
        center=[0,0],
        
        freq_cen = 700*u.MHz,
        dish_size = 6*u.m,
        Trcv = 30*u.K,
        latitude = (49.3*np.pi/180.0)*u.radian,
        integration_time= 10*u.s, # The time in sec, telescope integrates to give one sanpshot
        time_per_day = 6*u.hour,  # The time in hours, to observe per day (a typical choice of 8 hrs)
        n_days = 100 ,    # Total number of days of observation
        n_channels = 64 , # The number of channels
        bandwidth = 32*u.MHz,  # Bandwidth of obs
        coherent = False, # Whether to add different baselines coherently if they are not instantaneously redundant.
        bl_max = 120.0*u.m, # max baseline to include in uv-plane in meters
        bl_min =6.0*u.m,
        #tsky_ref_freq = 150.0 * u.MHz, 
        #tsky_amplitude = 260 *u.K,
        tsky_ref_freq = 400.0 * u.MHz, 
        tsky_amplitude = 25 *u.K,
        
        horizon_buffer = 0.1 * littleh/ u.Mpc,
        foreground_model = 'optimistic'
    ):
        self.spacing = spacing
        self.n_side = n_side
        self.orientation = orientation
        self.center = center
        self.freq_cen = freq_cen
        self.dish_size = dish_size
        self.Trcv =  Trcv
        self.latitude = latitude
        self.integration_time = integration_time
        self.time_per_day = time_per_day
        self.n_days = n_days
        self.n_channels = n_channels
        self.bandwidth = bandwidth
        self.coherent = coherent
        self.bl_max = bl_max
        self.bl_min = bl_min
        self.tsky_ref_freq = tsky_ref_freq
        self.tsky_amplitude = tsky_amplitude
        self. horizon_buffer =  horizon_buffer
        self.foreground_model = foreground_model
        
        
    def rectangle_generator(self):

        """
        ------------------------------------------------------------------------
        Generate a grid of baseline locations filling a rectangular array for CHORD/HIRAX. 
    
        Inputs:
            spacing      [2-element list or numpy array] positive integers specifying
                 the spacing between antennas. Must be specified, no default.
            n_side       [2-element list or numpy array] positive integers specifying
                 the number of antennas on each side of the rectangular array.
                 Atleast one value should be specified, no default.
            orientation  [scalar] counter-clockwise angle (in degrees) by which the
                 principal axis of the rectangular array is to be rotated.
                 Default = None (means 0 degrees)
            center       [2-element list or numpy array] specifies the center of the
                 array. Must be in the same units as spacing. The rectangular
                 array will be centered on this position.
        Outputs:
            Two element tuple with these elements in the following order:
            xy           [2-column array] x- and y-locations. x is in the first
                 column, y is in the second column. Number of xy-locations
                 is equal to the number of rows which is equal to n_total
            id           [numpy array of string] unique antenna identifier. Numbers
                 from 0 to n_antennas-1 in string format.
                 Notes:
        ------------------------------------------------------------------------
        """
        try:
            self.spacing
        except NameError:
            raise NameError('No spacing provided.')

        if self.spacing is not None:
            if not isinstance(self.spacing, (int, float, list, np.ndarray)):
                raise TypeError('spacing must be a scalar or list/numpy array')
            self.spacing = np.asarray(self.spacing)
            if self.spacing.size < 2:
                self.spacing = np.resize(self.spacing,(1,2))
            if np.all(np.less_equal(self.spacing,np.zeros((1,2)))):
                raise ValueError('spacing must be positive')

        if self.orientation is not None:
            if not isinstance(self.orientation, (int,float)):
                raise TypeError('orientation must be a scalar')

        if self.center is not None:
            if not isinstance(self.center, (list, np.ndarray)):
                raise TypeError('center must be a list or numpy array')
            self.center = np.asarray(self.center)
            if self.center.size != 2:
                raise ValueError('center should be a 2-element vector')
            self.center = self.center.reshape(1,-1)

        if self.n_side is None:
            raise NameError('Atleast one value of n_side must be provided')
        else:
            if not isinstance(self.n_side,  (int, float, list, np.ndarray)):
                raise TypeError('n_side must be a scalar or list/numpy array')
            self.n_side = np.asarray(self.n_side)
            if self.n_side.size < 2:
                self.n_side = np.resize(self.n_side,(1,2))
            if np.all(np.less_equal(self.n_side,np.zeros((1,2)))):
                raise ValueError('n_side must be positive')

            n_total = np.prod(self.n_side, dtype=np.uint8)
            xn,yn = np.hsplit(self.n_side,2)
            xn = xn.item()
            yn = yn.item()

            xs,ys = np.hsplit(self.spacing,2)
            xs = xs.item()
            ys = ys.item()

            n_total = xn*yn

            x = np.linspace(0, xn-1, xn)
            x = x - np.mean(x)
            x = x*xs

            y = np.linspace(0, yn-1, yn)
            y = y - np.mean(y)
            y = y*ys 
        
            z = np.zeros(n_total)
        
            xv, yv = np.meshgrid(x,y)

            xy = np.hstack((xv.reshape(-1,1),yv.reshape(-1,1)))

        if len(xy) != n_total:
            raise ValueError('Sizes of x- and y-locations do not agree with n_total')

        if self.orientation is not None:   # Perform any rotation
            angle = np.radians(self.orientation)
            rot_matrix = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            xy = np.dot(xy, rot_matrix.T)

        if self.center is not None:   # Shift the center
            xy += self.center
     
        z = np.zeros(shape=(n_total,1))
        XY = np.hstack((xy,z))

        return (np.asarray(XY)*u.m)
    
    
    def sense_1d(self):
        ant_pos = self.rectangle_generator()
        
        observatory = Observatory(antpos=ant_pos,
                          beam = GaussianBeam(frequency=self.freq_cen,
                                              dish_size=self.dish_size),
                          Trcv = self.Trcv,   # The receiver temp will dominate over sky temp at this freq. (unlike EoR)
                          latitude = self.latitude)
        
        observation = Observation(observatory = observatory,
                          integration_time = self.integration_time, # The time in sec, telescope integrates to give one sanpshot
                          time_per_day = self.time_per_day,  # The time in hours, to observe per day (a typical choice of 8 hrs)
                          #hours_per_day = self.time_per_day,  # The time in hours, to observe per day (a typical choice of 8 hrs)
                          n_days = self.n_days,    # Total number of days of observation
                          n_channels = self.n_channels, # The number of channels
                          bandwidth = self.bandwidth,  # Bandwidth of obs
                          coherent = self.coherent, # Whether to add different baselines coherently if they are not instantaneously redundant.
                          bl_max = self.bl_max, # max baseline to include in uv-plane in meters
                          bl_min = self.bl_min,
                          tsky_ref_freq = self.tsky_ref_freq,
                          tsky_amplitude = self.tsky_amplitude
                          )
        print(observation.n_days)
        sensitivity = PowerSpectrum(
            observation = observation,
            horizon_buffer = self. horizon_buffer,
            foreground_model = self.foreground_model)
        
        power_thermal = sensitivity.calculate_sensitivity_1d(thermal=True, sample=False)#only thermal
        
        return sensitivity.k1d, power_thermal
    
    def Nkmode(self):
        k_bins,pth = self.sense_1d()
        k_min = k_bins[0]
        #k_max = k_bins[-1]
        Vs = np.float_power(2.*np.pi/k_min,3.0) #approx
        Nk = Vs * np.float_power(k_bins,2.0) * (k_bins[1]-k_bins[0]) / 2 / np.float_power(np.pi, 2.0)
        return np.array(Nk)
        
        
    
'''    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    sen =chord_sense()
    ant_pos = sen.rectangle_generator()
    plt.figure(figsize=(6,4),dpi=80)
    plt.plot(ant_pos[:,0],ant_pos[:,1],marker='o',color='red',linestyle='none')
    plt.xlabel('X [m]',fontsize=18)
    plt.ylabel('Y [m]',fontsize=18)
    plt.tight_layout()
    plt.show()  
    plt.clf()
    
    k,pk = sen.sense_1d()
    plt.plot(k,pk)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()    
'''    
    
    
    
    
    