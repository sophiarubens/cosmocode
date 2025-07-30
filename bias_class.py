import numpy as np
import camb
from camb import model
from scipy.signal import convolve2d,convolve
from matplotlib import pyplot as plt
from power_class import *
import time

"""
this module helps compute contaminant power and cosmological parameter biases using
a Fisher-based formalism using two complementary strategies with different scopes:
1. analytical windowing for a cylindrically symmetric Gaussian beam
2. numerical  windowing for a Gaussian beam with different x- and y-pol widths
"""

class NotYetImplementedError(Exception):
    pass

class window_calcs(object):
    def __init__(self,
                 kpar,kperp,                                            # set by survey properties
                 primary_beam_type,primary_beam_args,primary_beam_uncs, # 
                 pars_set_cosmo,pars_forecast,                          # 
                 z,n_sph_modes,dpar                                     # 
                 ):
        """
        kpar,kperp        :: (Nkpar,),(Nkperp,) of floats :: mono incr. cyl binned curvey modes       :: 1/Mpc
        primary_beam      :: callable                     :: power beam in Cartesian coords           :: ---
        primary_beam_type :: str                          :: implement soon: Airy etc.                :: ---
        primary_beam_args :: (N_args,) of floats          :: Gaussian: "μ"s and "σ"s                  :: Gaussian: sigLoS, r0 in Mpc; fwhm_x, fwhm_y in rad
        primary_beam_uncs :: (N_uncertain_args) of floats :: Gaussian: frac uncs epsLoS, epsfwhm{x/y} :: ---
        pars_set_cosmo    :: (N_fid_pars,) of floats      :: params to condition a CAMB/etc. call     :: as found in ΛCDM
        pars_forecast     :: (N_forecast_pars,) of floats :: params for which you'd like to forecast  :: as found in ΛCDM
        z                 :: float                        :: z of fiducial MPS for CAMB/etc. call     :: ---
        n_sph_modes       :: int                          :: # modes to put in CAMB/etc. MPS          :: ---
        dpar              :: (N_forecast_pars,) of floats :: initial guess of num. dif. step sizes    :: same as for pars_forecast
        """
        self.kpar=kpar
        self.kperp=kperp

        if (primary_beam_type.lower()!="gaussian"):
            raise NotYetImplementedError
        self.primary_beam_type=primary_beam_type
        self.primary_beam_args=primary_beam_args
        self.primary_beam_uncs=primary_beam_uncs
        
        self.pars_set_cosmo=pars_set_cosmo
        self.pars_forecast=pars_forecast
        self.z=z
        self.n_sph_modes=n_sph_modes
        self.dpar=dpar