"""This function is a python reproduction of the analytic beam in the RTS
https://github.com/ICRAR/mwa-RTS, all credit to the original Authors"""

import numpy as np
import erfa

##angle conversions
DH2R = 0.26179938779914943653855361527329190701643078328126
DD2R = 0.017453292519943295769236907684886127134428718885417
DS2R = 7.2722052166430399038487115353692196393452995355905e-5

## astro constats
VEL_LIGHT = 299792458.0

## MWA defaults
MWA_LAT = -26.703319
MWA_LAT_RAD = MWA_LAT*DD2R

## RTS MWA FEE beam constants
NUM_DIPOLES = 16
DQ = (435e-12*VEL_LIGHT)  ## delay quantum of the MWA beamformer in meters.
MAX_POLS = 4
MWA_DIPOLE_HEIGHT = 0.29

R2C_SIGN = -1.0

def reorderDelays2RTS(metafits_delays):
    """The RTS analytic beam model uses to delays in the opposite direction,
    so this function reorders them"""
    rts_delays = np.empty(NUM_DIPOLES)
    for i in np.arange(4):
    	rts_delays[3-i] = metafits_delays[0+i*4]
    	rts_delays[7-i] = metafits_delays[1+i*4]
    	rts_delays[11-i] = metafits_delays[2+i*4]
    	rts_delays[15-i] = metafits_delays[3+i*4]

    return rts_delays

def RTS_analytic_beam(az, za, metafits_delays, freq, norm=True):
    """Calculates the MWA beam response using the RTS MWA analytic beam code.
    This is basically a direct translation from C code, I have preserved
    the comments within from the RTS itself

    Parameters
    ==========
    az : float
        Azimuth, where north = 0, and increases towards east
    za : float
        Zenith Angle
    metafits_delays : list/array
        Length 16 array of dipole delays as ordered in the metafits file
    freq : float
        Frequency to calculate beam at (Hz)
    norm : boolean
        If True, normalise the beam to zenith (default True)
    """

    rts_delays = reorderDelays2RTS(metafits_delays)

    wavelength = VEL_LIGHT / freq

    lat = MWA_LAT_RAD
    dpl_sep = 1.1
    dpl_hgt = MWA_DIPOLE_HEIGHT

    ##Hold the complex jones matrix
    response = np.zeros(4, dtype=complex)
    ha, dec = erfa.ae2hd(az, np.pi/2 - za, MWA_LAT_RAD)

    # // set elements of the look-dir vector
    proj_e = np.sin(za)*np.sin(az)
    proj_n = np.sin(za)*np.cos(az)
    proj_z = np.cos(za)

    n_cols = 4
    n_rows = 4

    multiplier = R2C_SIGN * 1j * 2 * np.pi / wavelength
    k = 0

    # /* loop over dipoles */
    for i in np.arange(n_cols):
        for j in np.arange(n_rows):

        # // set elements of the baseline vector
            dipl_e = (i - 1.5) * dpl_sep
            dipl_n = (j - 1.5) * dpl_sep
            dipl_z = 0.0

            PhaseShift = np.exp( multiplier * ( dipl_e*proj_e + dipl_n*proj_n + dipl_z*proj_z - rts_delays[k]*VEL_LIGHT ) )

            # // sum for p receptors

            k_gain  = k
            k_phase = k_gain + 2*NUM_DIPOLES
            tmp_response = complex(1.0, 0.0) * np.exp(0.0j) * PhaseShift
            response[0] += tmp_response
            response[1] += tmp_response

            # // sum for q receptors

            k_gain  += NUM_DIPOLES
            k_phase += NUM_DIPOLES
            tmp_response = complex(1.0, 0.0) * np.exp(0.0j) * PhaseShift
            response[2] += tmp_response
            response[3] += tmp_response

            k += 1

    ground_plane = 2.0*np.sin(2.0*np.pi*dpl_hgt/wavelength*np.cos(erfa.seps(0.0,lat,ha,dec)))

    if norm:
      ground_plane /= 2.0*np.sin(2.0*np.pi*dpl_hgt/wavelength)

    rot = np.empty(4)

    rot[0] =  np.cos(lat)*np.cos(dec) + np.sin(lat)*np.sin(dec)*np.cos(ha-0.0)
    rot[1] = -np.sin(lat)*np.sin(ha-0.0)
    rot[2] =  np.sin(dec)*np.sin(ha-0.0)
    rot[3] =  np.cos(ha-0.0)

    # // rot is the Jones matrix, response just contains the phases,
    # so this should be an element-wise multiplication.
    response[0] *= rot[0] * ground_plane / NUM_DIPOLES
    response[1] *= rot[1] * ground_plane / NUM_DIPOLES
    response[2] *= rot[2] * ground_plane / NUM_DIPOLES
    response[3] *= rot[3] * ground_plane / NUM_DIPOLES

    return response
