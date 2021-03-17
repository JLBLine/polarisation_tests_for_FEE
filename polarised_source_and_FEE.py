import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import erfa
from astropy.io import fits
from astropy.wcs import WCS
import mwa_hyperbeam
import os
from rts_analytic_beam import RTS_analytic_beam, RTS_analytic_beam_array
import matplotlib.animation as animation

##Some constants
D2R = np.pi / 180.0
VELC = 299792458.0
MWA_LAT = -26.7033194444
MWA_LAT_RAD = MWA_LAT * D2R

def get_pol_angle(rm, wavelength_metres, pol_ang_zero=0.0):
    """
    Takes a rotation measure `rm` and an array of wavelengths in metres
    `wavelength_metres`, and recovers the polarisation angle as a function
    of wavelength. Defaults to set the intrinsic polarisation angle `pol_ang_zero`
    to 0.0, but can be set optinally

    Parameters
    ==========
    rm : float
        Rotation measure (rad per metre squared)
    wavelength_metres : numpy array
        Wavelength values (metres)
    pol_ang_zero : float
        Intrinsic polarisation angle (defaults to zero)

    Returns
    =======
    pol_ang_zero + rm*wavelength_metres**2 : numpy array
        This is the calculated rotation measure per wavelength

    """
    return pol_ang_zero + rm*wavelength_metres**2

def extrap_stokes(freq_hz, ref_stokes_Jy, SI, ref_freq_hz=200e+6):
    """
    Given a reference frequency (ref_freq_hz) and flux denisty (ref_stokes_Jy),
    assumes a power law and used spectral index `SI` to extrapolate the
    flux density at `freq_hz`.

    Parameters
    ==========
    freq_hz : float or numpy array
        Frequency/ies to extrapolate to (Hz)
    ref_stokes_Jy : float
        Reference flux density to extrapolate from (Jy)
    SI : float
        Spectral index of source
    ref_freq_hz : float
        Reference frequency to extrapolate from (Hz). Defaults to 200e+6

    Returns
    =======
    stokes_Jy : float or numpy array
        The extrapolated flux density/ies
    """
    stokes_Jy = ref_stokes_Jy*(freq_hz / ref_freq_hz)**SI
    return stokes_Jy

def get_QU_complex(freq_hz, rm, ref_I_Jy, SI, frac_pol, pol_ang_zero=0.0, ref_freq_hz=200e+6):
    """
    Given input flux density and polarisation metrics for a source, return the
    Stokes I, Q, and U parameters at the requested frequencies in `freq_hz`

    Parameters
    ==========
    freq_hz : float or numpy array
        Frequency/ies to extrapolate to (Hz)
    rm : float
        Rotation measure (rad per metre squared)
    ref_stokes_Jy : float
        Reference flux density to extrapolate from (Jy)
    SI : float
        Spectral index of source
    frac_pol : float
        polarisation fraction
    pol_ang_zero : float
        Intrinsic polarisation angle (defaults to zero)
    ref_freq_hz : float
        Reference frequency to extrapolate from (Hz). Defaults to 200e+6

    Returns
    =======
    I : float / numpy array
        The Stokes I values for the given frequency/ies in `freq_hz`
    Q : float / numpy array
        The Stokes I values for the given frequency/ies in `freq_hz`
    U : float / numpy array
        The Stokes I values for the given frequency/ies in `freq_hz`
    """
    ##Convert freqs to wavelengths
    wavelength = VELC / freq_hz
    ##Calcualte polarisation angle
    pol_ang = get_pol_angle(rm, wavelength, pol_ang_zero=pol_ang_zero)

    ##Extrapolate Stokes I to given frequencies
    I = extrap_stokes(freq_hz, ref_I_Jy, SI, ref_freq_hz=ref_freq_hz)

    ##Calculate Stokes Q
    numer = frac_pol*I*np.exp(2j * pol_ang)
    denom = 1 + 1j*np.tan(2*pol_ang)
    Q = numer / denom

    ##Calculate Stokes U
    U = Q * np.tan(2*pol_ang)

    return I, Q, U

def add_colourbar(fig=None,ax=None,im=None,label=False,top=False):
    """Adds a colorbar to a given axes `ax` on a given figure `fig`
    which has has an imshow plot added `im`. Optionally add a
    `label` to the colorbar. Defaults to plotting colorbar on the
    right, optinally can plot on the top by setting `top`=True

    Parameters
    ==========
    fig : matplotlib.figure.Figure
        Figure the axes lives on
    ax : matplotlib.axes.Axes
        Axes the plot lives on
    im : matplotlib.image.AxesImage
        Output from running `imshow` on the Axes
    label : string
        Optional label for the colobar
    top : boolean
        If True add colorbar horizonally atop the Axes, rather than vertically
        and to the right

    Returns
    =======
    """

    divider = make_axes_locatable(ax)
    if top == True:
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax = cax,orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
    else:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax = cax)
    if label:
        cbar.set_label(label)

def apply_instrumental_to_stokes(jones, stokes):
    """
    Takes an instrumental gains `jones` complex array of multiple directions
    on the sky or frequencies (num_coords) in the shape (num_coords, 2, 2),
    and a Stokes paramaters complex array (I, Q, U, V) of shape (4,),
    and derives the resultant instrumental polarisations by applying the
    instrumental gains to the Stokes  parameters.

    Parameters
    ==========
    jones : complex numpy array
        Instrumental gains to use, shape = (num_coords, 2, 2)
    stokes : complex numpy array
        An array to represent the sky emission from all directions in Stokes
        parameters

    Returns
    =======
    inst_pols : complex numpy array
        The instrumental polarisations with shape = (num_coords, 2, 2), where:

        inst_pols[:,0,0] = XX
        inst_pols[:,0,1] = XY
        inst_pols[:,1,0] = YX
        inst_pols[:,1,1] = YY

        Here, XX is in the north-south direction, YY the east-west
    """

    sI,sQ,sU,sV = stokes

    g1xx = jones[:,0,0]
    g1xy = jones[:,0,1]
    g1yx = jones[:,1,0]
    g1yy = jones[:,1,1]

    g2xx_conj = np.conjugate(g1xx)
    g2xy_conj = np.conjugate(g1xy)
    g2yx_conj = np.conjugate(g1yx)
    g2yy_conj = np.conjugate(g1yy)

    XX = (g1xx*g2xx_conj + g1xy*g2xy_conj)*sI
    XX += (g1xx*g2xx_conj - g1xy*g2xy_conj)*sQ
    XX += (g1xx*g2xy_conj + g1xy*g2xx_conj)*sU
    XX += 1j*(g1xx*g2xy_conj - g1xy*g2xx_conj)*sV

    XY = (g1xx*g2yx_conj + g1xy*g2yy_conj)*sI
    XY += (g1xx*g2yx_conj - g1xy*g2yy_conj)*sQ
    XY += (g1xx*g2yy_conj + g1xy*g2yx_conj)*sU
    XY += 1j*(g1xx*g2yy_conj - g1xy*g2yx_conj)*sV

    YX = (g1yx*g2xx_conj + g1yy*g2xy_conj)*sI
    YX += (g1yx*g2xx_conj - g1yy*g2xy_conj)*sQ
    YX += (g1yx*g2xy_conj + g1yy*g2xx_conj)*sU
    YX += 1j*(g1yx*g2xy_conj - g1yy*g2xx_conj)*sV

    YY = (g1yx*g2yx_conj + g1yy*g2yy_conj)*sI
    YY += (g1yx*g2yx_conj - g1yy*g2yy_conj)*sQ
    YY += (g1yx*g2yy_conj + g1yy*g2yx_conj)*sU
    YY += 1j*(g1yx*g2yy_conj - g1yy*g2yx_conj)*sV

    inst_pols = np.empty(jones.shape,dtype=complex)
    inst_pols[:,0,0] = XX
    inst_pols[:,0,1] = XY
    inst_pols[:,1,0] = YX
    inst_pols[:,1,1] = YY

    return inst_pols

def convert_inst_back_to_stokes(inst_pols):
    """
    Takes an instrumental polarisations complex array `inst_pols` of multiple
    directions on the sky or frequencies (num_coords) in the shape
    (num_coords, 2, 2), and converts back into Stokes parameters as seen
    by the instrument. Returns the Stokes response `stokes_beam` as a
    complex array of shape (num_coords, 2, 2)

    Parameters
    ==========
    inst_pols : complex numpy array
        Instrumental polarisations to use, shape = (num_coords, 2, 2), where

        inst_pols[:,0,0] = XX
        inst_pols[:,0,1] = XY
        inst_pols[:,1,0] = YX
        inst_pols[:,1,1] = YY

        Here, XX is in the north-south direction, YY the east-west

    Returns
    =======
    stokes : complex numpy array
        The Stokes response on the instrument of shape = (num_coords, 2, 2), where:
        stokes[:,0,0] = I
        stokes[:,0,1] = Q
        stokes[:,1,0] = U
        stokes[:,1,1] = V
    """
    XX = inst_pols[:,0,0]
    XY = inst_pols[:,0,1]
    YX = inst_pols[:,1,0]
    YY = inst_pols[:,1,1]

    sI = 0.5*(XX + YY)
    sQ = 0.5*(XX - YY)
    sU = 0.5*(XY + YX)
    sV = -0.5j*(XY - YX)

    stokes = np.empty(inst_pols.shape,dtype=complex)
    stokes[:,0,0] = sI
    stokes[:,0,1] = sQ
    stokes[:,1,0] = sU
    stokes[:,1,1] = sV

    return stokes

def rotate_jones_para(ha_rad, dec_rad, jones, para_angle_offset=np.pi/2):
    """
    Takes an instrumental gains `jones` complex array of multiple frequencies
    or directions on the sky (num_coords) in the shape (num_coords, 2, 2),
    calculates the parallactic angle for the give hour angle and declination
    `ha_rad, dec_rad`), and rotates `jones` by the parallactic
    angle + `para_angle_offset` (which defaults to `np.pi/2`.)

    Parameters
    ==========
    ha_rad: float or numpy array
        Hour angle to calculate parallactic angle towards. If Jones has
        mulitple sky directions, len(ha_rad) = num_coords. If Jones has
        only has multiple frequencies for one sky direction, use a single
        value for `ha_rad`
    dec_rad: float or numpy array
        Hour angle to calculate parallactic angle towards. If Jones has
        mulitple sky directions, len(dec_rad) = num_coords. If Jones has
        only has multiple frequencies for one sky direction, use a single
        value for `dec_rad`
    jones : complex numpy array
        Instrumental gains to use, shape = (num_coords, 2, 2)
    stokes : complex numpy array
        An array to represent the sky emission from all directions in Stokes
        parameters

    Returns
    =======
    rot_jones : complex numpy array
        The rotated jones matrix shape = (num_coords, 2, 2):
    """

    prerot0 = jones[:,0,0]
    prerot1 = jones[:,0,1]
    prerot2 = jones[:,1,0]
    prerot3 = jones[:,1,1]

    para_angles = erfa.hd2pa(ha_rad, dec_rad, MWA_LAT_RAD)

    cosrot = np.cos(para_angles + para_angle_offset)
    sinrot = np.sin(para_angles + para_angle_offset)

    rot_jones = np.empty(jones.shape,dtype=complex)

    rot_jones[:,0,0] = prerot0*cosrot - prerot1*sinrot
    rot_jones[:,0,1] = prerot0*sinrot + prerot1*cosrot
    rot_jones[:,1,0] = prerot2*cosrot - prerot3*sinrot
    rot_jones[:,1,1] = prerot2*sinrot + prerot3*cosrot

    return rot_jones


def recover_stokes_rm_from_jones(jones_per_freq, stokes, wavelens_squared):
    """
    Recovers observed Stokes params and Faraday Depth Function (FDF) for a
    gives Jones matrix as a function of frequency `jones_per_freq` and
    sky Stokes parameters `stokes`.
    WARNING - function assumes that the frequency sampling of
    the `jones_per_freq` is evenly space in wavelength**2. If
    not, the recovered FDF will be incorrect

    Take number a number of beam jones matrices as a function
    of freq where:
    jones_per_freq.shape = (num_freqs, 2, 2)
    and a Stokes vector of stokes = [I,Q,U,V] where:
    len(I) = len(Q) = len(U) = len(V) = num_freqs
    This function constructs the Faraday Dispersion Function by
    applying the beam Jones to the input Stokes vectors
    to create instrumental XX,XY,YX,YY instrumental pols,
    and the uses the instrumental pols calculate observed Stokes.
    Uses the observed Stokes it then calulates P = Q +iU as
    a function of wavelength. Using this it performs the necessary fourier
    transforms to calculate the FDF.

    Parameters
    ==========
    jones_per_freq : complex numpy array
        Instrumental gains to use, shape = (num_freqs, 2, 2)
    stokes : complex numpy array of length 4
        An array to represent the sky emission Stokes parameters

    Returns
    =======
    recover_stokes_per_freq : complex numpy array
        The Stokes response as see by the instrument with
        shape = (num_coords, 2, 2), where:
        stokes[:,0,0] = I
        stokes[:,0,1] = Q
        stokes[:,1,0] = U
        stokes[:,1,1] = V
    faraday_depth : numpy array
        The faraday depth values of the returned faraday rotation function `fdf`
    fdf : complex numpy array
        The FDF as calculated through from the outputs in `recover_stokes_per_freq`
    """

    ##This applies the beam jones to the input Stokes parameters, to create
    ##instrumental XX,XY,YX,YY
    inst_pols_per_freq = apply_instrumental_to_stokes(jones_per_freq, stokes)

    ##Takes the generated XX, XY, YX, YY to recover the Stokes pols
    recover_stokes_per_freq = convert_inst_back_to_stokes(inst_pols_per_freq)
    recovered_Q = recover_stokes_per_freq[:,0,1]
    recovered_U = recover_stokes_per_freq[:,1,0]

    ##P = Q + iU
    P_per_lambda_sqr = recovered_Q + 1j*recovered_U

    ##FDF is the fourier transform of P
    fdf = np.fft.fftshift(np.fft.fft(P_per_lambda_sqr))

    ##Get resolution of the wavelengths
    wave_sqr_res = wavelens_squared[1] - wavelens_squared[0]
    ##numpy definition of ft frequencies is off by pi compared to FDF def
    faraday_depth = np.pi*np.fft.fftshift(np.fft.fftfreq(len(wavelens_squared),wave_sqr_res))

    return recover_stokes_per_freq, faraday_depth, fdf


def plot_abs_real_imag(fig, axs, plot_arrays, labels, vmins=False, vmaxs=False):
    """
    Given a figure instance `fig` and a 4 by 3 subplots axes instance `axs`,
    plot the absolute, real, and imaginary of 4 complex 2D arrays as
    given in `plot_arrays`. Label them using the 4 labels in `labels`.
    Optionally set the vmin/vmax values of the absolute plots by supplying
    `vmin` and `vmax`.

    Parameters
    ==========
    fig : matplotlib.figure.Figure
        Figure instance to plot on
    axs : 2D numpy array with shape (4,3) containing `matplotlib.axes._subplots.AxesSubplot`
        Three by four array of subplots instances to plot on
    plot_arrays : list/array of length 4, containg 2D complex numpy arrays
        Four 2D complex arrays to plot on `axs`
    labels : list/array of length 4
        Labels that match the arrays in `plot_arrays`
    vmins : list/array of length 4
        List of 4 vmin values to set vmin for the 4 imshow plots of the
        absolute values in `plot_arrays`
    vmaxs : list/array of length 4
        List of 4 vmax values to set vmax for the 4 imshow plots of the
        absolute values in `plot_arrays`

    Returns
    =======
    """
    ##Loop through input arrays and plot
    for ind, (plot_array, label) in enumerate(zip(plot_arrays, labels)):

        if vmins and vmaxs:
            im0 = axs[ind,0].imshow(np.abs(plot_array),vmin=vmins[ind],vmax=vmaxs[ind],origin='lower')
        else:
            im0 = axs[ind,0].imshow(np.abs(plot_array),origin='lower')


        im1 = axs[ind,1].imshow(np.real(plot_array),origin='lower')
        im2 = axs[ind,2].imshow(np.imag(plot_array),origin='lower')

        t_labels = ['abs ', 'real ', 'imag ']
        ims = [im0, im1, im2]

        ##Loop through abs, real, imag plots and label + add colourbar
        for ax, t_label, im in zip(axs[ind,:], t_labels, ims):
            ax.set_title(t_label + label, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            add_colourbar(ax=ax, im=im, fig=fig)

def reshape_to_sky(array_to_reshape, shape):
    """Takes an array of shape=(N,2,2) `array_to_reshape`, and produces four
    2D arrays of tuple `shape` by performing the transforms

    array_to_reshape[:,0,0].shape = shape
    array_to_reshape[:,0,1].shape = shape
    array_to_reshape[:,1,0].shape = shape
    array_to_reshape[:,1,1].shape = shape

    Parameters
    ==========
    array_to_reshape : 3D numpy array
        Array of shape=(N,2,2) to reshape
    shape : tuple
        The 2D shape to change the entries in `array_to_reshape` to

    Returns
    =======
    [arr0, arr1, arr2, arr3] : list of four 2D array
        A list of the reshaped arrays were
        arr0 = array_to_reshape[:,0,0]
        arr1 = array_to_reshape[:,0,1]
        arr2 = array_to_reshape[:,1,0]
        arr3 = array_to_reshape[:,1,1]
    """

    arr0 = array_to_reshape[:,0,0]
    arr1 = array_to_reshape[:,0,1]
    arr2 = array_to_reshape[:,1,0]
    arr3 = array_to_reshape[:,1,1]

    arr0.shape = shape
    arr1.shape = shape
    arr2.shape = shape
    arr3.shape = shape

    return [arr0, arr1, arr2, arr3]


def plot_jones_matrix(shape, jones, title, save_plot=False):
    """
    Takes an instrumental gains `jones` complex array of multiple directions
    on the sky (num_coords) in the shape (num_coords, 2, 2). The coords should
    have been flattened from a 2D array with shape described by the
    tuple `shape`, so that each polarisation can be modified into a 2D array
    e.g. jones[:,0,0].shape = `shape`. Function then plots the
    amplitude, real, and imaginary of all four 2D sky maps.

    The final plot will consist of:
    1st row: abs, real, imag of 2D mapping of data in jones[:,0,0]
    2nd row: abs, real, imag of 2D mapping of data in jones[:,0,1]
    3rd row: abs, real, imag of 2D mapping of data in jones[:,1,0]
    4th row: abs, real, imag of 2D mapping of data in jones[:,1,1]

    Outputs will be labelled appropriately.

    Parameters
    ==========
    shape : tuple
        2D dimensions to reshape the input Jones arrays for on sky plot
    jones : complex numpy array
        Instrumental gains to use, shape = (num_coords, 2, 2)
    title : string
        Adds a suptitle to the figure, and if save_plot=True, use the string
        in the filename
    save_plot : boolean
        If True, saves plot to a png. Else, uses plt.show()

    Returns
    =======
    """
    fig, axs = plt.subplots(4,3,figsize=(10,12))

    j_2Ds = reshape_to_sky(jones, shape)
    j_labels = ['j(0,0)', 'j(0,1)', 'j(1,0)', 'j(1,1)']

    plot_abs_real_imag(fig, axs, j_2Ds, j_labels)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.suptitle(title)

    if save_plot:
        fig.savefig("jones_{:s}.png".format(title))
        plt.close()
    else:
        plt.show()

def plot_stokes_beam(shape, stokes_beam, title, save_plot=False, vmins=False, vmaxs=False):
    """
    Takes an observed Stokes polarisations complex array of multiple directions
    on the sky (num_coords) in the shape (num_coords, 2, 2). The coords should
    have been flattened from a 2D array with shape described by the
    tuple `shape`, so that each polarisation can be modified into a 2D array
    e.g. stokes_beam[:,0,0].shape = `shape`. Function then plots the
    amplitude, real, and imaginary of all four 2D sky maps.
    Optionally set the vmin/vmax values of the absolute plots by supplying
    `vmin` and `vmax`.

    The final plot will consist of:
    1st row: abs, real, imag of 2D mapping of data in stokes_beam[:,0,0]
    2nd row: abs, real, imag of 2D mapping of data in stokes_beam[:,0,1]
    3rd row: abs, real, imag of 2D mapping of data in stokes_beam[:,1,0]
    4th row: abs, real, imag of 2D mapping of data in stokes_beam[:,1,1]

    Outputs will be labelled appropriately.

    Parameters
    ==========
    shape : tuple
        2D dimensions to reshape the input Jones arrays for on sky plot
    stokes_beam : complex numpy array
        Observed Stokes params to use, of shape = (num_coords, 2, 2)
    title : string
        Adds a suptitle to the figure, and if save_plot=True, use the string
        in the filename
    save_plot : boolean
        If True, saves plot to a png. Else, uses plt.show()
    vmins : list/array of length 4
        List of 4 vmin values to set vmin for the 4 imshow plots of the
        absolute values in `plot_arrays`
    vmaxs : list/array of length 4
        List of 4 vmax values to set vmax for the 4 imshow plots of the
        absolute values in `plot_arrays`

    Returns
    =======
    """
    fig, axs = plt.subplots(4,3,figsize=(10,12))

    stokes_beam_2Ds = reshape_to_sky(stokes_beam, shape)
    stokes_labels = ['Stokes I', 'Stokes Q', 'Stokes U', 'Stokes V']

    plot_abs_real_imag(fig, axs, stokes_beam_2Ds, stokes_labels, vmins=vmins, vmaxs=vmaxs)

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.suptitle(title, fontsize=16)

    if save_plot:
        fig.savefig("inst-stokes_{:s}.png".format(title))
        plt.close()
    else:
        plt.show()


def plot_inst_pols(shape, inst_pols, title, save_plot=False, vmins=False, vmaxs=False):
    """
    Takes an instrumental polarisation complex array of multiple directions
    on the sky (num_coords) in the shape (num_coords, 2, 2). The coords should
    have been flattened from a 2D array with shape described by the
    tuple `shape`, so that each polarisation can be modified into a 2D array
    e.g. inst_pols[:,0,0].shape = `shape`. Function then plots the
    amplitude, real, and imaginary of all four 2D sky maps.
    Optionally set the vmin/vmax values of the absolute plots by supplying
    `vmin` and `vmax`.

    The final plot will consist of:
    1st row: abs, real, imag of 2D mapping of data in inst_pols[:,0,0]
    2nd row: abs, real, imag of 2D mapping of data in inst_pols[:,0,1]
    3rd row: abs, real, imag of 2D mapping of data in inst_pols[:,1,0]
    4th row: abs, real, imag of 2D mapping of data in inst_pols[:,1,1]

    Outputs will be labelled appropriately.

    Parameters
    ==========
    shape : tuple
        2D dimensions to reshape the input Jones arrays for on sky plot
    inst_pols : complex numpy array
        Instrumental polarisations to use, shape = (num_coords, 2, 2)
    title : string
        Adds a suptitle to the figure, and if save_plot=True, use the string
        in the filename
    save_plot : boolean
        If True, saves plot to a png. Else, uses plt.show()
    vmins : list/array of length 4
        List of 4 vmin values to set vmin for the 4 imshow plots of the
        absolute values in `plot_arrays`
    vmaxs : list/array of length 4
        List of 4 vmax values to set vmax for the 4 imshow plots of the
        absolute values in `plot_arrays`

    Returns
    =======
    """
    fig, axs = plt.subplots(4,3,figsize=(10,12))

    inst_pol_2Ds = reshape_to_sky(inst_pols, shape)
    inst_labels = ['XX', 'XY', 'YX', 'YY']

    plot_abs_real_imag(fig, axs, inst_pol_2Ds, inst_labels, vmins=vmins, vmaxs=vmaxs)

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.suptitle(title, fontsize=16)

    if save_plot:
        fig.savefig("inst-pol_{:s}.png".format(title))
        plt.close()
    else:
        plt.show()

def remap_hbeam_jones(hbeam_jones):
    """Takes input hyperbeam jones array of shape=(num_coords, 4)
    and reshapes to shape=(num_coords, 2, 2)"""

    num_coords = hbeam_jones.shape[0]

    new_shape_jones = np.empty((num_coords,2,2),dtype=complex)
    new_shape_jones[:,0,0] = hbeam_jones[:,0]
    new_shape_jones[:,0,1] = hbeam_jones[:,1]
    new_shape_jones[:,1,0] = hbeam_jones[:,2]
    new_shape_jones[:,1,1] = hbeam_jones[:,3]

    return new_shape_jones

def get_sin_projected_coords(nside = 301):
    ##Give it 301 pixel for each axis

    header = fits.Header()

    lst_deg = 0.0

    ##This resolution seems to cover the full sky nicely
    cpix = int(nside // 2)
    cdelt = 0.25
    cdelt = 125 / nside

    header['NAXIS']   = 2
    header['NAXIS1']  = nside
    header['NAXIS2']  = nside
    header['CTYPE1']  = 'RA---SIN'
    header['CRPIX1']  = cpix
    header['CRVAL1']  = lst_deg
    header['CDELT1']  = cdelt
    header['CUNIT1']  = 'deg     '
    header['CTYPE2']  = 'DEC--SIN'
    header['CRPIX2']  = cpix
    header['CRVAL2']  = MWA_LAT
    header['CDELT2']  = cdelt
    header['CUNIT2']  = 'deg     '

    ##Make a world coord system
    wcs = WCS(header)

    ##Set up x/y pixels that cover the whole image
    x_mesh, y_mesh = np.meshgrid(np.arange(nside), np.arange(nside))

    x_pixels = x_mesh.flatten()
    y_pixels = y_mesh.flatten()

    ##convert to ra, dec
    ras, decs = wcs.all_pix2world(x_pixels, y_pixels, 0.0)

    ##Then use erfa to convert these values into azs, els
    has = lst_deg - ras

    ##use this erfa function to convert to azimuth and elevation
    ##were using degrees, but erfa uses rads, so convert here
    az_grid, els = erfa.hd2ae(has*D2R, decs*D2R, MWA_LAT_RAD);

    ##convert elevation to zenith angle
    za_grid = np.pi/2 - els

    za_grid.shape = (nside,nside)
    az_grid.shape = (nside,nside)

    ##Mask below horizon for better plots
    below_horizon = np.where(za_grid > np.pi/2)
    az_grid[below_horizon] = np.nan
    za_grid[below_horizon] = np.nan

    return wcs, az_grid.flatten(), za_grid.flatten(), has*D2R, decs*D2R

def get_jones_per_freq(az_rad, za_rad, ha_rad, dec_rad, freqs_hz, beam_model='unity', delays=[1]*16):
    if beam_model == 'unity':
        rot_jones_per_freq = np.zeros((len(freqs_hz),2,2),dtype=complex)
        rot_jones_per_freq[:,0,0] = complex(1,0)
        rot_jones_per_freq[:,1,1] = complex(1,0)

    elif beam_model=='hyper' or beam_model=='hyper_rev' or beam_model=='hyper_flip':
        ##Create the beam
        beam = mwa_hyperbeam.FEEBeam()

        ##Container for Jones as a function of frequency
        jones_per_freq = np.empty((len(freqs_hz),2,2),dtype=complex)

        ##For each frequency, calculate the Jones, and shove into jones_per_freq
        for ind, freq_hz in enumerate(freqs_hz):
            this_jones = beam.calc_jones(az_rad, za_rad, freq_hz, delays, [1]*16, True)
            this_jones.shape = (2,2)

            if beam_model=='hyper_rev':
                jones_per_freq[ind,0,0] = this_jones[1,1]
                jones_per_freq[ind,0,1] = this_jones[1,0]
                jones_per_freq[ind,1,0] = this_jones[0,1]
                jones_per_freq[ind,1,1] = this_jones[0,0]
            elif beam_model=='hyper_flip':
                jones_per_freq[ind,0,0] = -this_jones[1,1]
                jones_per_freq[ind,0,1] = this_jones[1,0]
                jones_per_freq[ind,1,0] = -this_jones[0,1]
                jones_per_freq[ind,1,1] = this_jones[0,0]
            else:
                jones_per_freq[ind,:,:] = this_jones

        rot_jones_per_freq = rotate_jones_para(ha_rad, dec_rad, jones_per_freq)

    elif beam_model=='rts':
        rot_jones_per_freq = np.empty((len(freqs_hz),2,2),dtype=complex)

        ##For each frequency, calculate the Jones, and shove into jones_per_freq
        for ind, freq_hz in enumerate(freqs_hz):
            ##Beam calculation, via RTS analytic beam code
            this_jones = RTS_analytic_beam(az_rad, za_rad,  delays, freq_hz, norm=True)

            this_jones.shape = (2,2)
            rot_jones_per_freq[ind,:,:] = this_jones

    return rot_jones_per_freq

def setup_rm_plot(fig, ha_rad, dec_rad, beam_model='unity',
                  low_freq=160e+6, high_freq=180e+6, num_samples=200,
                  frac_pol=0.24, rm=37.41, ref_I_Jy=7.075,
                  SI=-0.5, ref_V_Jy=0.2436, savefig=False,
                  delays = [0]*16, nside=301):
    ##sample evenly in wavelength squared
    wavelen_low = 3e+8 / high_freq
    wavelen_high = 3e+8 / low_freq
    wavelens_squared = np.linspace(wavelen_low**2, wavelen_high**2,num_samples)

    ##convert back to frequencies as that makes the most sense in my head
    freqs_hz = VELC / np.sqrt(wavelens_squared)

    ##This function applies Equations 9 and 10 above to derive the Q and U values
    Is_Jy, Qs_Jy, Us_Jy = get_QU_complex(freqs_hz, rm, ref_I_Jy, SI, frac_pol)

    ##This function just applies a spectral index to extrapolate
    ##flux over frequencies
    Vs_Jy = extrap_stokes(freqs_hz, ref_V_Jy, SI)

    stokes_source = np.array([Is_Jy, Qs_Jy, Us_Jy, Vs_Jy])

    az_rad, el_rad = erfa.hd2ae(ha_rad, dec_rad, MWA_LAT_RAD)
    ##Convert elevaltion into zenith angle
    za_rad = np.pi/2 - el_rad

    wcs, azs_flat, zas_flat, has_flat, decs_flat = get_sin_projected_coords(nside)

    freq_cent = (low_freq + high_freq) / 2

    if beam_model == 'unity':
        rot_jones_on_sky = np.zeros((len(azs_flat),2,2),dtype=complex)
        rot_jones_on_sky[:,0,0] = complex(1,0)
        rot_jones_on_sky[:,1,1] = complex(1,0)

    elif beam_model=='hyper' or beam_model=='hyper_rev' or beam_model=='hyper_flip':
        ##Create the beam
        beam = mwa_hyperbeam.FEEBeam()
        jones_on_sky = beam.calc_jones_array(azs_flat, zas_flat, freq_cent, delays, [1]*16, True)
        ##I've defined all my plotting stuff for 2x2 jones matrices (long story) so reshape
        jones_on_sky = remap_hbeam_jones(jones_on_sky)

        rot_jones_on_sky = np.empty((len(azs_flat),2,2),dtype=complex)
        if beam_model=='hyper_rev':
            rot_jones_on_sky[:,0,0] = jones_on_sky[:,1,1]
            rot_jones_on_sky[:,0,1] = jones_on_sky[:,1,0]
            rot_jones_on_sky[:,1,0] = jones_on_sky[:,0,1]
            rot_jones_on_sky[:,1,1] = jones_on_sky[:,0,0]
        elif beam_model=='hyper_flip':
            rot_jones_on_sky[:,0,0] = -jones_on_sky[:,1,1]
            rot_jones_on_sky[:,0,1] = jones_on_sky[:,1,0]
            rot_jones_on_sky[:,1,0] = -jones_on_sky[:,0,1]
            rot_jones_on_sky[:,1,1] = jones_on_sky[:,0,0]
        else:
            rot_jones_on_sky = jones_on_sky

        rot_jones_on_sky = rotate_jones_para(has_flat, decs_flat, rot_jones_on_sky)

    elif beam_model=='rts':
        rot_jones_on_sky = RTS_analytic_beam_array(azs_flat, zas_flat, delays, freq_cent, norm=True)
        rot_jones_on_sky = remap_hbeam_jones(rot_jones_on_sky)


    rot_jones_per_freq = get_jones_per_freq(az_rad, za_rad, ha_rad, dec_rad, freqs_hz, beam_model=beam_model, delays=delays)

    ##Apply Jones to stokes params and get the FDF
    stokes_per_freq, faraday_depth, fdf = recover_stokes_rm_from_jones(rot_jones_per_freq,
                                   stokes_source, wavelens_squared)

    stokesI = [complex(1,0),complex(0,0),complex(0,0),complex(0,0)]
    inst_pols_sky = apply_instrumental_to_stokes(rot_jones_on_sky, stokesI)
    recover_stokes_sky = convert_inst_back_to_stokes(inst_pols_sky)

    recover_stokes_skyI = np.real(recover_stokes_sky[:,0,0])
    recover_stokes_skyI.shape = (nside, nside)

    width = 0.21
    ytop = 0.6
    x0 = 0.07
    y0 = 0.1
    x1 = x0 + 0.31
    x2 = x0 + 0.62
    height = 0.35

    ax1 = fig.add_axes([x0, ytop, width, height])
    ax2 = fig.add_axes([x1, ytop, width, height])
    ax3 = fig.add_axes([x2, ytop, width, height])
    ax4 = fig.add_axes([x0, y0, width, height])
    ax5 = fig.add_axes([x1, y0, width, height])
    ax6 = fig.add_axes([x2, y0, width, height])

    ln1, = ax1.plot(freqs_hz/1e6, np.abs(rot_jones_per_freq[:,0,0]))
    ln4, = ax4.plot(freqs_hz/1e6, np.abs(rot_jones_per_freq[:,1,1]))

    ax1.set_ylabel('Gain')
    ax4.set_ylabel('Gain')

    ln2, = ax2.plot(freqs_hz/1e6, np.real(stokes_per_freq[:,0,1]),)
    ln5, = ax5.plot(freqs_hz/1e6, np.real(stokes_per_freq[:,1,0]))

    ax2.set_ylabel('Flux density (Jy)')
    ax5.set_ylabel('Flux density (Jy)')

    for ax in [ax1,ax2,ax4,ax5]:
        ax.set_xlabel('Freq (Hz)')

    ln3, = ax3.plot(faraday_depth, abs(fdf), 'C0o-' ,mfc='none')
    ax3.set_ylabel('abs( $F(\phi) $)')
    ax3.set_xlabel('$\phi ( \mathrm{rad} \, m^{-2}$)')
    ax3.axvline(rm,color='k',label='Expected RM',alpha=0.5, linestyle='--')
    ax3.set_xlim(-100,100)
    ax3.legend(loc='upper left')

    im = ax6.imshow(np.log10(recover_stokes_skyI))

    source_x, source_y = wcs.all_world2pix(ha_rad/D2R, dec_rad/D2R, 0)
    circ6, = ax6.plot(source_x, source_y, 'C1o',mfc='none', mew=2.0,label='Source location')

    add_colourbar(ax=ax6,im=im,fig=fig,label='log10[Normalised Beam Stokes I]')
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_xlabel('HA')
    ax6.set_ylabel('Dec')

    title = ax6.text(0.5,0.9, "az,za = {:4.1f},{:4.1f}".format(az_rad/D2R, za_rad/D2R), bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax6.transAxes, ha="center")

    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    iterables = [ln1, ln2, ln3, ln4, ln5, circ6, title]

    titles = ['Amp $g_x$', 'Recover Stokes Q', 'RM synthesis',
                  'Amp $g_y$', 'Recover Stokes U', 'Beam at {:.3f} MHz'.format(freq_cent/1e+6)]
    #
    for ax, title in zip(axs, titles):
        ax.set_title(title)

    return fig, axs, iterables

def plot_rm_sythnesis(ha_rad, dec_rad, beam_model='unity',
                      low_freq=160e+6, high_freq=180e+6, num_samples=200,
                      frac_pol=0.24, rm=37.41, ref_I_Jy=7.075,
                      SI=-0.5, ref_V_Jy=0.2436, savefig=False,
                      delays = [0]*16, nside=301):

    fig = plt.figure(figsize=(12,7))
    fig, axs, iterables = setup_rm_plot(fig, ha_rad, dec_rad, beam_model=beam_model,
                      low_freq=low_freq, high_freq=high_freq, num_samples=num_samples,
                      frac_pol=frac_pol, rm=rm, ref_I_Jy=ref_I_Jy,
                      SI=SI, ref_V_Jy=ref_V_Jy, savefig=savefig,
                      delays=delays, nside=nside)

    freq_cent = (low_freq + high_freq) / 2

    if savefig:
        if not os.path.isdir("./rm_beam_plots"):
            print('Making directory "./rm_beam_plots"')
            os.mkdir("./rm_beam_plots")
        else:
            pass
        fig.savefig("./rm_beam_plots/{:s}-beam_rm-recover_{:07.1f}MHz-ha{:04.1f}dec{:02.1f}.png".format(beam_model,freq_cent/1e+6,ha_rad/D2R,dec_rad/D2R),bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return "./rm_beam_plots/{:s}-beam_rm-recover_{:07.1f}MHz-ha{:04.1f}dec{:02.1f}.png".format(beam_model,freq_cent/1e+6,ha_rad/D2R,dec_rad/D2R)

def rm_sythnesis_movie(has_rad, decs_rad, beam_model='unity', save=True,
                      low_freq=160e+6, high_freq=180e+6, num_samples=200,
                      frac_pol=0.24, rm=37.41, ref_I_Jy=7.075,
                      SI=-0.5, ref_V_Jy=0.2436, savefig=False,
                      delays = [0]*16, nside=301):
    # ##sample evenly in wavelength squared
    wavelen_low = 3e+8 / high_freq
    wavelen_high = 3e+8 / low_freq
    wavelens_squared = np.linspace(wavelen_low**2, wavelen_high**2,num_samples)

    ##convert back to frequencies as that makes the most sense in my head
    freqs_hz = VELC / np.sqrt(wavelens_squared)

    ##This function applies Equations 9 and 10 above to derive the Q and U values
    Is_Jy, Qs_Jy, Us_Jy = get_QU_complex(freqs_hz, rm, ref_I_Jy, SI, frac_pol)

    ##This function just applies a spectral index to extrapolate
    ##flux over frequencies
    Vs_Jy = extrap_stokes(freqs_hz, ref_V_Jy, SI)

    stokes_source = np.array([Is_Jy, Qs_Jy, Us_Jy, Vs_Jy])

    azs_rad, els_rad = erfa.hd2ae(has_rad, decs_rad, MWA_LAT_RAD)
    ##Convert elevaltion into zenith angle
    zas_rad = np.pi/2 - els_rad

    fig = plt.figure(figsize=(12,7))
    fig, axs, iterables = setup_rm_plot(fig, has_rad[0], decs_rad[0], beam_model=beam_model,
                      low_freq=low_freq, high_freq=high_freq, num_samples=num_samples,
                      frac_pol=frac_pol, rm=rm, ref_I_Jy=ref_I_Jy,
                      SI=SI, ref_V_Jy=ref_V_Jy, savefig=savefig,
                      delays=delays, nside=nside)

    wcs, azs_flat, zas_flat, has_flat, decs_flat = get_sin_projected_coords(nside)
    ln1, ln2, ln3, ln4, ln5, circ6, title = iterables
    ax1, ax2, ax3, ax4, ax5, ax6 = axs

    def update(frame):

        rot_jones_per_freq = get_jones_per_freq(azs_rad[frame], zas_rad[frame],
                             has_rad[frame], decs_rad[frame],
                             freqs_hz, beam_model=beam_model, delays=delays)

        ##Apply Jones to stokes params and get the FDF
        stokes_per_freq, faraday_depth, fdf = recover_stokes_rm_from_jones(rot_jones_per_freq,
                                       stokes_source, wavelens_squared)

        stokesQ = np.real(stokes_per_freq[:,0,1])
        stokesU = np.real(stokes_per_freq[:,1,0])

        gain1 = np.abs(rot_jones_per_freq[:,0,0])
        gain2 = np.abs(rot_jones_per_freq[:,1,1])

        ln2.set_data(freqs_hz/1e6, stokesQ)
        ln5.set_data(freqs_hz/1e6, stokesU)

        ln1.set_data(freqs_hz/1e6, gain1)
        ln4.set_data(freqs_hz/1e6, gain2)

        ln3.set_data(faraday_depth, abs(fdf))

        for ax, data in zip([ax1, ax2, ax3, ax4, ax5], [gain1, stokesQ, abs(fdf), gain2, stokesU]):
            offset = (data.max() - data.min()) / 10.0
            ax.set_ylim(data.min()-offset, data.max()+offset)


        source_x, source_y = wcs.all_world2pix(has_rad[frame]/D2R, decs_rad[frame]/D2R, 0)
        circ6.set_data(source_x, source_y)

        title.set_text("az,za = {:4.1f},{:4.1f}".format(azs_rad[frame]/D2R, zas_rad[frame]/D2R))

        return iterables

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0,len(has_rad)),
                     blit=True) #

    freq_cent = (low_freq + high_freq) / 2
    if save:
        ani.save('{:s}_rm-on-sky_{:7.3f}MHz.mp4'.format(beam_model,freq_cent/1e+6),
                  writer='ffmpeg', fps=1)

    return ani
