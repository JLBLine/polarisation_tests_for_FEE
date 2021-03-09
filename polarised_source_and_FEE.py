import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import erfa

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


def recover_stokes_rm_from_jones(jones_per_freq, stokes):
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
