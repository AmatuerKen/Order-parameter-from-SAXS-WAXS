import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
import gc

'''
def create_qtheta_mask_full_angle(Q, angle, qmin, qmax, Nq, Ntheta):
    """
    Create a mask array labeling each pixel in Q/angle space by its bin index,
    and a lookup table mapping bin index → (q_center, theta_center).

    Returns:
        mask       : 2D array of bin indices (NaN for out-of-bounds)
        list_q     : 1D array of q-bin centers
        list_theta : 1D array of theta-bin centers
        bin_lookup : 2D array of shape (Nq*Ntheta, 2)
        mesh_q     : 2D array of q-bin centers
        mesh_theta : 2D array of theta-bin centers
    """
    # --- Q bins ---
    q_edges = np.linspace(qmin, qmax, Nq + 1)
    list_q = 0.5 * (q_edges[:-1] + q_edges[1:])  # bin centers

    # --- Theta bins from -π to 0 ---
    theta_edges = np.linspace(-np.pi, np.pi, Ntheta + 1)
    list_theta = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    # Digitize Q and theta
    q_indices = np.digitize(Q, q_edges) - 1
    theta_indices = np.digitize(angle, theta_edges) - 1

    # Initialize mask
    mask = np.full(Q.shape, np.nan)

    # Valid bin locations
    valid = (q_indices >= 0) & (q_indices < Nq) & (theta_indices >= 0) & (theta_indices < Ntheta)
    bin_index = theta_indices * Nq + q_indices
    mask[valid] = bin_index[valid]

    # Create lookup table and mesh
    bin_lookup = np.array([[q, t] for t in list_theta for q in list_q])  # (Nq*Ntheta, 2)
    mesh_q, mesh_theta = np.meshgrid(list_q, list_theta, indexing='ij')

    return mask, list_q, list_theta, bin_lookup, mesh_q, mesh_theta
'''


def create_qtheta_mask_smectic(Q, angle, qmin, qmax, Nq, Ntheta, initial_angle = 60/180*np.pi):
    """
    Create a mask array labeling each pixel in Q/angle space by its bin index,
    and a lookup table mapping bin index → (q_center, theta_center).

    Returns:
        mask       : 2D array of bin indices (NaN for out-of-bounds)
        list_q     : 1D array of q-bin centers
        list_theta : 1D array of theta-bin centers
        bin_lookup : 2D array of shape (Nq*Ntheta, 2)
        mesh_q     : 2D array of q-bin centers
        mesh_theta : 2D array of theta-bin centers
    """

    angle = np.where(angle < 0, angle + 2*np.pi, angle)

    # --- Q bins ---
    q_edges = np.linspace(qmin, qmax, Nq + 1)
    list_q = 0.5 * (q_edges[:-1] + q_edges[1:])  # bin centers

    #initial_angle = 60
    # --- Theta bins from -π to 0 ---
    theta_edges = np.linspace(initial_angle, initial_angle + np.pi, Ntheta + 1)
    list_theta = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    # Digitize Q and theta
    q_indices = np.digitize(Q, q_edges) - 1
    theta_indices = np.digitize(angle, theta_edges) - 1

    # Initialize mask
    mask = np.full(Q.shape, np.nan)

    # Valid bin locations
    valid = (q_indices >= 0) & (q_indices < Nq) & (theta_indices >= 0) & (theta_indices < Ntheta)
    bin_index = theta_indices * Nq + q_indices
    mask[valid] = bin_index[valid]

    # Create lookup table and mesh
    bin_lookup = np.array([[q, t] for t in list_theta for q in list_q])  # (Nq*Ntheta, 2)
    mesh_q, mesh_theta = np.meshgrid(list_q, list_theta, indexing='ij')

    return mask, list_q, list_theta, bin_lookup, mesh_q, mesh_theta

'''
def show_overlay_mask_full_angle(image, params, Q, angle,
                        qmin, qmax, Nq, Ntheta,
                        vmin=None, vmax=None,
                        q_color='red', theta_color='blue'):
    """
    Overlay q and theta bin boundaries on the image using contours.
    """
    # Compute bin edges
    q_edges = np.linspace(qmin, qmax, Nq + 1)
    
    list_theta = np.linspace(-np.pi, np.pi, Ntheta+1)
    list_theta = list_theta[1:]
    binsize_theta = np.pi/Ntheta
    theta_edges = np.linspace(list_theta[0] - binsize_theta / 2, list_theta[-1] + binsize_theta / 2, Ntheta + 1)

    # Set contrast
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(image, [1, 99])


    xc = int(params["Center_1"])
    yc = int(params["Center_2"])
    interval = 50
    # Plot image
    plt.figure(figsize=(8, 6))
    plt.imshow(image[yc - interval:yc + interval, xc - interval:xc + interval], cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    
    
    # Overlay q-bin contours
    for q in q_edges:
        plt.contour(Q[yc - interval:yc + interval, xc - interval:xc + interval], levels=[q], colors=q_color, linewidths=0.5, linestyles='dotted')

    # Overlay theta-bin contours
    for theta in theta_edges:
        plt.contour(angle[yc - interval:yc + interval, xc - interval:xc + interval], levels=[theta], colors=theta_color, linewidths=0.5, linestyles='dotted')

    plt.title("Q–θ Bin Boundaries Overlay")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.show()
'''


def remove_bad_or_beamstop_from_mask(mask, image, threshold=1e4, lower_threshold = 3):
    mask = remove_bad_pixels_from_mask(mask, image, threshold = threshold)
    
    #lower_threshold = 3
    beamstop = (image < lower_threshold)
    mask[beamstop] = np.nan
    return mask



def show_overlay_mask_smectic(image, Q, angle,
                        qmin, qmax, Nq, Ntheta,
                        vmin=None, vmax=None, initial_angle = 60/180*np.pi,
                        q_color='red', theta_color='blue'):
    """
    Overlay q and theta bin boundaries on the image using contours.
    """

    angle = np.where(angle < 0, angle + 2*np.pi, angle)
    # Compute bin edges
    q_edges = np.linspace(qmin, qmax, Nq + 1)
    
    list_theta = np.linspace(initial_angle, initial_angle + np.pi, Ntheta+1)
    list_theta = list_theta[1:]
    binsize_theta = np.pi/Ntheta
    theta_edges = np.linspace(list_theta[0] - binsize_theta / 2, list_theta[-1] + binsize_theta / 2, Ntheta + 1)

    # Set contrast
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(image, [1, 99])

    # Plot image
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)

    # Overlay q-bin contours
    for q in q_edges:
        plt.contour(Q, levels=[q], colors=q_color, linewidths=0.5, linestyles='dotted')

    # Overlay theta-bin contours
    for theta in theta_edges:
        plt.contour(angle, levels=[theta], colors=theta_color, linewidths=0.5, linestyles='dotted')

    plt.title("Q–θ Bin Boundaries Overlay")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.xlim([450, 600])
    plt.ylim([625, 750])
    plt.tight_layout()
    plt.show()

def create_smectic_mask(directory, mask_filename, image, Q, angle, qmin, 
                                                  qmax, Nq, Ntheta, 
                                                  initial_angle, vmin, vmax):
    
    mask, list_q, list_theta, bin_lookup, mesh_q, mesh_theta = create_qtheta_mask_smectic(Q, angle, qmin, 
                                                                                          qmax, Nq, Ntheta, 
                                                                                          initial_angle = initial_angle)
    show_overlay_mask_smectic(image, Q, angle,
                            qmin, qmax, Nq, Ntheta,
                            vmin=vmin, vmax=vmax, initial_angle = initial_angle)
    
    mask = remove_bad_or_beamstop_from_mask(mask, image, threshold=1e4, lower_threshold = 3)
    
    save_qtheta_mask(directory + mask_filename, mask, list_q, list_theta, bin_lookup, mesh_q, mesh_theta, qmin, qmax)
    



def lorentz(q, I0, q0, gamma):
    return I0 * gamma**2 / ((q - q0)**2 + gamma**2)

    

def find_smectic_q_peak(I_ave, list_q, flag_plot = False):
    # Step 1: Find max ignoring NaNs
    max_value = np.nanmax(I_ave)  # ignore NaNs
    max_idx = np.unravel_index(np.nanargmax(I_ave), I_ave.shape)
    theta_idx = max_idx[1]
    #print(f"Max value: {max_value}, at indices (q_idx, theta_idx): {max_idx}")
    
    # Step 2: Fit the column ignoring NaNs
    q_values = list_q
    I_column = I_ave[:, theta_idx]
    
    # Mask out NaNs
    NaNmask = ~np.isnan(I_column)
    q_valid = q_values[NaNmask]
    I_valid = I_column[NaNmask]
    
    # Initial guess
    I0_guess = np.max(I_valid)
    q0_guess = q_valid[np.argmax(I_valid)]
    gamma_guess = (q_valid[-1] - q_valid[0]) / 10
    p0 = [I0_guess, q0_guess, gamma_guess]
    
    # Fit Lorentzian
    popt, pcov = curve_fit(lorentz, q_valid, I_valid, p0=p0)
    I0_fit, q0_fit, gamma_fit = popt
    #print(f"Fitted Lorentzian parameters: I0={I0_fit}, q0={q0_fit}, gamma={gamma_fit}")

    if flag_plot:
        # Optional: generate fitted curve
        q_fit = np.linspace(q_valid[0], q_valid[-1], 200)
        I_fit = lorentz(q_fit, *popt)
    
        # Step 4: Generate fitted curve
        q_fit = np.linspace(q_valid[0], q_valid[-1], 500)
        I_fit = lorentz(q_fit, *popt)
        
        # Step 5: Plot data and fit
        plt.figure(figsize=(4, 3))
        plt.plot(q_valid, I_valid, 'bo', label='Data (theta column)')
        plt.plot(q_fit, I_fit, 'r-', label='Lorentzian fit')
        plt.xlabel('q')
        plt.ylabel('I')
        plt.title(f'Lorentzian Fit for theta index {theta_idx}')
        #plt.legend(loc = 'upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return q0_fit


def fit_parabola_peak_smectic(theta, intensity, initial_angle, num_points=5):
    """
    Fit a parabola to a peak in angular profile I(theta), accounting for periodicity.

    Parameters:
        theta     : 1D array of angles in radians, range [-π, 0]
        intensity : 1D array of corresponding intensity values
        num_points: number of points to use around peak for fitting (odd number recommended)

    Returns:
        theta_peak : angle at which the fitted parabola reaches maximum (radians, wrapped to [-π, 0])
        I_peak     : maximum intensity of fitted parabola
    """
    assert len(theta) == len(intensity)
    N = len(theta)

    # Step 1: extend angle and intensity to handle wrap-around
    theta_extended = np.concatenate((theta - np.pi, theta, theta + np.pi))
    intensity_extended = np.concatenate((intensity, intensity, intensity))
    #plt.figure
    #plt.plot(theta_extended, intensity_extended)
    # Step 2: find coarse max in extended region
    max_idx = np.argmax(intensity_extended)
    
    half_window = num_points // 2
    if max_idx < half_window:
        max_idx = max_idx + N
    fit_indices = np.arange(max_idx - half_window, max_idx + half_window + 1)

    # Step 3: extract local region
    t_fit = theta_extended[fit_indices]
    I_fit = intensity_extended[fit_indices]

    # Step 4: Fit parabola I(θ) = aθ² + bθ + c
    coeffs = np.polyfit(t_fit, I_fit, 2)  # returns [a, b, c]
    a, b, c = coeffs

    if a == 0:
        raise ValueError("Fitted parabola is degenerate (a=0).")

    # Step 5: Vertex of parabola
    theta_peak = -b / (2 * a)
    I_peak = np.polyval(coeffs, theta_peak)
    #print(theta_peak)
    # Wrap θ to [0,  2*pi]
    theta_peak_wrapped = (theta_peak + 2 * np.pi) % (2 * np.pi)
    if theta_peak_wrapped < initial_angle:
        theta_peak_wrapped = theta_peak_wrapped + np.pi
    elif theta_peak_wrapped > (initial_angle + np.pi):
        theta_peak_wrapped = theta_peak_wrapped - np.pi
    return theta_peak_wrapped, I_peak


def find_smectic_peak(I_ave, list_q, list_theta, initial_angle, flag_plot):

    q_peak = find_smectic_q_peak(I_ave, list_q, flag_plot = False)
    q_idx_closest = np.argmin(np.abs(list_q - q_peak))
    
    intensity = I_ave[q_idx_closest, :]
    theta_peak_horizontal_zero, I_peak = fit_parabola_peak_smectic(list_theta, intensity, initial_angle, num_points=5)
    #print(theta_peak_horizontal_zero)
    #plt.figure
    #plt.plot(list_theta, intensity)
    
    theta_peak = ( theta_peak_horizontal_zero + 2 * np.pi)% (2 * np.pi)
    #print(theta_peak)
    theta_idx_closest = np.argmin(np.abs(list_theta - theta_peak))

    return q_peak, q_idx_closest, theta_peak_horizontal_zero, theta_peak, theta_idx_closest


def shift_and_average_over_theta(theta_range, intensity_profile, theta_max):
    """
    Translate theta such that the maximum is at 0, 
    keep only [0, π/2] after shift,
    and average intensity at ±theta.
    The returned angle is the average of |+θ| and |-θ|.

    Parameters:
        theta_range        : 1D array of original theta bin centers (radians)
        intensity_profile  : 1D array of intensity values (same shape)
        theta_max          : scalar (radians), location of max found earlier

    Returns:
        theta_filtered     : 1D array of averaged angles in [0, π/2]
        intensity_filtered : 1D array of averaged intensity values
    """
    # Step 1: shift so that theta_max → 0
    theta_shifted = theta_range - theta_max

    # duplicate and wrap into [-π, π]
    theta_shifted = np.concatenate((theta_shifted, theta_shifted + np.pi))
    intensity_profile = np.concatenate((intensity_profile, intensity_profile))
    theta_shifted = (theta_shifted + np.pi) % (2 * np.pi) - np.pi

    # Step 2: select only [0, π/2] (positive thetas)
    valid = (theta_shifted >= 0) & (theta_shifted <= np.pi / 2)
    theta_pos = theta_shifted[valid]
    I_pos = intensity_profile[valid]

    # Step 3: average intensity at ±θ
    I_avg = np.zeros_like(I_pos)
    theta_avg = np.zeros_like(theta_pos)
    for i, th in enumerate(theta_pos):
        # find closest index for -theta
        j = np.argmin(np.abs(theta_shifted - (-th)))
        I_neg = intensity_profile[j]

        # average intensity
        I_avg[i] = 0.5 * (I_pos[i] + I_neg)

        # angle is average of absolute values of ±theta
        theta_avg[i] = 0.5 * (abs(th) + abs(-th))

    return theta_avg, I_avg



def compute_baseline_intensity_smectic(Q, angle, image, theta_max, initial_angle, qmin, qmax, pixels=5, angle_width_deg=2.5, flag_plot = True):
    """
    Compute average baseline intensity from two background regions defined in q–θ space.

    Parameters:
        Q         : 2D array of q values (Å⁻¹)
        angle     : 2D array of angles (radians)
        image     : 2D array of intensity values
        theta_max : float, location of peak in radians
        list_q    : 1D array of q-bin centers
        q_pixels  : how many pixels near qmin/qmax to use
        angle_width_deg : angular half-width in degrees (± this range)

    Returns:
        baseline : float, average intensity from both background regions
    """
    angle = np.where(angle < 0, angle + 2*np.pi, angle)
    
    # Convert 90° to radians
    theta_plus = theta_max + np.radians(90)
    theta_minus = theta_max - np.radians(90)

    # Choose the one within [-π, π] and between -180° and 30°
    candidates = [theta_plus, theta_minus]
    theta_bg = None
    for t in candidates:
        t_wrapped = ((t + np.pi) % (2 * np.pi)) - np.pi
        deg = np.degrees(t_wrapped)
        if initial_angle <= deg <= (initial_angle + 180):
            theta_bg = t_wrapped
            break
    if theta_bg is None:
        raise ValueError("No valid background angle found in the desired range.")

    # Angular mask
    dtheta = np.radians(angle_width_deg)
    angle_shifted = (angle - theta_bg + np.pi) % (2 * np.pi) - np.pi
    angle_mask = np.abs(angle_shifted) <= dtheta

    qc = (qmin + qmax)/2
    dq_perpixel = gradient_near_qc(Q, qc, delta_q=0.05)
    # Q mask
    q_lo_mask = (Q >= qmin) & (Q <= qmin + pixels * dq_perpixel)
    q_hi_mask = (Q >= qmax - pixels * dq_perpixel) & (Q <= qmax)

    # Combine masks
    region_mask = angle_mask & (q_lo_mask | q_hi_mask)

    if flag_plot:
        plt.figure(figsize = (6,6))
        plt.imshow(region_mask)
        plt.xlim([450, 600])
        plt.ylim([625, 750])
        plt.show()
        plt.close()
        
        
    # Compute average
    if np.any(region_mask):
        baseline = np.nanmean(image[region_mask])
    else:
        raise ValueError("No pixels found in baseline region.")

    return baseline

'''
def get_inner_ring_profile(directory, filename, mask_filename, Q, angle, initial_angle, flag_plot_qpeak = False, flag_plot_baseline = False, flag_plot_profile = False):

    image = read_esrf_edf_image(directory + filename)
    
    mask, list_q, list_theta, mesh_q, mesh_theta, bin_lookup, qmin, qmax = load_qtheta_mask(directory + mask_filename)
    Nq = np.size(list_q)
    Ntheta = np.size(list_theta)
    
    I_ave = calculate_average_intensity_within_mask(image, mask, mesh_q, mesh_theta)
    
    q_peak, q_idx_closest, theta_peak_horizontal_zero, theta_peak, theta_idx_closest = find_smectic_peak(I_ave, list_q, list_theta, flag_plot = flag_plot_qpeak)

    Iq = np.nanmean(I_ave[:, theta_idx_closest-1:theta_idx_closest+1], axis = 1)
    Itheta = np.nanmean(I_ave[q_idx_closest-1:q_idx_closest+1, :], axis = 0)
    shifted_theta, shifted_Itheta = shift_and_average_over_theta(list_theta, Itheta, theta_peak)
    baseline = compute_baseline_intensity_smectic(Q, angle, image, theta_peak, initial_angle, qmin, qmax, pixels=5, angle_width_deg=2.5, flag_plot = flag_plot_baseline)


    if flag_plot_profile:
        fig, axes = plt.subplots(2,1, figsize=(4,6))
        
        # Subplot 1: I(q)
        axes[0].scatter(list_q, Iq - baseline, s=15, color='b')
        axes[0].set_xlabel("q")
        axes[0].set_ylabel("I(q) - baseline")
        axes[0].set_title("Intensity vs q")
        
        # Subplot 2: I(theta)
        axes[1].scatter(shifted_theta, shifted_Itheta - baseline, s=15, color='r')
        axes[1].set_xlabel("θ (radians)")
        axes[1].set_ylabel("I(θ) - baseline")
        axes[1].set_title("Intensity vs θ (shifted)")
        #axes[1].set_yscale("log")
        plt.tight_layout()
        plt.show()
        
    return q_peak, theta_peak_horizontal_zero, Iq, Itheta, shifted_theta, shifted_Itheta, baseline, I_ave
'''


def Ipeak_all_pixel(image, mask, Q, angle, qmin, qmax, theta_peak, theta_bin):
    # Condition on q
    in_qbin = (Q >= qmin) & (Q < qmax)

    # Condition on theta (within ±theta_bin around theta_peak)
    in_thetabin = (angle >= theta_peak - theta_bin) & (angle < theta_peak + theta_bin)

    # Apply mask + conditions
    valid_pixels = ~np.isnan(mask) & in_qbin & in_thetabin

    # Extract values
    Iq_pixel = image[valid_pixels]
    qvals = Q[valid_pixels]

    return Iq_pixel, qvals

def lorentz_para(q, I0, q0, xi):
    return I0 / ((q - q0)**2 * xi**2 + 1)

def lorentz_para_mix(q, I0, I1, q0, xi):
    return I0 / ((q - q0)**2 * xi**2 + 1) + I1 / ((q - q0)**2 * xi**2 + 1)**2

def fit_lorentz_para(Iq, qpara, q0):
    
    I0_guess = np.max(Iq)
    xi_guess = 1.0

    p0 = [I0_guess, q0, xi_guess]
    
    try:
        popt, pcov = curve_fit(lorentz_para, qpara, Iq, p0=p0, maxfev=10000)
        
    except RuntimeError:
        print("Lorentz fit did not converge.")
        return None, None

    return popt, pcov

def corr_length_parallel_mix(Iq, qpara, q0):

    I0_guess = np.max(Iq) * 0.5
    I1_guess = np.max(Iq) * 0.5
    xi_guess = 100
    p0 = [I0_guess, I1_guess, q0, xi_guess]
    bounds = ([0, 0, q0-0.1, 0], [np.inf, np.inf, q0+0.1, np.inf])
    try:
        popt, pcov = curve_fit(lorentz_para_mix, qpara, Iq, p0=p0, bounds=bounds, maxfev=20000)
        return popt, pcov
    except RuntimeError:
        print("Lorentz mix fit did not converge.")
        return None, None


def gaussian_theta(theta, I0, sigma):
    """Gaussian centered at zero with no baseline"""
    return I0 * np.exp(-theta**2 / (2 * sigma**2))


def fit_gaussian(shifted_theta, shifted_Itheta):
    """
    Fit angular data to Gaussian centered at 0, return FWHM and half-max range.
    """
    # Initial guesses
    I0_guess = np.max(shifted_Itheta)
    sigma_guess = 0.2  # radians
    p0 = [I0_guess, sigma_guess]

    # Fit
    popt, pcov = curve_fit(gaussian_theta, shifted_theta, shifted_Itheta, p0=p0, maxfev=10000)

    return popt, pcov

def lorentz_perp(q, I0, xi):

    return I0 / (q**2 * xi**2 + 1)

def fit_lorentz_perp(qvals, Iq):
    """
    Fit perpendicular intensity profile to Lorentzian.
    Uses inverse of half-height width as xi initial guess.
    """
    # Initial guess for I0
    I0_guess = np.max(Iq)
    '''
    # Estimate q at half maximum
    half_max = I0_guess / 2
    try:
        q_hwhm_candidates = qvals[np.where(Iq >= half_max)[0]]
        if len(q_hwhm_candidates) > 1:
            q_hwhm = q_hwhm_candidates[-1] - q_hwhm_candidates[0]
            q_hwhm = max(q_hwhm, 1e-6)  # avoid zero
            xi_guess = 1.0 / q_hwhm
        else:
            xi_guess = 0.01  # fallback
    except Exception:
        xi_guess = 0.01
    '''
    xi_guess = 0.01
    p0 = [I0_guess, xi_guess]

    # Fit
    popt, pcov = curve_fit(lorentz_perp, qvals, Iq, p0=p0, maxfev=10000)

    return popt, pcov


def fit_peak(Iq_pixel, qvals, q_peak, shifted_Itheta, shifted_theta, baseline):
    
    popt, pcov = fit_lorentz_para(Iq_pixel - baseline, qvals, q_peak)
    
    I0_fit, q0_fit, xi_fit = popt
    fwhm = 2/xi_fit
    domain_size = 2*np.pi/fwhm
    
    popt, pcov = fit_lorentz_perp(shifted_theta, shifted_Itheta - baseline)
    
    I0_fit2, xi_fit2 = popt
    theta_range = 2/xi_fit2
    
    return I0_fit, q0_fit, xi_fit, fwhm, domain_size, I0_fit2, xi_fit2, theta_range

def save_smectic_plot(filename,
                      Iq_pixel, qvals, q_peak, shifted_Itheta, shifted_theta, baseline,
                      I0_fit, q0_fit, xi_fit, I0_fit2, xi_fit2,
                      image, Q, angle,
                        qmin, qmax, Nq, Ntheta,
                        vmin=None, vmax=None, initial_angle = 60/180*np.pi,
                        q_color='red', theta_color='blue'):

    
    angle = np.where(angle < 0, angle + 2*np.pi, angle)
    # Compute bin edges
    q_edges = np.linspace(qmin, qmax, Nq + 1)
    
    list_theta = np.linspace(initial_angle, initial_angle + np.pi, Ntheta+1)
    list_theta = list_theta[1:]
    binsize_theta = np.pi/Ntheta
    theta_edges = np.linspace(list_theta[0] - binsize_theta / 2, list_theta[-1] + binsize_theta / 2, Ntheta + 1)
    
    # Set contrast
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(image, [1, 99])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    # -----------------------------
    # (1) Image with bin boundaries
    # -----------------------------
    axes[0].imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    
    # Overlay q-bin contours
    for q in q_edges:
        axes[0].contour(Q, levels=[q], colors=q_color, linewidths=0.5, linestyles='dotted')
    
    # Overlay theta-bin contours
    for theta in theta_edges:
        axes[0].contour(angle, levels=[theta], colors=theta_color, linewidths=0.5, linestyles='dotted')
    
    axes[0].set_title("Q–θ Bin Boundaries Overlay")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    axes[0].set_xlim([450, 600])
    axes[0].set_ylim([625, 750])
    
    # -----------------------------
    # (2) q vs intensity (Lorentz fit)
    # -----------------------------
    axes[1].scatter(qvals, Iq_pixel - baseline, label="Data", color="blue", alpha=0.6)

    popt = I0_fit, q0_fit, xi_fit
    # Smooth curve
    q_fit = np.linspace(np.min(qvals), np.max(qvals), 500)
    I_fit = lorentz_para(q_fit, *popt)
    axes[1].plot(q_fit, I_fit, "k-", label="Lorentz fit", linewidth=2)
    
    axes[1].set_xlabel("q")
    axes[1].set_ylabel("Intensity")
    axes[1].set_title("I_Parallel Fit (q)")
    axes[1].legend()
    
    # -----------------------------
    # (3) θ vs intensity (Lorentz-perp fit)
    # -----------------------------
    popt = I0_fit2, xi_fit2
    #print(f"Perpendicular fit: I0={I0_fit:.3f}, xi={xi_fit:.3f}, theta_range={theta_range:.3f}")
    
    theta_fit = np.linspace(0, np.max(shifted_theta), 500)
    I_fit = lorentz_perp(theta_fit, *popt)
    
    axes[2].scatter(shifted_theta, shifted_Itheta - baseline, label="Data", color="blue", alpha=0.6)
    axes[2].plot(theta_fit, I_fit, "r-", label="Lorentz-perp fit", linewidth=2)
    
    axes[2].set_xlabel("shifted θ (radians)")
    axes[2].set_ylabel("Intensity (a.u.)")
    axes[2].set_title("I_Perp Fit (θ)")
    axes[2].legend()
    
    #plt.tight_layout()
    plt.savefig(filename + 'fitSmecticRing.png')
    plt.close(fig)

def smectic_procedure(directory, filename, image, mask, mesh_q, mesh_theta, list_q, list_theta, Q, angle, qmin, qmax, initial_angle):
    
    I_ave = calculate_average_intensity_within_mask(image, mask, mesh_q, mesh_theta)
    
    q_peak, q_idx_closest, theta_peak_horizontal_zero, theta_peak, theta_idx_closest = find_smectic_peak(I_ave, list_q, list_theta, initial_angle, flag_plot = False)
    
    theta_bin = list_theta[1] - list_theta[0]
    
    Iq_pixel, qvals = Ipeak_all_pixel(image, mask, Q, angle, qmin, qmax, theta_peak, theta_bin)
    
    #Iq = np.nanmean(I_ave[:, theta_idx_closest-1:theta_idx_closest+1], axis = 1)
    Itheta = np.nanmean(I_ave[q_idx_closest-1:q_idx_closest+1, :], axis = 0)
    
    shifted_theta, shifted_Itheta = shift_and_average_over_theta(list_theta, Itheta, theta_peak)
    
    baseline = compute_baseline_intensity_smectic(Q, angle, image, theta_peak, initial_angle, qmin, qmax, pixels=5, angle_width_deg=2.5, flag_plot = False)
    
    I0_fit, q0_fit, xi_fit, fwhm, domain_size, I0_fit2, xi_fit2, theta_range = fit_peak(Iq_pixel, qvals, q_peak, shifted_Itheta, shifted_theta, baseline)
    
    Nq = len(list_q)
    Ntheta = len(list_theta)
    '''
    save_smectic_plot(directory + filename,
                        Iq_pixel, qvals, q_peak, shifted_Itheta, shifted_theta, baseline,
                        I0_fit, q0_fit, xi_fit, I0_fit2, xi_fit2,
                        image, Q, angle,
                        qmin, qmax, Nq, Ntheta,
                        vmin=0, vmax=100, initial_angle = 60/180*np.pi)
    '''
    del image
    del Q
    del angle
    del mask
    gc.collect()
    #plt.close('all')
    return I0_fit, q0_fit, xi_fit, fwhm, domain_size, I0_fit2, xi_fit2, theta_range


def get_smaller_peak_range(file_path, column_name="smaller peak q [A-1]"):
    """
    Load CSV file and return min and max values of the 'smaller peak q [A-1]' column.
    
    Parameters:
        file_path (str): Path to the CSV file
        column_name (str): Name of the column to analyze (default: 'smaller peak q [A-1]')
    
    Returns:
        (min_val, max_val): tuple of floats
    """
    df = pd.read_csv(file_path)
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    return min_val, max_val
