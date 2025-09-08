import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

def load_all_edf_files(directory):
    #edf_files = [f for f in os.listdir(directory) if f.endswith('.edf') and not f.startswith('._')]
    edf_files = sorted(
        [f for f in os.listdir(directory) if f.endswith('.edf') and not f.startswith('._')]
    )
    return edf_files

def read_esrf_edf_image(filename):
    # Parameters known from the header (you can also parse these dynamically if needed)
    header_size = get_edf_header_size(filename)
    dim1, dim2 = 1004, 1066
    dtype = np.dtype('<f4')  # little-endian float32

    with open(filename, 'rb') as f:
        f.seek(header_size)
        
        data = np.fromfile(f, dtype=dtype, count=dim1 * dim2)
        image = data.reshape((dim2, dim1))  # rows x cols

    return image

def show_image(filename, image, cmap='gray', vmin=None, vmax=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Intensity')
    plt.title(filename)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

def save_image(image, directory, filename, cmap = 'gray', vmin=0, vmax=100):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Intensity')
    plt.title(filename)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig(directory + filename + '.png')
    plt.close()

def get_edf_header_size(filename):
    with open(filename, 'rb') as f:
        header = b""
        while True:
            chunk = f.read(512)
            header += chunk
            if b'}\n' in chunk:
                break
        header_end = header.find(b'}\n') + 2
        # Round up to the next multiple of 512
        header_size = ((header_end + 511) // 512) * 512
        return header_size

def extract_edf_parameters(filename):
    """Extracts key metadata parameters from an ESRF-style EDF file header."""
    with open(filename, 'rb') as f:
        header_bytes = f.read(4096)  # EDF header typically < 4 KB
    header_str = header_bytes.decode('ascii', errors='ignore')
    header_str = header_str[:header_str.find('}') + 1]

    # Parse header into dictionary
    header = {}
    for line in header_str.strip('{} \n\r').split('\n'):
        if '=' in line:
            key, value = line.strip().split('=', 1)
            header[key.strip()] = value.strip().strip(';')

    # Extract parameters and convert to float
    center_1 = float(header.get("Center_1", "nan"))
    center_2 = float(header.get("Center_2", "nan"))
    Dim_1 = float(header.get("Dim_1", "nan"))
    Dim_2 = float(header.get("Dim_2", "nan"))
    sample_distance = float(header.get("SampleDistance", "nan"))
    wavelength = float(header.get("WaveLength", header.get("Wavelength", "nan")))
    psize_1 = float(header.get("PSize_1", "nan"))  # fast axis (x)
    psize_2 = float(header.get("PSize_2", "nan"))  # slow axis (y)
    
    return {
        "Center_1": center_1,
        "Center_2": center_2,
        "Dim_1": Dim_1,
        "Dim_2": Dim_2,
        "SampleDistance": sample_distance,
        "WaveLength": wavelength,
        "PSize_1": psize_1,
        "PSize_2": psize_2
    }

def create_qmap(params, xc = None, yc = None):
    if xc == None:
        x = np.arange(params["Dim_1"]) - params["Center_1"] + 1
    else:
        x = np.arange(params["Dim_1"]) - xc + 1

    if yc == None:
        y = np.arange(params["Dim_2"]) - params["Center_2"] + 1
    else:
        y = np.arange(params["Dim_2"]) - yc + 1

    X, Y = np.meshgrid(x * params["PSize_1"], y * params["PSize_2"], indexing = 'xy')
    r = np.sqrt(X**2 + Y**2)
    theta2 = np.arctan2(r, params["SampleDistance"])
    Q = 4*np.pi/params["WaveLength"] * np.sin(theta2/2) * 1e-10
    
    angle = np.arctan2(Y, X)
    Qx = Q * np.cos(angle)
    Qy = Q * np.sin(angle)

    return Q, angle, Qx, Qy

def plot_image_with_q_contours(image, q_map, q_levels=[0.5, 1.0, 1.5], vmin = None, vmax = None):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    contours = plt.contour(q_map, levels=q_levels, colors='red')
    plt.clabel(contours, inline=True, fmt="q=%.1f Å⁻¹", fontsize=10)
    plt.title("EDF Image with Q-Contours")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.show()

def check_Qmap_center(image, params, Q, q_levels=[0.5, 1.0, 1.5], vmin=None, vmax=None, cmap='gray'):
    
    xc = int(params["Center_1"])
    yc = int(params["Center_2"])
    interval = 70
    
    plt.figure(figsize=(8, 6))
    plt.imshow(image[yc - interval:yc + interval, xc - interval:xc + interval], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    contours = plt.contour(Q[yc - interval:yc + interval, xc - interval:xc + interval], levels=q_levels, colors='red')
    plt.clabel(contours, inline=True, fmt="q=%.1f Å⁻¹", fontsize=10)
    #plt.colorbar(label='Intensity')
    plt.title('ESRF EDF Image')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

def save_Qmaps(Q, angle, Qx, Qy, xc, yc, output_file= "qmap_output.npz"):
    np.savez(output_file, Q=Q, angle=angle, Qx=Qx, Qy=Qy, xc = xc, yc = yc)

def load_Qmaps(filename):
    data = np.load(filename)
    return data["Q"], data["angle"], data["Qx"], data["Qy"]


def create_qtheta_mask(Q, angle, qmin, qmax, Nq, Ntheta):
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
    theta_edges = np.linspace(-np.pi, 0, Ntheta + 1)
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


def show_overlay_mask(image, Q, angle,
                        qmin, qmax, Nq, Ntheta,
                        vmin=None, vmax=None,
                        q_color='red', theta_color='blue'):
    """
    Overlay q and theta bin boundaries on the image using contours.
    """
    # Compute bin edges
    q_edges = np.linspace(qmin, qmax, Nq + 1)
    
    list_theta = np.linspace(-np.pi, 0, Ntheta+1)
    list_theta = list_theta[1:]
    binsize_theta = np.pi/Ntheta
    theta_edges = np.linspace(list_theta[0] - binsize_theta / 2, list_theta[-1] + binsize_theta / 2, Ntheta + 1)

    # Set contrast
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(image, [1, 99])

    # Plot image
    plt.figure(figsize=(8, 6))
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
    plt.tight_layout()
    plt.show()


def calculate_average_intensity_within_mask(image, mask, mesh_q, mesh_theta):
    """
    Compute average intensity in each bin defined by the mask.

    Parameters:
        image      : 2D array of raw intensity values
        mask       : 2D array of bin indices (NaN for out-of-bin pixels)
        mesh_q     : 2D array of q-bin centers (shape: Nq × Ntheta)
        mesh_theta : 2D array of theta-bin centers (same shape)

    Returns:
        Iqtheta : 2D array (Nq × Ntheta) of average intensity in each bin
                 (NaN if no pixels fall into a bin)
    """
    Nq, Ntheta = mesh_q.shape
    num_bins = Nq * Ntheta

    # Flatten image and mask
    flat_image = image.flatten()
    flat_mask = mask.flatten()

    # Initialize arrays to accumulate sums and counts
    bin_sums = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    # Iterate over valid pixels only
    valid_mask = ~np.isnan(flat_mask)
    indices = flat_mask[valid_mask].astype(int)
    values = flat_image[valid_mask]

    # Accumulate sum and count for each bin
    np.add.at(bin_sums, indices, values)
    np.add.at(bin_counts, indices, 1)

    # Compute average, set 0-count bins to NaN
    with np.errstate(invalid='ignore', divide='ignore'):
        bin_avg = bin_sums / bin_counts
    bin_avg[bin_counts == 0] = np.nan

    # Reshape to match mesh_q / mesh_theta
    #Iqtheta = bin_avg.reshape(Nq, Ntheta)
    Iqtheta = bin_avg.reshape(Ntheta, Nq).T

    return Iqtheta


def save_qtheta_mask(filename, mask, list_q, list_theta, bin_lookup, mesh_q, mesh_theta, qmin, qmax):
    """
    Save Q–theta mask and bin definitions into a .npz file.

    Parameters:
        filename   : str, path to save the file (e.g., 'qtheta_mask.npz')
        mask       : 2D array of bin indices (NaN for out-of-bounds)
        list_q     : 1D array of q-bin centers
        list_theta : 1D array of theta-bin centers
    """
    # Save everything into compressed .npz file
    np.savez(filename,
             mask=mask,
             list_q=list_q,
             list_theta=list_theta,
             mesh_q=mesh_q,
             mesh_theta=mesh_theta,
             bin_lookup = bin_lookup,
             qmin = qmin,
             qmax = qmax)
    print(f"Saved Q–θ mask data to: {filename}")

def load_qtheta_mask(filename):
    data = np.load(filename)
    return data["mask"], data["list_q"], data["list_theta"], data["mesh_q"], data["mesh_theta"], data["bin_lookup"], data["qmin"], data["qmax"]



def check_bad_pixels(image, threshold=1e6):
    """
    Check for bad pixels in the image.

    Parameters:
        image     : 2D numpy array of intensity values
        threshold : float, upper bound beyond which values are flagged

    Returns:
        report : dict with counts of NaNs and large values
    """
    num_nans = np.isnan(image).sum()
    num_large = np.sum(image > threshold)

    report = {
        "NaN pixels": num_nans,
        f"Pixels > {threshold:g}": num_large,
        "Total pixels": image.size,
    }

    # Print report
    print("=== Bad Pixel Check ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    return report


def remove_bad_pixels_from_mask(mask, image, threshold=1e6):
    """
    Return a new mask where bad pixels (NaN or too large) are excluded.

    Parameters:
        mask     : 2D array of bin indices (can include NaNs)
        image    : 2D array of intensity values
        threshold: float, any pixel > threshold is marked invalid

    Returns:
        new_mask : 2D array, same as `mask` but with bad pixels labeled as NaN
    """
    new_mask = mask.copy()

    # Identify bad pixels
    bad_pixel_mask = np.isnan(image) | (image > threshold)

    # Apply to mask
    new_mask[bad_pixel_mask] = np.nan

    return new_mask

#find the angle of the peak, fit the peak to a parabola

def fit_parabola_peak(theta, intensity, num_points=5):
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

    # Wrap θ to [-π, 0]
    theta_peak_wrapped = ((theta_peak + np.pi) % np.pi) - np.pi

    return theta_peak_wrapped, I_peak


def shift_and_filter_theta(theta_range, intensity_profile, theta_max):
    """
    Translate theta such that the maximum is at 0, and keep only [0, 90°] after shift.

    Parameters:
        theta_range        : 1D array of original theta bin centers (in radians)
        intensity_profile  : 1D array of intensity values (same shape)
        theta_max          : scalar (radians), location of max found earlier

    Returns:
        theta_filtered     : 1D array of shifted angles in [0, π/2]
        intensity_filtered : 1D array of intensity values at those angles
    """
    # Shift and wrap
    theta_shifted = theta_range - theta_max
    theta_shifted = np.concatenate((theta_shifted, theta_shifted + np.pi))
    intensity_profile = np.concatenate((intensity_profile, intensity_profile))
    
    theta_shifted = (theta_shifted + 4 * np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
    #print(theta_shifted)
    # Filter range [0, π/2]
    valid = (theta_shifted >= 0) & (theta_shifted <= np.pi / 2)
    theta_filtered = theta_shifted[valid]
    intensity_filtered = intensity_profile[valid]

    return theta_filtered, intensity_filtered

#baseline
def compute_baseline_intensity(Q, angle, image, theta_max, qmin, qmax, pixels=5, angle_width_deg=2.5, flag_plot = True):
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

    # Convert 90° to radians
    theta_plus = theta_max + np.radians(90)
    theta_minus = theta_max - np.radians(90)

    # Choose the one within [-π, π] and between -180° and 30°
    candidates = [theta_plus, theta_minus]
    theta_bg = None
    for t in candidates:
        t_wrapped = ((t + np.pi) % (2 * np.pi)) - np.pi
        deg = np.degrees(t_wrapped)
        if -180 <= deg <= 30:
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
        plt.show()
        
    # Compute average
    if np.any(region_mask):
        baseline = np.nanmean(image[region_mask])
    else:
        raise ValueError("No pixels found in baseline region.")

    return baseline


def gradient_near_qc(Q, qc, delta_q=0.05):
    """
    Compute the gradient of Q near a target value qc.

    Parameters:
        Q        : 2D array of q values
        qc       : target q value to examine
        delta_q  : half-width of the q-range to consider

    Returns:
        dq_map   : 2D array of |∇Q| over the full image
        grad_qx  : ∂Q/∂x
        grad_qy  : ∂Q/∂y
        mask     : Boolean mask of where Q is within [qc - δq, qc + δq]
        dq_mean  : Average |∇Q| in that region
    """
    # Gradient components
    grad_qy, grad_qx = np.gradient(Q)  # rows=y, cols=x
    dq_map = np.sqrt(grad_qx**2 + grad_qy**2)

    # Create a mask for values near qc
    mask = (Q >= qc - delta_q) & (Q <= qc + delta_q)
    #plt.imshow(mask)
    #plt.show()
    # Mean gradient in that region
    if np.any(mask):
        dq_mean = np.nanmean(dq_map[mask])
    else:
        dq_mean = np.nan
    
    return dq_mean

# Kratky function with six parameters
def Kratky_function(kai, p0, p1, p2, p3, p4, p5):
    cos_sq = np.cos(kai)**2
    return (p0 
            + 1/2 * p1 * cos_sq 
            + 3/8 * p2 * cos_sq ** 2 
            + 5/16 * p3 * cos_sq ** 3 
            + 35/128 * p4 * cos_sq ** 4 
            + 63/256 * p5 * cos_sq ** 5)

def order_parameters(params):
    
    narray = np.arange(6) * 2
    denominator1 = 1/(2 * narray + 1)
    denominator2 = 1/(2 * narray + 3)
    denominator3 = 1/(2 * narray + 5)
    
    cos2beta = np.sum(denominator2 * params)/np.sum(denominator1 * params)
    cos4beta = np.sum(denominator3 * params)/np.sum(denominator1 * params)
    
    P2 = (3 * cos2beta - 1)/2
    P4 = (35 * cos4beta - 30 * cos2beta + 3)/8
    return P2, P4


def procedure(filename, Q, angle, mask, mesh_q, mesh_theta, qmin, qmax, flag_plot = True, save_plot = False):
    
    image = read_esrf_edf_image(filename)
    mask = remove_bad_pixels_from_mask(mask, image, threshold=1e6)
    
    I = calculate_average_intensity_within_mask(image, mask, mesh_q, mesh_theta)
    #plt.figure(figsize = (4,3))
    #plt.scatter(mesh_theta, I)
    
    theta_max, I_max = fit_parabola_peak(mesh_theta.flatten(), I.flatten())
    #print(f"Maximum intensity: {I_max:.2f} at θ = {np.degrees(theta_max):.2f}°")
    
    kai, I_kai = shift_and_filter_theta(mesh_theta.flatten(), I.flatten(), theta_max)
    #plt.figure(figsize = (4,3))
    #plt.scatter(kai, I_kai)
    
    baseline = compute_baseline_intensity(Q, angle, image, theta_max, qmin, qmax, pixels=5, angle_width_deg=2.5, flag_plot = False)
    I_kai_offset = I_kai - baseline
    
    # Fit the data
    initial_guess = np.ones(6)
    params_opt, _ = curve_fit(Kratky_function, kai, I_kai_offset, p0=initial_guess)
    
    # Store as 1D array
    params = np.array(params_opt)
    
    angle = np.linspace(0, np.pi/2, 100)
    # Evaluate the fitted curve
    I_fit = Kratky_function(angle, *params)
    
    # Output parameters
    #print("Best-fit parameters (1D array):", params)
    P2, P4 = order_parameters(params)
    #print("order parameters:", P2, P4)
    # Optional plot
    if flag_plot:
        plt.figure(figsize = (4,3))
        plt.plot(kai, I_kai_offset, 'o', label='Data')
        plt.plot(angle, I_fit, '-', label='Fit')
        plt.xlabel("Kai")
        plt.ylabel("I_kai_offset")
        plt.title(filename)
        plt.legend()
        plt.show()

    if save_plot:
        plt.figure(figsize = (4,3))
        plt.plot(kai, I_kai_offset, 'o', label='Data')
        plt.plot(angle, I_fit, '-', label='Fit')
        plt.xlabel("Kai")
        plt.ylabel("I_kai_offset")
        plt.title(filename)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename + 'fitorderparameter.png')
        plt.close()
        
    return P2, P4, params, theta_max





######        1D  #############
def read_dat_three_columns(file_path):
    # First, detect how many header lines to skip
    skip_rows = 0

    with open(file_path, 'r') as f:
        
        for line in f:
            parts = line.strip().split()
            try:
                # Try converting first token to float to detect numeric start
                float(parts[0])
                break  # found first numeric line
            except (ValueError, IndexError):
                skip_rows += 1

    # Load only numeric data
    data = np.loadtxt(file_path, skiprows=skip_rows)

    if data.shape[1] != 3:
        raise ValueError(f"Expected 3 columns, but got {data.shape[1]} columns")

    
    return data

def peaks_in_1D(data):
    x, y = data[:, 0], data[:, 1]
    peaks, _ = find_peaks(y)
    
    # Identify peaks near the two target q values
    target_qs = [0.14, 1.4]
    tolerance = 0.05
    peak_positions = []
    for target in target_qs:
        mask = (x[peaks] > target - tolerance) & (x[peaks] < target + tolerance)
        if np.any(mask):
            idx_max = peaks[mask][np.argmax(y[peaks][mask])]
            peak_positions.append(x[idx_max])
        else:
            peak_positions.append(None)
    
    return peak_positions

def save_1D_plot(directory, filename, data, peak_positions):
    plt.figure(figsize=(4,3))
    plt.errorbar(data[:,0], data[:,1], data[:,2])
    plt.title(filename)
    x = data[:,0]
    y = data[:,1]
    for pos in peak_positions:
        if pos is not None:
            plt.scatter(pos, y[np.argmin(np.abs(x - pos))], 
                        color='red', s=20, zorder=5, label=f"Peak at {pos:.3f}")
    plt.xlabel(r'q [A$^{-1}$]')
    plt.ylabel('I(q)')
    plt.tight_layout()
    plt.savefig(directory + filename + '.png')
    plt.close()
    #plt.show()

def process_1D_data(filename, directory, save_plot = False):
    dat_filename = os.path.splitext(filename)[0] + '.dat'
    #print(dat_filename)
    data = read_dat_three_columns(directory + dat_filename)

    peak_positions = peaks_in_1D(data)
    
    if save_plot:
        save_1D_plot(directory, dat_filename, data, peak_positions)

    return peak_positions