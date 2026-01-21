import numpy as np


def gaussian_bias_1D(numAds, centers, amplitudes, sigma):
    """
    Evaluate 1D Gaussian bias function at given numAds values.
    
    Parameters:
    -----------
    numAds : scalar or array-like
        The numAds value(s) at which to evaluate the bias
    centers : array-like
        Centers of the Gaussian hills
    amplitudes : array-like
        Amplitudes (heights) of the Gaussian hills
    sigma : float
        Width of the Gaussian potentials
    
    Returns:
    --------
    bias : scalar or array
        The total bias energy (sum of all Gaussians)
    """
    numAds = np.asarray(numAds)
    bias = np.zeros_like(numAds, dtype=float)
    for center, amp in zip(centers, amplitudes):
        bias += amp * np.exp(-0.5 * ((numAds - center) / sigma)**2)
    return bias


def fill_gaussian_bias_1D(numAds_GM, gcfe_GM, sigma=None, amplitude=None, tolerance=0.05, max_iter=1000000):
    """
    Create a 1D metadynamics-style bias function by iteratively adding Gaussians to lowest GCFE regions
    until the landscape is roughly flat.
    
    Parameters:
    -----------
    numAds_GM : array-like
        The numAds values at GM positions (centers of Gaussians)
    gcfe_GM : array-like
        The gcfe values at GM positions
    sigma : float, optional
        Width of the Gaussian potentials. If None, uses adaptive width based on spacing.
    amplitude : float, optional
        Height (amplitude) of all Gaussian hills. If None, uses adaptive value based on energy range.
    tolerance : float, optional
        Maximum allowed standard deviation of biased GCFE for convergence. Default is 0.05 eV.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    
    Returns:
    --------
    func_bias_gaussian : callable
        A function that takes numAds (scalar or array) and returns the bias energy.
    """
    numAds_GM = np.array(numAds_GM)
    gcfe_GM = np.array(gcfe_GM)
    
    if sigma is None:
        if len(numAds_GM) > 1:
            spacing = np.diff(np.sort(numAds_GM))
            sigma = np.median(spacing[spacing > 0]) * 1
        else:
            sigma = 1
    
    # Determine adaptive amplitude if not provided
    if amplitude is None:
        # if len(gcfe_GM) > 1:
        #     energy_range = np.ptp(gcfe_GM)  # Peak-to-peak range
        #     # Use 5% of energy range as default amplitude (smaller for iterative approach)
        #     amplitude = energy_range * 0.05
        # else:
        amplitude = 0.02  # Default small value
    
    # Initialize: start with empty bias
    centers = []
    amplitudes = []
    current_bias = np.zeros(len(numAds_GM))
    
    # Iterative process: add Gaussians to lowest regions until flat
    for iteration in range(max_iter):
        # Calculate current biased GCFE (bias is ADDED to raise the landscape)
        gcfe_biased = gcfe_GM + current_bias
        
        # Check convergence: if std dev is below tolerance, we're done
        std_dev = np.std(gcfe_biased)
        if std_dev < tolerance:
            print(f"Converged after {iteration} iterations. Final std dev: {std_dev:.6f} eV")
            break
        
        # Find the position with the lowest biased GCFE (deepest well)
        idx_min = np.argmin(gcfe_biased)
        center_new = numAds_GM[idx_min]
        
        # Add a Gaussian at this position
        centers.append(center_new)
        amplitudes.append(amplitude)
        
        # Update bias at all GM positions using the single Gaussian function
        current_bias += gaussian_bias_1D(numAds_GM, [center_new], [amplitude], sigma)
    
    else:
        # If we didn't break, we hit max_iter
        std_dev = np.std(gcfe_GM + current_bias)
        print(f"Reached max_iter ({max_iter}). Final std dev: {std_dev:.6f} eV")
    
    centers = np.array(centers)
    amplitudes = np.array(amplitudes)
    
    def func_bias_gaussian(numAds):
        return gaussian_bias_1D(numAds, centers, amplitudes, sigma)
    
    func_bias_gaussian.centers = centers
    func_bias_gaussian.amplitudes = amplitudes
    func_bias_gaussian.sigma = sigma
    func_bias_gaussian.n_gaussians = len(centers)
    func_bias_gaussian.amplitude = amplitude
    
    return func_bias_gaussian


def save_bias_to_file_1D(func_bias_gaussian, filename):
    """Save 1D Gaussian bias parameters to a text file."""
    with open(filename, 'w') as f:
        f.write(f"# sigma: {func_bias_gaussian.sigma}\n")
        f.write(f"# amplitude: {func_bias_gaussian.amplitude}\n")
        f.write(f"# n_gaussians: {func_bias_gaussian.n_gaussians}\n")
        f.write("# center    amplitude\n")
        for center, amp in zip(func_bias_gaussian.centers, func_bias_gaussian.amplitudes):
            f.write(f"{center:.10f}    {amp:.10f}\n")
    print(f"Saved {func_bias_gaussian.n_gaussians} Gaussians to {filename}")


def load_bias_from_file_1D(filename):
    """Load 1D Gaussian bias parameters from a text file and reconstruct the bias function."""
    centers = []
    amplitudes = []
    sigma = None
    amplitude = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if 'sigma:' in line:
                    sigma = float(line.split('sigma:')[1].strip())
                elif 'amplitude:' in line:
                    amplitude = float(line.split('amplitude:')[1].strip())
                continue
            parts = line.split()
            if len(parts) >= 2:
                centers.append(float(parts[0]))
                amplitudes.append(float(parts[1]))
    
    if sigma is None:
        raise ValueError("Could not find sigma in file header")
    
    centers = np.array(centers)
    amplitudes = np.array(amplitudes)
    
    # Use amplitude from file if available, otherwise use mean of amplitudes
    if amplitude is None:
        amplitude = np.mean(amplitudes) if len(amplitudes) > 0 else 0.1
    
    def func_bias_gaussian(numAds):
        return gaussian_bias_1D(numAds, centers, amplitudes, sigma)
    
    func_bias_gaussian.centers = centers
    func_bias_gaussian.amplitudes = amplitudes
    func_bias_gaussian.sigma = sigma
    func_bias_gaussian.n_gaussians = len(centers)
    func_bias_gaussian.amplitude = amplitude
    
    print(f"Loaded {func_bias_gaussian.n_gaussians} Gaussians from {filename}")
    print(f"Sigma: {sigma:.6f}")
    print(f"Amplitude: {amplitude:.6f} eV")
    
    return func_bias_gaussian

