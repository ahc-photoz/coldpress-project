import numpy as np

QUANTITY_DESCRIPTIONS = {
    'Z_MODE': 'Mode of the redshift PDF, defined as the redshift with maximum probability density.',
    'Z_MEAN': 'Mean of the redshift PDF, defined as the integral over z of z*P(z).',
    'Z_MEDIAN': 'Median of the redshift PDF (i.e., the redshift that has a 50/50 chance of the true redshift being on either side).',
    'Z_RANDOM': 'A random redshift value obtained with the PDF as the underlying probability distribution.',
    'Z_MODE_ERR': '1-sigma uncertainty in Z_MODE.',
    'Z_MEAN_ERR': '1-sigma uncertainty in Z_MEAN.',
    'ODDS_MODE': 'Odds parameter for Z_MODE.',
    'ODDS_MEAN': 'Odds parameter for Z_MEAN.',
    'Z_MIN_HPDCI68': 'Lower bound of the 68% highest posterior density credible interval.',
    'Z_MAX_HPDCI68': 'Upper bound of the 68% highest posterior density credible interval.',
    'Z_MIN_HPDCI95': 'Lower bound of the 95% highest posterior density credible interval.',
    'Z_MAX_HPDCI95': 'Upper bound of the 95% highest posterior density credible interval.',
    'ODDS_MODE': 'Probability that the true redshift lies within a specific interval around Z_MODE (default is ± 0.03 × (1 + Z_MODE).',
    'ODDS_MEAN': 'Probability that the true redshift lies within a specific interval around Z_MEAN (default is ± 0.03 × (1 + Z_MEAN).'
}

ALL_QUANTITIES = set(QUANTITY_DESCRIPTIONS.keys())

def measure_from_quantiles(quantiles, quantities_to_measure, odds_window=0.03):
    """
    Computes a set of statistical quantities from a single PDF's quantiles.

    Args:
        quantiles (np.ndarray): The array of quantile values for one PDF.
        quantities_to_measure (list): A list of strings of the desired quantities.
                                      Use 'ALL' to compute all available quantities.
        odds_window (float, optional): Half-width for the odds calculation. Defaults to 0.03.

    Raises:
        ValueError: If an unknown quantity is requested.

    Returns:
        dict: A dictionary mapping the name of each requested quantity to its value.
    """
    dependencies = {
        'Z_MODE_ERR': ['Z_MODE'],
        'ODDS_MODE': ['Z_MODE'],
        'ODDS_MEAN': ['Z_MEAN']
    }

    # Determine which quantities to compute
    requested_q = {q.upper() for q in quantities_to_measure}

    if 'ALL' in requested_q:
        q_to_compute = ALL_QUANTITIES
    else:
        unknown_q = requested_q - ALL_QUANTITIES
        if unknown_q:
            raise ValueError(f"Unknown quantities specified: {', '.join(unknown_q)}")
        q_to_compute = requested_q

    # Resolve dependencies to determine all internal calculations needed
    internal_calcs = set(q_to_compute)
    for q in q_to_compute:
        if q in dependencies:
            internal_calcs.update(dependencies[q])

    # --- Perform all necessary calculations ---
    temp_results = {}
    if 'Z_MODE' in internal_calcs:
        temp_results['Z_MODE'] = zmode_from_quantiles(quantiles, width=0.005)
    if 'Z_MEAN' in internal_calcs:
        temp_results['Z_MEAN'] = zmean_from_quantiles(quantiles)
    if 'Z_MEDIAN' in internal_calcs:
        temp_results['Z_MEDIAN'] = zmedian_from_quantiles(quantiles)
    if 'Z_RANDOM' in internal_calcs:
        temp_results['Z_RANDOM'] = zrandom_from_quantiles(quantiles)
    if 'Z_MEAN_ERR' in internal_calcs:
        temp_results['Z_MEAN_ERR'] = zmean_err_from_quantiles(quantiles)
    if 'ODDS_MODE' in internal_calcs:
        temp_results['ODDS_MODE'] = odds_from_quantiles(quantiles, temp_results['Z_MODE'], odds_window=odds_window)
    if 'ODDS_MEAN' in internal_calcs:
        temp_results['ODDS_MEAN'] = odds_from_quantiles(quantiles, temp_results['Z_MEAN'], odds_window=odds_window)
    if 'Z_MODE_ERR' in internal_calcs:
        HPDCI68_mode_zmin, HPDCI68_mode_zmax = HPDCI_from_quantiles(quantiles, conf=0.68, zinside=temp_results['Z_MODE'])
        temp_results['Z_MODE_ERR'] = 0.5 * (HPDCI68_mode_zmax - HPDCI68_mode_zmin)
    if 'Z_MIN_HPDCI68' in internal_calcs or 'Z_MAX_HPDCI68' in internal_calcs:
        HPDCI68_zmin, HPDCI68_zmax = HPDCI_from_quantiles(quantiles, conf=0.68, zinside=None)
        temp_results['Z_MIN_HPDCI68'] = HPDCI68_zmin
        temp_results['Z_MAX_HPDCI68'] = HPDCI68_zmax
    if 'Z_MIN_HPDCI95' in internal_calcs or 'Z_MAX_HPDCI95' in internal_calcs:
        HPDCI95_zmin, HPDCI95_zmax = HPDCI_from_quantiles(quantiles, conf=0.95, zinside=None)
        temp_results['Z_MIN_HPDCI95'] = HPDCI95_zmin
        temp_results['Z_MAX_HPDCI95'] = HPDCI95_zmax

    # Filter the results to return only the originally requested quantities
    final_results = {key: temp_results[key] for key in q_to_compute if key in temp_results}
    
    return final_results

def zmode_from_quantiles(quantiles, width=0.005):
    """More robust approach against effects of z quantization in very narrow peaks."""
    Nq = len(quantiles)
    knots = np.linspace(0,1.0,Nq)
    dkplus = np.interp(quantiles+width,quantiles,knots,left=0,right=1)-knots
    dkminus = knots-np.interp(quantiles-width,quantiles,knots,left=0,right=1)
    if max(dkplus) > max(dkminus):
        i = np.argmax(dkplus)
        return quantiles[i]+width/2.
    else:
        i = np.argmax(dkminus)
        return quantiles[i]-width/2.

def zmedian_from_quantiles(quantiles):
    """Obtain the median redshift of the probability distribution"""
    knots = np.linspace(0, 1, len(quantiles))
    return np.interp(0.5, knots, quantiles)

def zmean_from_quantiles(quantiles):
    """Obtain the mean redshift of the probability distribution (integral of z * P(z))"""
    knots = np.linspace(0, 1, len(quantiles))
    return np.trapz(quantiles, knots)

def zmean_err_from_quantiles(quantiles):
    """Obtain the stddev of the mean redshift of the probability distribution"""
    knots = np.linspace(0, 1, len(quantiles))
    mean = np.trapz(quantiles, knots)
    ez2 = np.trapz(quantiles**2, knots)
    variance = ez2 - mean**2
    return np.sqrt(variance)

def zrandom_from_quantiles(quantiles):
    """Obtain the redshift corresponding to a uniform random value between 0 and 1 of the cPDF"""
    u = np.random.uniform(0, 1)
    knots = np.linspace(0, 1, len(quantiles))
    return np.interp(u, knots, quantiles)

def odds_from_quantiles(quantiles, zcenter, odds_window=0.03):
    """Obtain the odds parameter"""
    knots = np.linspace(0, 1, len(quantiles))
    zbinmin = zcenter - odds_window*(1+zcenter)
    zbinmax = zcenter + odds_window*(1+zcenter)
    qz = np.interp([zbinmin,zbinmax],quantiles,knots,left=0,right=1)
    return qz[1]-qz[0]

def HPDCI_from_quantiles(quantiles, conf=0.68, zinside=None):
    """Obtain zmin and zmax corresponding to the highest probability density confidence interval."""
    knots = np.linspace(0, 1, len(quantiles))
    knot_interval = knots[1]-knots[0]

    if zinside is not None:
        qin = np.interp(zinside,quantiles,knots,left=0,right=1)
        qmin = max([0,qin-conf])
        qmax = min([1,qin+conf])
    else:
        qmin = 0
        qmax = 1

    start = np.arange(qmin,qmax-conf,0.2*knot_interval)
    end = np.arange(qmin+conf,qmax,0.2*knot_interval)
    
    if len(start) == 0 or len(end) == 0:
        z_center = np.interp(0.5, knots, quantiles)
        return (z_center, z_center)
        
    zmin = np.interp(start,knots,quantiles)
    zmax = np.interp(end,knots,quantiles)
    dz = zmax-zmin
    best = np.argmin(dz)
    return (zmin[best], zmax[best])