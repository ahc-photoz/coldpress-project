import numpy as np

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