import numpy as np
import struct
import sys
from scipy.interpolate import CubicSpline, PchipInterpolator

def decode_quantiles(packet):
    """
    Decode a byte packet into quantiles (including endpoints).
    Returns a numpy array of length M = 2 + number of deltas.
    """
        
    eps_int = packet[0]
    eps = eps_int*0.00001
    xmin_int, xmax_int = struct.unpack('<HH', packet[1:5])

    zmin = xmin_int * 0.0002 - 0.01
    zmax = xmax_int * 0.0002 - 0.01

    payload = packet[5:]
    zs = [zmin]
    i = 0
    length = len(payload)
    while i < length:
        b = payload[i]
        if (b == 0) and (max(payload[i:]) == 0): # end if just trailing zeros remain
            break
        if b < 255:
            d = b
            i += 1
        else:
            d = struct.unpack('>H', payload[i+1:i+3])[0]
            i += 3
        zs.append(zs[-1] + d * eps)

    # finally add zmax endpoint
    zs.append(zmax)
    return np.array(zs)


def cdf_to_pdf(z_quantiles, dz=None, z_min=None, z_max=None, zvector=None, method='linear'):
    """
    Given quantile locations z_quantiles (monotonic array of length M),
    return a PDF sampled on z = z_min, zmin+dz, ..., up to z_max.
    """
    zq = np.asarray(z_quantiles)
    M = len(zq)
    Fq = np.linspace(0.0, 1.0, M)
    
    if zvector is not None:
        z_grid = zvector
        dz = z_grid[1] - z_grid[0]
    else:    
        z_grid = np.arange(z_min, z_max + dz/2, dz)
    
    edges = np.empty(len(z_grid)+1, dtype=float)
    edges[0] = z_grid[0] - dz/2
    edges[1:] = z_grid + dz/2

    if method == 'linear':
        F_grid = np.interp(edges, zq, Fq, left=0.0, right=1.0)
    if method == 'spline':
        F_grid = np.zeros(len(edges))
        F_grid[edges < zq[0]] = 0.
        F_grid[edges > zq[-1]] = 1.
        inside = (edges >= zq[0]) & (edges <= zq[-1])
        F_inside = monotone_natural_spline(edges[inside], zq, Fq)
        F_grid[inside] = F_inside/F_inside[-1] 
    pdf = F_grid[1:] - F_grid[:-1]
        
    pdf /= np.sum(pdf*dz)
        
    if zvector is not None:
        return pdf
    else:
        return z_grid, pdf

def decode_to_histograms(int32col, zvector, force_range=False):
    """
    Decode the PDFs to a histogram PDF with bins defined by zvector.
    """
    Nsources = int32col.shape[0]
    PDF = np.zeros((Nsources,len(zvector)),dtype=np.float32)
    
    for i in range(Nsources):
        if np.any(int32col[i] != 0):
            packet = int32col[i].tobytes()
            qrecovered = decode_quantiles(packet)

            zmin_q, zmax_q = qrecovered[0], qrecovered[-1]
            zmin_vec, zmax_vec = zvector[0], zvector[-1]

            if zmin_q < zmin_vec or zmax_q > zmax_vec:
                if not force_range:
                    raise ValueError(f"Source {i}: Decoded redshift range [{zmin_q:.3f}, {zmax_q:.3f}] "
                                     f"exceeds the target zvector range [{zmin_vec:.3f}, {zmax_vec:.3f}]. "
                                     "Use force_range=True to override.")
                print(f"Warning: Source {i}: PDF range [{zmin_q:.3f}, {zmax_q:.3f}] truncated to zvector range [{zmin_vec:.3f}, {zmax_vec:.3f}].", file=sys.stderr)

            PDF[i] = cdf_to_pdf(qrecovered, zvector=zvector)

    return PDF

def decode_to_cdf(int32col, zv):
    """
    Recover the cumulative distribution functions evaluated at fixed set of redshifts zv from the 
    encoded representation of the quantiles
    """    
    Nsources = int32col.shape[0]
    cdf = np.zeros((Nsources,len(zv)),dtype=np.float32)
    for i in range(Nsources):
        if np.any(int32col[i] != 0):
            packet = int32col[i].tobytes()
            qrecovered = decode_quantiles(packet)
            qs = np.linspace(0,1,len(qrecovered))
            cdf[i] = np.interp(zv, qrecovered, qs, left=0.0, right=1.0) 
    return cdf

def monotone_natural_spline(Xout, X, Y):
    """
    Interpolate (X, Y) with a natural cubic spline, then correct any
    non-monotonic intervals using PCHIP interpolation.
    """
    spline = CubicSpline(X, Y, bc_type='natural')
    pchip = PchipInterpolator(X, Y)
    
    Yout = spline(Xout)
    
    Yp  = spline(X, 1)
    dYout = spline(Xout, 1)
        
    idx = np.searchsorted(X, Xout) - 1
    idx = np.clip(idx, 0, len(X)-2)
    
    for i in range(len(X)-1):
        mask = (idx == i) & ((dYout < 0) | (Yp[i] <= 0) | (Yp[i+1] <= 0))
        if np.any(mask):
            mask = (idx == i)
            Yout[mask] = pchip(Xout[mask])
             
    return Yout