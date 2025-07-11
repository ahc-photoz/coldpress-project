import numpy as np
import struct
import sys

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

def quantiles_to_binned(z_quantiles, dz=None, Nbins=None, z_min=None, z_max=None, zvector=None, method='linear', force_range=False):
    """
    Given quantile locations, return a PDF on a regular grid.
    The grid can be defined by zvector, Nbins, or dz.
    """
    if method == 'spline':
        from .utils import _monotone_natural_spline

    # --- New and updated sanity checks for conflicting arguments ---
    if force_range and zvector is None and z_min is None and z_max is None:
        raise ValueError("force_range=True is only meaningful when an explicit range is provided via 'zvector' or 'z_min'/'z_max'.")
    if zvector is not None and (dz is not None or Nbins is not None):
        raise ValueError("Cannot specify 'dz' or 'Nbins' when 'zvector' is provided.")
    if dz is not None and Nbins is not None:
        raise ValueError("Cannot specify both 'dz' and 'Nbins' simultaneously.")
        
    zq = np.asarray(z_quantiles)
    
    # Grid creation logic
    if zvector is not None:
        z_grid = zvector
    else:
        # Determine range boundaries, inferring from data if not provided
        range_min = zq[0] if z_min is None else z_min
        range_max = zq[-1] if z_max is None else z_max

        if Nbins is not None:
            # Handle case of a delta function to avoid a zero-width range
            if range_min == range_max:
                range_min -= 0.01
                range_max += 0.01
            z_grid = np.linspace(range_min, range_max, Nbins)
        elif dz is not None:
            # Snap auto-calculated range to the dz grid if z_min/z_max not provided
            if z_min is None: range_min = dz * np.floor(zq[0] / dz)
            if z_max is None: range_max = dz * np.ceil(zq[-1] / dz)
            z_grid = np.arange(range_min, range_max + dz/2, dz)
        else:
            raise ValueError("Must provide one of 'zvector', 'Nbins', or 'dz'.")

    # Perform the range check on the final z_grid
    zmin_q, zmax_q = zq[0], zq[-1]
    zmin_grid, zmax_grid = z_grid[0], z_grid[-1]

    if zmin_q < zmin_grid or zmax_q > zmax_grid:
        if not force_range:
            raise ValueError(f"Decoded redshift range [{zmin_q:.3f}, {zmax_q:.3f}] "
                             f"exceeds the target grid range [{zmin_grid:.3f}, {zmax_grid:.3f}]. "
                             "Use force_range=True to override.")
        print(f"Warning: PDF range [{zmin_q:.3f}, {zmax_q:.3f}] truncated to grid range [{zmin_grid:.3f}, {zmax_grid:.3f}].", file=sys.stderr)
    
    # Proceed with calculations
    M = len(zq)
    Fq = np.linspace(0.0, 1.0, M)
    dz_eff = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 0
    
    edges = np.empty(len(z_grid)+1, dtype=float)
    edges[0] = z_grid[0] - dz_eff/2
    edges[1:] = z_grid + dz_eff/2

    if method == 'linear':
        F_grid = np.interp(edges, zq, Fq, left=0.0, right=1.0)
    elif method == 'spline':
        F_grid = np.zeros(len(edges))
        F_grid[edges < zq[0]] = 0.
        F_grid[edges > zq[-1]] = 1.
        inside = (edges >= zq[0]) & (edges <= zq[-1])
        F_inside = _monotone_natural_spline(edges[inside], zq, Fq)
        if F_inside.size > 0 and F_inside[-1] > 0:
            F_grid[inside] = F_inside / F_inside[-1]
    else:
        raise ValueError(f"Unknown interpolation method '{method}'. Choose 'linear' or 'spline'.")
        
    pdf = F_grid[1:] - F_grid[:-1]
        
    pdf_sum = np.sum(pdf * dz_eff)
    if pdf_sum > 0:
        pdf /= pdf_sum
        
    # Return both the grid and the PDF if the grid was generated internally
    if zvector is None:
        return z_grid, pdf
    else:
        return pdf
        
def quantiles_to_samples(z_quantiles, Nsamples=100, method='linear', ss_factor=10):
    """
    Given quantile locations z_quantiles (monotonic array of length M),
    return Nsamples random samples of the PDF. 
    """      
    if method == 'spline':
        from .utils import _monotone_natural_spline
  
    zq = np.asarray(z_quantiles)
    M = len(zq)
    Fq = np.linspace(0.0, 1.0, M)
   
    u = np.random.uniform(0, 1, size=Nsamples)

    if method == 'linear':
        samples = np.interp(u, Fq, zq)
    if method == 'spline':
        samples = _monotone_natural_spline(u, Fq, zq)
            
    return samples
    
def decode_to_binned(int32col, zvector, force_range=False, method='linear'):
    """
    Decode compressed PDFs to binned PDFs with bins defined by zvector.
    """
    Nsources = int32col.shape[0]
    PDF = np.zeros((Nsources,len(zvector)),dtype=np.float32)
    
    for i in range(Nsources):
        if np.any(int32col[i] != 0):
            packet = int32col[i].tobytes()
            qrecovered = decode_quantiles(packet)

            try:
                PDF[i] = quantiles_to_binned(qrecovered, zvector=zvector, method=method, force_range=force_range)
            except ValueError as e:
                raise ValueError(f"Source {i}: {e}") from e

    return PDF
    
def decode_to_samples(int32col, Nsamples=None, method='linear'):
    """
    Decode compressed PDFs to array of random samples from the PDF.
    """
    Nsources = int32col.shape[0]
    samples = np.full((Nsources,Nsamples),np.nan,dtype=np.float32)
    
    for i in range(Nsources):
        if np.any(int32col[i] != 0):
            packet = int32col[i].tobytes()
            qrecovered = decode_quantiles(packet)

            try:
                samples[i] = quantiles_to_samples(qrecovered, Nsamples=Nsamples, method=method)
            except ValueError as e:
                raise ValueError(f"Source {i}: {e}") from e

    return PDF
    
