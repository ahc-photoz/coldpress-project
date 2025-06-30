"""
ColdPress: efficient compression for redshift probability density functions

This module provides functions to encode and decode redshift probability density
functions (PDFs) into a compact, fixed-size byte representation suitable for
efficient storage in databases. 
It works by computing the redshifts {z_i} that correspond to the quantiles {q_i}
of the cumulative distribution function (CDF) and encoding the differences
∆i = z_i - z_{i-1} using (mostly) a single byte.

Citation
--------
If you use ColdPress in your research, please cite:
  Hernán-Caballero, A. 2025, Research Notes of the AAS,  <Volume>, <ID>.
  DOI: <Your DOI Here>
  arXiv: <Your arXiv ID Here>
"""

__author__ = "Antonio Hernán Caballero"
__email__ = "ahernan@cefca.es"
__version__ = "0.1.0"  
__license__ = "GPLv3"
__copyright__ = "Copyright 2025, Antonio Hernán Caballero"

__all__ = [
    'pdf_to_cdf',
    'cdf_to_pdf',
    'encode_quantiles',
    'decode_quantiles',
    'encode_from_histograms',
    'decode_to_histograms',
    'decode_to_cdf',
    'zmode_from_quantiles',
    'zmedian_from_quantiles',
    'zmean_from_quantiles',
    'zmean_err_from_quantiles',
    'zrandom_from_quantiles',
    'odds_from_quantiles',
    'HPDCI_from_quantiles',
    'reconstruct_pdf_from_quantiles',
    'reconstruct_pdf_variational',
]

import numpy as np
import struct
import sys

# Make scipy an optional import for non-plotting functions
try:
    from scipy.interpolate import CubicSpline
except ImportError:
    CubicSpline = None


def pdf_to_cdf(z_grid, Pz, Nquantiles=100):
    """ obtain a cumulative PDF by quantiles from a PDF """

    nonzero = np.where(Pz > 0)[0]
       
    if len(nonzero) == 1: # special case, Pz is a single delta 
        dz = z_grid[1]-z_grid[0]
        qs = np.linspace(z_grid[nonzero[0]]-dz/2,z_grid[nonzero[0]]+dz/2,Nquantiles)
        return qs

    # remove trailing and leading zeros in Pz    
    imin = np.min(nonzero)
    imax = np.max(nonzero)
    z_grid = z_grid[imin:imax+1]
    Pz = Pz[imin:imax+1]
    
    # compute edges of bins
    dz = z_grid[1] - z_grid[0]
    edges = np.empty(len(z_grid)+1, dtype=float)
    edges[0] = z_grid[0] - dz/2
    edges[1:] = z_grid + dz/2

    # Build CDF at edges: cdf_edges[0]=0; then cdf_edges[i+1] = cdf_edges[i] + p[i]*dz
    cdf_edges = np.empty_like(edges)
    cdf_edges[0] = 0.0
    cdf_edges[1:] = np.cumsum(Pz * dz)

    # Normalize to ensure the final CDF is exactly 1
    cdf_edges /= cdf_edges[-1]
    
    # compute quantiles
    targets = np.linspace(0.0, 1.0, Nquantiles) # target probability for quantiles
    qs = np.interp(targets, cdf_edges, edges)
    
    return qs
    
def cdf_to_pdf(z_quantiles, dz=None, z_min=None, z_max=None, zvector=None):
    """
    Given quantile locations z_quantiles (monotonic array of length M),
    where CDF F(z_quantiles[j]) = j/(M-1),
    return a PDF sampled on z = z_min, zmin+dz, ..., up to z_max.
    
    Parameters
    ----------
    z_quantiles : array_like, shape (M,)
        Monotonic array of quantile redshift values from 0% to 100%.
    dz : float
        Desired regular grid spacing for PDF.
    z_max : float, optional
        Upper limit of the PDF grid (default 1.5).
    
    Returns
    -------
    z_grid : ndarray
        1D array of grid points [0, dz, 2dz, ..., <= z_max].
    pdf : ndarray
        PDF values at each z_grid (same length as z_grid).
    """
    zq = np.asarray(z_quantiles)
    M = len(zq)
    # Corresponding CDF values
    Fq = np.linspace(0.0, 1.0, M)
    
    # Regular grid
    if zvector is not None:
        z_grid = zvector
        dz = z_grid[1] - z_grid[0]
    else:    
        z_grid = np.arange(z_min, z_max + dz/2, dz)
    
    # obtain edges of histogram
    edges = np.empty(len(z_grid)+1, dtype=float)
    edges[0] = z_grid[0] - dz/2
    edges[1:] = z_grid + dz/2

    # 1) Interpolate the CDF onto the regular grid
    #    Values outside [zq[0], zq[-1]] are clamped to 0 or 1
    F_grid = np.interp(edges, zq, Fq, left=0.0, right=1.0)
    
    pdf = F_grid[1:] - F_grid[:-1]
        
    if zvector is not None:
        return pdf
    else:
        return z_grid, pdf

def encode_quantiles(quantiles, packetsize=80, validate=True, tolerance=0.001):
    """
    Encode an array of redshift values (quantiles of a cumulative PDF)
    into a byte array.
    Warning: if L+5 > packetsize, the packet will be truncated!
    Use validate=False to skip verification of the packet. This decreases cpu cost by ~10%.
    tolerance indicates the maximum shift allowed for the redshift of the quantiles.
    """
    
    Nq = len(quantiles)
    if Nq > packetsize-3:
        raise ValueError(f'Error: cannot fit {Nq} quantiles in an {packetsize}-bytes packet')
        
    zmin, zmax = quantiles[0], quantiles[-1]

    # encode endpoints as uint16
    xmin_int = int(np.floor((zmin+0.01) / 0.0002))
    xmax_int = int(np.ceil((zmax+0.01) / 0.0002))

    # recompute true zmin/zmax & epsilon
    zmin_rec = xmin_int * 0.0002 - 0.01
    zmax_rec = xmax_int * 0.0002 - 0.01
    
    # compute the redshift quantization step, epsilon
    # it needs to meet 3 conditions:
    # a) larger than the resolution of its internal integer representation: epsilon > 1.e-5
    # b) large enough to encode the bigest gap between quantiles in two bytes: epsilon > max_gap/(256**2 -1)
    # c) small enough that its internal representation fits in one byte: epsilon <= 255 * 1.e-5
    #
    # we want it as small as possible for accuracy in regions of high density,
    # but on the other hand, in order to fit in the requested packet size there is a 
    # maximum number of long jumps (encoded in 2 bytes + 1 marker) we can afford.
    
    max_big_gaps = (packetsize-(Nq+3)) // 2 # maximum number of big gaps we can afford for given packetsize and number of quantiles
    
    # find the minimum value of epsilon that we would need to encode all but max_big_gaps
    # jumps in one byte each  
    gaps = np.sort(quantiles[1:-1]-quantiles[:-2]) # the last quantile is stored in the header, so its gap is irrelevant
    gapthreshold = gaps[-max_big_gaps-1] 
    eps_min = 0.00001*np.ceil(100000*gapthreshold/254)
    
    # check if eps_min is too large to encode in one byte
    if eps_min > 0.00255:
        raise ValueError('Error: epsilon value is too large. Need to increase packet length or decrease number of quantiles.')

    # find the minimum value of epsilon that we would need to encode the largest gap in two bytes
    eps_min2 = 0.00001*np.ceil(100000*gaps[-1]/(256**2 -1))
    
    # choose the final epsilon value:    
    eps = np.max([0.00001,eps_min,eps_min2])    

    if int(np.round(eps*100000)) > 255:
        print('Error: epsilon value is too large, but it was not catched before!')
        import code
        code.interact(local=locals())

    # build payload from the *interior* N-2 quantiles
    payload = bytearray()
    prev = zmin_rec
    for z in quantiles[1:-1]:
        d = int(round((z - prev) / eps))
        if 0 <= d <= 254:
            payload.append(d)
        else:
            payload.append(255)
            payload += struct.pack('>H', d)
        prev = prev + d * eps

    L = len(payload)
    if L > packetsize-5: 
        raise ValueError('Error: payload is too long. Decrease number of quantiles.')
        
    packet = bytearray(packetsize)
    packet[0] = int(np.round(eps*100000))
    packet[1:3] = struct.pack('<H', xmin_int)            
    packet[3:5] = struct.pack('<H', xmax_int)
    packet[5:5+L] = payload
    
    if validate: # verify that the packet decodes correctly to the original quantiles 
        qrecovered = decode_quantiles(packet)
        if len(qrecovered) != len(quantiles):
            raise ValueError('Error: packet decodes to wrong number of quantiles.')
        diff = quantiles-qrecovered
        if max(abs(diff)) > tolerance:
            raise ValueError('Error: shift in quantiles exceeds tolerance.')
        
    return L, bytes(packet)
    
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
    
def encode_from_histograms(PDF, zvector, ini_quantiles=71, packetsize=80, tolerance=None, validate=None, debug=False):
    """
    Encode the PDFs given by the arrays PDF and zvector as a packet of bytes
    containing a compressed representation of the quantiles of the cumulative
    distribution function.
    """
    if packetsize % 4 != 0:
        raise ValueError(f"Error: packetsize must be a multiple of 4, but got {packetsize}.")

    if packetsize - ini_quantiles < 3:
        raise ValueError('Error: ini_quantiles must be at most packetsize-3')
        
    Nsources = PDF.shape[0]
    int32col = np.zeros((Nsources,packetsize//4),dtype='>i4') # each 4 bytes will be encoded as a 32-bit signed integer (big-endian)
    
    for i in range(Nsources):
        if np.sum(PDF[i]) > 0: 
            Nquantiles = ini_quantiles
            lastgood = None
            trycount = 0
            while True:
                if debug:
                    trycount += 1
                    if trycount > 10:
                        print('We got stuck in infinite loop.')
                        import code
                        code.interact(local=locals())
                    
                    print(f'Trying: {Nquantiles} quantiles')
                quantiles = pdf_to_cdf(zvector,PDF[i],Nquantiles=Nquantiles) # compute quantiles from PDZ
                try:
                    payload_length, packet = encode_quantiles(quantiles,packetsize=packetsize,tolerance=tolerance,validate=validate) # encode quantiles as byte array                     
                except ValueError as e:
                    if lastgood is not None: # we already tried with fewer quantiles and it worked. We are done.
                        if debug:
                            print('Warning: we tried to increase number of quantiles and it didnt work. We are good with previous solution.')
                        packet = lastgood
                        break
                    else: # lets try decreasing number of quantiles  
                        if debug:  
                            msg = str(e)
                            if 'tolerance' in msg:
                                print('Warning: some quantile shifted beyond tolerance. Retrying with fewer quantiles...')
                            if 'epsilon' in msg:
                                print(f'Warning: epsilon is too large with {Nquantiles}. Retrying with fewer quantiles...')
                            if 'payload is too long' in msg:
                                print(f'Warning: payload too long with {Nquantiles}. Retrying with fewer quantiles...')    
                        Nquantiles -= 2
                        continue    
                    
                if payload_length < packetsize-5: # try again with more quantiles
                    lastgood = packet
                    Nquantiles += 2
                    if debug:
                        print('Warning: there are trailing zeros in packet. Trying to squeeze more quantiles...')
                    continue
                        
                if payload_length == packetsize-5: # payload completely fills the packet, we are done
                    if debug:
                        print('Warning: payload is full. we are good!')
                    break
             
            int32col[i] = np.frombuffer(packet, dtype='>i4')

    return int32col

def decode_to_histograms(int32col, zvector, force_range=False):
    """
    Decode the PDFs that have been encoded with the encode_from_histograms() method
    """
    Nsources = int32col.shape[0]
    PDF = np.zeros((Nsources,len(zvector)),dtype=np.float32)
    
    for i in range(Nsources):
        if np.any(int32col[i] != 0):
            packet = int32col[i].tobytes()
            qrecovered = decode_quantiles(packet) # recover quantiles from byte array

            # Safety check to prevent PDF truncation
            zmin_q, zmax_q = qrecovered[0], qrecovered[-1]
            zmin_vec, zmax_vec = zvector[0], zvector[-1]

            if zmin_q < zmin_vec or zmax_q > zmax_vec:
                if not force_range:
                    raise ValueError(f"Source {i}: Decoded redshift range [{zmin_q:.3f}, {zmax_q:.3f}] "
                                     f"exceeds the target zvector range [{zmin_vec:.3f}, {zmax_vec:.3f}]. "
                                     "PDF would be truncated. Use force_range=True to override.")
                print(f"Warning: Source {i}: PDF range [{zmin_q:.3f}, {zmax_q:.3f}] truncated to zvector range [{zmin_vec:.3f}, {zmax_vec:.3f}].", file=sys.stderr)

            PDF[i] = cdf_to_pdf(qrecovered, zvector=zvector) # recover PDZ from quantiles

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
            qrecovered = decode_quantiles(packet) # recover quantiles from byte array
            qs = np.linspace(0,1,len(qrecovered))
            cdf[i] = np.interp(zv, qrecovered, qs, left=0.0, right=1.0) 
    return cdf

def zmode_from_quantiles(quantiles, width=0.005):
    """More robust approach against effects of z quantization in very narrow peaks.
    Find the interval of width 'width' that maximizes the integral of P(z)dz in it.
    Such interval always has to start or end in a knot.
    Returns the center of such interval.
    """
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
    """Obtain the stddev of the mean redshift of the probability distribution (integral of (z-zmean)^2 * P(z))"""
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

def odds_from_quantiles(quantiles, zbest, zwidth=0.03):
    """Obtain the odds parameter defined as the integral of P(z) in the interval
       zbest +/- zwidth(1+zbest), where zbest could be zmode, zmean, zmedian, etc
    """
    knots = np.linspace(0, 1, len(quantiles))
    zbinmin = zbest - zwidth*(1+zbest)
    zbinmax = zbest + zwidth*(1+zbest)
    qz = np.interp([zbinmin,zbinmax],quantiles,knots,left=0,right=1)
    return qz[1]-qz[0]

def HPDCI_from_quantiles(quantiles, conf=0.68, zinside=None):
    """Obtain zmin and zmax corresponding to the highest probability density confidence
    interval at the given confidence level.
    If a redshift is provided with the zinside keyword, only confidence intervals
    that contain zinside will be considered
    """
    knots = np.linspace(0, 1, len(quantiles))
    knot_interval = knots[1]-knots[0] # interval (in quantiles) between consecutive knots of the cumulative PDF

    # if zinside is provided, the valid search range is not the entire cumulative
    # PDF but only the range within qin +/- conf, where qin is the quantile corresponding
    # to the redshift zinside.
    if zinside is not None:
        qin = np.interp(zinside,quantiles,knots,left=0,right=1) # quantile for zinside
        qmin = max([0,qin-conf]) # first quantile of the search range
        qmax = min([1,qin+conf]) # last quantile of search range
    else: # no restriction
        qmin = 0
        qmax = 1

    start = np.arange(qmin,qmax-conf,0.2*knot_interval)
    end = np.arange(qmin+conf,qmax,0.2*knot_interval)
    
    if len(start) == 0 or len(end) == 0:
        # This can happen if the PDF is extremely narrow (like a delta function)
        # In this case, the 68% interval is effectively zero width around the center.
        z_center = np.interp(0.5, knots, quantiles)
        return (z_center, z_center)
        
    zmin = np.interp(start,knots,quantiles)
    zmax = np.interp(end,knots,quantiles)
    dz = zmax-zmin
    best = np.argmin(dz)
    return (zmin[best], zmax[best])

def reconstruct_pdf_from_quantiles(quantiles):
    """
    Reconstructs a stepwise PDF from its quantiles.

    The probability density P(z) is assumed to be constant between
    two consecutive quantiles, z_i and z_{i+1}.

    Returns
    -------
    z_steps : ndarray
        The redshift values for the edges of the steps (for plotting).
    p_steps : ndarray
        The probability density values for each step.
    """
    Nq = len(quantiles)
    p_steps = (1.0 / (Nq - 1)) / (quantiles[1:] - quantiles[:-1])
    z_steps = quantiles
    return z_steps, p_steps

def reconstruct_pdf_variational(quantiles, n_points=500, densify_threshold_factor=5.0, pad_factor=0.1):
    """
    Reconstructs a smooth PDF using a "guided" variational approach.

    This method creates a smooth, physically-plausible PDF by:
    1. Padding the boundaries to enforce a PDF of zero outside the data range.
    2. Densifying large gaps between quantiles with linear guide points.
    3. Fitting a minimum-curvature (natural cubic) spline to the guided points.

    Returns
    -------
    z_fine : ndarray
        A fine grid of redshift values for plotting.
    pdf_fine : ndarray
        The smooth probability density values on the z_fine grid.
    is_monotonic : bool
        True if the resulting PDF is non-negative everywhere, False otherwise.
    """
    if CubicSpline is None:
        raise ImportError("scipy is required for the variational plotting method.")

    # --- Step 1: Boundary Padding ---
    z_coords = list(quantiles)
    cdf_coords = list(np.linspace(0.0, 1.0, len(quantiles)))

    # Pad the beginning to enforce zero slope
    start_epsilon = pad_factor * (z_coords[1] - z_coords[0])
    z_coords.insert(0, z_coords[0] - start_epsilon)
    cdf_coords.insert(0, 0.0)

    # Pad the end to enforce zero slope
    end_epsilon = pad_factor * (z_coords[-1] - z_coords[-2])
    z_coords.append(z_coords[-1] + end_epsilon)
    cdf_coords.append(1.0)

    # --- Step 2: Densification ---
    z_dense = []
    cdf_dense = []
    
    dz = np.diff(z_coords)
    # Use a robust threshold; ignore zero-width gaps if any
    median_dz = np.median(dz[dz > 0])
    if median_dz == 0: # Handle case of all quantiles being the same
        median_dz = 1e-9
    threshold = densify_threshold_factor * median_dz

    for i in range(len(z_coords) - 1):
        z_dense.append(z_coords[i])
        cdf_dense.append(cdf_coords[i])
        gap = z_coords[i+1] - z_coords[i]
        if gap > threshold:
            # Add a guide point in the middle of large gaps
            z_mid = (z_coords[i] + z_coords[i+1]) / 2.0
            cdf_mid = (cdf_coords[i] + cdf_coords[i+1]) / 2.0
            z_dense.append(z_mid)
            cdf_dense.append(cdf_mid)
    
    z_dense.append(z_coords[-1])
    cdf_dense.append(cdf_coords[-1])

    # --- Step 3: Minimum-Energy Spline Fitting ---
    cdf_spline = CubicSpline(z_dense, cdf_dense, bc_type='natural')

    # --- Step 4: Differentiation and Verification ---
    z_fine = np.linspace(quantiles[0], quantiles[-1], n_points)
    pdf_fine = cdf_spline(z_fine, nu=1)

    is_monotonic = np.all(pdf_fine >= -1e-9)
    
    # Return the PDF, but do not clip to zero. Let the caller decide.
    return z_fine, pdf_fine, is_monotonic
