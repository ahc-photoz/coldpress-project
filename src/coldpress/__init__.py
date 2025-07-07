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
    'samples_to_cdf',
    'pdf_to_cdf',
    'cdf_to_pdf',
    'encode_quantiles',
    'decode_quantiles',
    'encode_from_histograms',
    'encode_from_samples',
    'decode_to_histograms',
    'decode_to_cdf',
    'zmode_from_quantiles',
    'zmedian_from_quantiles',
    'zmean_from_quantiles',
    'zmean_err_from_quantiles',
    'zrandom_from_quantiles',
    'odds_from_quantiles',
    'HPDCI_from_quantiles',
    'reconstruct_pdf_from_quantiles'
]

import numpy as np
import struct
import sys
from scipy.interpolate import CubicSpline, PchipInterpolator
#from scipy.integrate import quad
#from numpy.polynomial.legendre import leggauss

def samples_to_cdf(samples, Nquantiles=100):
    """ obtain a cumulative PDF by quantiles from random samples of the underlying PDF"""
    valid = np.isfinite(samples)
    zsorted = np.sort(samples[valid])
    targets = np.linspace(0.0, 1.0, Nquantiles) # target probability for quantiles
    return np.quantiles(zsorted, targets, method='linear')


def pdf_to_cdf(z_grid, Pz, Nquantiles=100):
    """ obtain a cumulative PDF by quantiles from a binned PDF (histogram)"""

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
    
def cdf_to_pdf(z_quantiles, dz=None, z_min=None, z_max=None, zvector=None, method='linear'):
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
    if method == 'linear':
        F_grid = np.interp(edges, zq, Fq, left=0.0, right=1.0)
    if method == 'spline':
        F_grid = np.zeros(len(edges))
        F_grid[edges < zq[0]] = 0.
        F_grid[edges > zq[-1]] = 1.
        inside = (edges >= zq[0]) & (edges <= zq[-1])
        F_inside = monotone_natural_spline(edges[inside], zq, Fq)
        F_grid[inside] = F_inside/F_inside[-1] # numerical errors in integration make F_inside[-1] not exactly 1 
    pdf = F_grid[1:] - F_grid[:-1]
        
    pdf /= np.sum(pdf*dz)
        
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
        raise ValueError('Error: minimum usable epsilon={eps_min} is too large. Increase packet length or decrease number of quantiles.')

    # find the minimum value of epsilon that we would need to encode the largest gap in two bytes
    eps_min2 = 0.00001*np.ceil(100000*gaps[-1]/(256**2 -1))
    
    # choose the final epsilon value:    
    eps = np.max([0.00001,eps_min,eps_min2])    

    if int(np.round(eps*100000)) > 255:
        raise ValueError(f'Error: epsilon={eps} is too large for 1 byte encoding.')

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
        raise ValueError(f'Error: payload of length {L} does not fit in packet of size {packetsize}.')
        
    packet = bytearray(packetsize)
    packet[0] = int(np.round(eps*100000))
    packet[1:3] = struct.pack('<H', xmin_int)            
    packet[3:5] = struct.pack('<H', xmax_int)
    packet[5:5+L] = payload
    
    if validate: 
        # verify that the packet decodes correctly to the original number of quantiles 
        qrecovered = decode_quantiles(packet)
        if len(qrecovered) != len(quantiles):
            raise ValueError('Error: packet decodes to wrong number of quantiles.')
        # verify that the shift in quantile redshifts is within tolerance (excluding first and last quantiles)
        shift = quantiles[1:-1]-qrecovered[1:-1]
        if max(abs(shift)) > tolerance:
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
    
def batch_encode(data, ini_quantiles=71, packetsize=80, tolerance=None, validate=None, debug=False):
    """
    This function handles the batch encoding of multiple PDFs. Accepted input formats are:
     - PDF histograms
     - redshift samples 
    The data dictionary must contain the keywords:
      - format: type of PDF: 'PDF_histogram' or 'samples'
      - the data itself (PDF and zvector arrays for 'PDF_histogram', z samples array for 'samples') 
    """    
    if packetsize % 4 != 0:
        raise ValueError(f"Error: packetsize must be a multiple of 4, but got {packetsize}.")

    if packetsize - ini_quantiles < 3:
        raise ValueError('Error: ini_quantiles must be at most packetsize-3')
        
    # filter out invalid PDFs. Their encoded representation will be all zeros.
    if data['format'] == 'PDF_histogram':
        valid = np.sum(data['PDF'],axis=1) > 0
    if data['format'] == 'samples':
        valid = np.isfinite(data['samples'],axis=1)
            
    # prepare array to contain encoded PDFs        
    int32col = np.zeros((len(valid),packetsize//4),dtype='>i4') # each 4 bytes will be encoded as a 32-bit signed integer (big-endian)
      
    # iterate sources    
    for i in range(len(valid)):
        if not valid[i]:
            continue
    
        # initialize loop for selection of optimal number of quantiles
        Nquantiles = ini_quantiles
        lastgood = None
        while True:
            if debug:
                trycount += 1
                if trycount > 10:
                    print('We got stuck in infinite loop.')
                    import code
                    code.interact(local=locals())
                print(f'Trying: {Nquantiles} quantiles')
            
            # get quantiles for current value of Nquantiles
            if data['format'] == 'PDF_histogram':    
                quantiles = pdf_to_cdf(data['zvector'],data['PDF'][i],Nquantiles=Nquantiles) # compute quantiles from PDZ
            if data['format'] == 'samples':
                quantiles = samples_to_cdf(data['samples'][i],Nquantiles=Nquantiles) # compute quantiles from samples
                    
            # encode and validate results    
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
                        if 'payload of length' in msg:
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
 
    
def encode_from_histograms(PDF, zvector, ini_quantiles=71, packetsize=80, tolerance=None, validate=None, debug=False):
    """
    Encode the PDFs given by the arrays PDF and zvector as a packet of bytes
    containing a compressed representation of the quantiles of the cumulative
    distribution function.
    """
    data = {'format': 'PDF_histogram', 'zvector': zvector, 'PDF': PDF}    
    return batch_encode(data, ini_quantiles=ini_quantiles, packetsize=packetsize, tolerance=tolerance, validate=validate, debug=debug)
 
def encode_from_samples(samples, ini_quantiles=71, packetsize=80, tolerance=None, validate=None, debug=False):
    """
    Encode as quantiles the PDFs given as an array individual random redshift samples taken from the underlying PDF.
    """   
    data = {'format': 'samples', 'samples': samples}
    return batch_encode(data, ini_quantiles=ini_quantiles, packetsize=packetsize, tolerance=tolerance, validate=validate, debug=debug)     

def decode_to_histograms(int32col, zvector, force_range=False):
    """
    Decode the PDFs to a histogram PDF with bins defined by zvector.
    In the case of P(z) > 0 outside the range defined by zvector, an exception is raised.
    Set force_range = True if you prefer to truncate the PDF instead of raising an
    exception.
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

def odds_from_quantiles(quantiles, zcenter, odds_window=0.03):
    """Obtain the odds parameter defined as the integral of P(z) in the interval
       zcenter +/- odds_window*(1+zcenter), where zcenter could be zmode, zmean, zmedian, etc
    """
    knots = np.linspace(0, 1, len(quantiles))
    zbinmin = zcenter - odds_window*(1+zcenter)
    zbinmax = zcenter + odds_window*(1+zcenter)
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
    z_steps_extended = np.concatenate(([z_steps[0]-0.001],z_steps,[z_steps[-1]+0.001]))
    p_steps_extended = np.concatenate(([0],p_steps,[0]))
    return z_steps_extended, p_steps_extended
    
def monotone_natural_spline(Xout, X, Y):
    """
    Interpolate (X, Y) with a natural cubic spline, then correct any
    non-monotonic intervals using PCHIP interpolation.
    
    Parameters
    ----------
    X : array_like, shape (N,)
        Strictly increasing abscissas of the knots.
    Y : array_like, shape (N,)
        Ordinates of the knots.
    Xout : array_like
        Points at which to evaluate the corrected monotone spline.
    
    Returns
    -------
    Yout : ndarray
        Interpolated values at Xout, guaranteed monotonic on each segment.
    """
    # Fit natural and PCHIP splines
    spline = CubicSpline(X, Y, bc_type='natural')
    pchip = PchipInterpolator(X, Y)
    
    # natural is the default interpolation
    Yout = spline(Xout)
    
    # Compute first derivative at knots and in the output grid
    Yp  = spline(X, 1)
    Ypp = spline(X, 2)
    dYout = spline(Xout, 1)
        
    # Determine which interval each Xout belongs to
    idx = np.searchsorted(X, Xout) - 1
    idx = np.clip(idx, 0, len(X)-2)
    
    # Check monotonicity and switch to PCHIP if violated
    for i in range(len(X)-1):
        # intervals with negative slope OR non-positive knot slopes
        mask = (idx == i) & ((dYout < 0) | (Yp[i] <= 0) | (Yp[i+1] <= 0))
        if np.any(mask):
            mask = (idx == i)
            Yout[mask] = pchip(Xout[mask]) # intermediate quantile, use PCHIP
             
    return Yout
    

  