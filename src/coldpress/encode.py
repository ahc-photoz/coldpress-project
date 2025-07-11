import numpy as np
import struct

def samples_to_quantiles(samples, Nquantiles=100):
    """ obtain a cumulative PDF by quantiles from random samples of the underlying PDF"""
    valid = np.isfinite(samples)
    zsorted = np.sort(samples[valid])
    targets = np.linspace(0.0, 1.0, Nquantiles) # target probability for quantiles
    return np.quantile(zsorted, targets, method='linear')


def binned_to_quantiles(z_grid, Pz, Nquantiles=100):
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


def encode_quantiles(quantiles, packetsize=80, validate=True, tolerance=0.001):
    """
    Encode an array of redshift values (quantiles of a cumulative PDF)
    into a byte array.
    Warning: if L+5 > packetsize, the packet will be truncated!
    Use validate=False to skip verification of the packet. This decreases cpu cost by ~10%.
    tolerance indicates the maximum shift allowed for the redshift of the quantiles.
    """
    from .decode import decode_quantiles # Local import to avoid circular dependency at module level

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

    max_big_gaps = (packetsize-(Nq+3)) // 2 

    gaps = np.sort(quantiles[1:-1]-quantiles[:-2]) 
    gapthreshold = gaps[-max_big_gaps-1] 
    eps_min = 0.00001*np.ceil(100000*gapthreshold/254)

    if eps_min > 0.00255:
        raise ValueError('Error: minimum usable epsilon={eps_min} is too large. Increase packet length or decrease number of quantiles.')

    eps_min2 = 0.00001*np.ceil(100000*gaps[-1]/(256**2 -1))

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
        qrecovered = decode_quantiles(packet)
        if len(qrecovered) != len(quantiles):
            raise ValueError('Error: packet decodes to wrong number of quantiles.')
        shift = quantiles[1:-1]-qrecovered[1:-1]
        if max(abs(shift)) > tolerance:
            raise ValueError('Error: shift in quantiles exceeds tolerance.')

    return L, bytes(packet)

def _batch_encode(data, ini_quantiles=71, packetsize=80, tolerance=None, validate=None, debug=False):
    """
    This function handles the batch encoding of multiple PDFs. Accepted input formats are:
     - PDF histograms
     - redshift samples 
    """    
    if packetsize % 4 != 0:
        raise ValueError(f"Error: packetsize must be a multiple of 4, but got {packetsize}.")

    if packetsize - ini_quantiles < 3:
        raise ValueError('Error: ini_quantiles must be at most packetsize-3')

    if data['format'] == 'PDF_histogram':
        valid = np.sum(data['PDF'],axis=1) > 0
    if data['format'] == 'samples':
        valid = np.all(np.isfinite(data['samples']), axis=1)

    int32col = np.zeros((len(valid),packetsize//4),dtype='>i4') 

    for i in range(len(valid)):
        if not valid[i]:
            continue

        Nquantiles = ini_quantiles
        lastgood = None
        while True:
            if data['format'] == 'PDF_histogram':    
                quantiles = binned_to_quantiles(data['zvector'],data['PDF'][i],Nquantiles=Nquantiles)
            if data['format'] == 'samples':
                quantiles = samples_to_quantiles(data['samples'][i],Nquantiles=Nquantiles)

            try:
                payload_length, packet = encode_quantiles(quantiles,packetsize=packetsize,tolerance=tolerance,validate=validate)
            except ValueError as e:
                if lastgood is not None:
                    packet = lastgood
                    break
                else:
                    Nquantiles -= 2
                    continue    

            if payload_length < packetsize-5:
                lastgood = packet
                Nquantiles += 2
                continue

            if payload_length == packetsize-5:
                break
         
        int32col[i] = np.frombuffer(packet, dtype='>i4')

    return int32col


def encode_from_binned(PDF, zvector, ini_quantiles=71, packetsize=80, tolerance=None, validate=None, debug=False):
    """
    Encode the PDFs given by the arrays PDF and zvector as a packet of bytes
    containing a compressed representation of the quantiles of the cumulative
    distribution function.
    """
    data = {'format': 'PDF_histogram', 'zvector': zvector, 'PDF': PDF}    
    return _batch_encode(data, ini_quantiles=ini_quantiles, packetsize=packetsize, tolerance=tolerance, validate=validate, debug=debug)

def encode_from_samples(samples, ini_quantiles=71, packetsize=80, tolerance=None, validate=None, debug=False):
    """
    Encode as quantiles the PDFs given as an array individual random redshift samples taken from the underlying PDF.
    """   
    data = {'format': 'samples', 'samples': samples}
    return _batch_encode(data, ini_quantiles=ini_quantiles, packetsize=packetsize, tolerance=tolerance, validate=validate, debug=debug)