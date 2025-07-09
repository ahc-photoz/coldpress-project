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
__version__ = "1.0.0"
__license__ = "GPLv3"
__copyright__ = "Copyright 2025, Antonio Hernán Caballero"

from .encode import encode_from_binned, encode_from_samples
from .decode import decode_quantiles, quantiles_to_binned, decode_to_binned 
from .stats import (
    measure_from_quantiles,
    zmode_from_quantiles,
    zmedian_from_quantiles,
    zmean_from_quantiles,
    zmean_err_from_quantiles,
    zrandom_from_quantiles,
    odds_from_quantiles,
    HPDCI_from_quantiles,
)
from .utils import reconstruct_pdf_from_quantiles, plot_from_quantiles

__all__ = [
    'encode_from_histograms',
    'encode_from_samples',
    'decode_to_histograms',
    'decode_to_cdf',
    'decode_quantiles',
    'measure_from_quantiles',
    'zmode_from_quantiles',
    'zmedian_from_quantiles',
    'zmean_from_quantiles',
    'zmean_err_from_quantiles',
    'zrandom_from_quantiles',
    'odds_from_quantiles',
    'HPDCI_from_quantiles',
    'reconstruct_pdf_from_quantiles',
    'plot_from_quantiles'
]