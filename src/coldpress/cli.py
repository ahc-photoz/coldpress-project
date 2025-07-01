#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
from astropy.io import fits
# Use relative imports as this file is inside the coldpress package
from . import (
    encode_from_histograms,
    decode_to_histograms,
    decode_quantiles,
    zmode_from_quantiles,
    zmedian_from_quantiles,
    zmean_from_quantiles,
    zrandom_from_quantiles,
    zmean_err_from_quantiles,
    odds_from_quantiles,
    HPDCI_from_quantiles)

# --- Logic for the 'encode' command ---
def encode_logic(args):
    """Contains all the logic from the original coldpress_encode.py script."""
    print(f"Opening input file: {args.input}")

    # Open the input file ONCE and do everything inside the 'with' block
    with fits.open(args.input) as h:
        # Get the header and all necessary columns while the file is open
        header = h[1].header
        PDF = h[1].data[args.pdfcol]
        orig_cols = list(h[1].columns)

        # Now, perform all the processing that depends on the input data
        zvector = np.linspace(args.zmin, args.zmax, PDF.shape[1])

        print("Compressing PDFs...")
        coldpress_PDF = encode_from_histograms(PDF, zvector, packetsize=args.length, ini_quantiles=args.length-9,
                                               validate=args.validate, tolerance=args.tolerance)

        # Create the new column for the output
        nints = args.length // 4
        new_col = fits.Column(name=args.out_pdfcol, format=f'{nints}J', array=coldpress_PDF)

        # Manipulate the list of columns
        # First, remove the output column if it already exists to prevent duplicates
        final_cols = [c for c in orig_cols if c.name != args.out_pdfcol]

        # Conditionally remove the original PDF column if --keep-orig is NOT set
        if not args.keep_orig:
            print(f"Removing original PDF column: '{args.pdfcol}'")
            final_cols = [c for c in final_cols if c.name != args.pdfcol]
        else:
            print(f"Keeping original PDF column: '{args.pdfcol}'")

        # Add the new compressed column to the final list
        final_cols.append(new_col)

        # Build the new HDU. This will now work because the file is still open
        # and astropy can read the data from the original columns.
        new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)

    # Add history and write the new HDU to the output file
    new_hdu.header.add_history(f'PDFs in column {args.pdfcol} cold-pressed as {args.out_pdfcol}')
    print(f"Writing compressed data to: {args.output}")
    new_hdu.writeto(args.output, overwrite=True)
    print('Done.')


# --- Logic for the 'decode' command ---
def decode_logic(args):
    """Contains all the logic from the original coldpress_decode.py script."""
    print(f"Opening input file: {args.input}")

    # Open the input file ONCE and perform all read operations inside
    with fits.open(args.input) as h:
        # Get all necessary data and metadata from the input file
        header = h[1].header
        coldpress_PDF = h[1].data[args.pdfcol]
        orig_cols = list(h[1].columns)

        # Define the z-axis for the output PDF
        zvector = np.arange(args.zmin, args.zmax + args.zstep/2, args.zstep)
        zvsize = len(zvector)

        # Decompress the PDFs. This uses data already in memory.
        print("Decompressing PDFs...")
        try:
            decoded_PDF = decode_to_histograms(coldpress_PDF, zvector, force_range=args.force_range)
        except ValueError as e:
            # Provide a user-friendly error message and exit
            print(f"Error: {e}", file=sys.stderr)
            print("Hint: Use the --force-range flag to proceed with truncation at your own risk.", file=sys.stderr)
            sys.exit(1)

        # Create the new FITS column for the decompressed data
        new_col = fits.Column(name=args.out_pdfcol, format=f'{zvsize}E', array=decoded_PDF)

        # Manipulate the list of columns
        final_cols = [c for c in orig_cols if c.name != args.out_pdfcol]
        final_cols.append(new_col)

        # Build the new HDU. This now works because the input file is still open.
        new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)
    
    # Add history and write the new HDU to the output file
    new_hdu.header.add_history(f'PDZs in column {args.pdfcol} extracted as {args.out_pdfcol}')
    print(f"Writing decompressed data to: {args.output}")
    new_hdu.writeto(args.output, overwrite=True)
    print('Done.')

# --- Logic for the 'measure' command ---
def measure_logic(args):
    """Logic to compute point estimates from compressed PDFs."""
    print(f"Opening input file: {args.input}")
    with fits.open(args.input) as h:
        data = h[1].data
        header = h[1].header
        original_columns = data.columns

    qcold = data[args.pdfcol]
    Nsources = qcold.shape[0]

    # Initialize dictionaries to hold new columns
    d = {}
    for k in ['Z_MODE','Z_MEAN','Z_MEDIAN',
              'Z_MODE_ERR','Z_MEAN_ERR',
              'Z_RANDOM',
              'Z_MIN_HPDCI68','Z_MAX_HPDCI68',
              'Z_MIN_HPDCI95','Z_MAX_HPDCI95',
              'ODDS_MODE','ODDS_MEAN']:
        d[k] = np.full(Nsources, np.nan, dtype=np.float32)
    d['Z_FLAGS'] = np.zeros(Nsources, dtype=np.int16)

    valid = np.any(qcold != 0, axis=1)
    d['Z_FLAGS'][~valid] = 1024 # Flag for no PDF info

    valid_indices = np.where(valid)[0]
    print(f"Calculating point estimates for {len(valid_indices)} valid sources...")
    for i in valid_indices:
        quantiles = decode_quantiles(qcold[i].tobytes())
        d['Z_MODE'][i] = zmode_from_quantiles(quantiles, width=0.005)
        d['Z_MEAN'][i] = zmean_from_quantiles(quantiles)
        d['Z_MEDIAN'][i] = zmedian_from_quantiles(quantiles)
        d['Z_RANDOM'][i] = zrandom_from_quantiles(quantiles)
        d['ODDS_MODE'][i] = odds_from_quantiles(quantiles, d['Z_MODE'][i])
        d['ODDS_MEAN'][i] = odds_from_quantiles(quantiles, d['Z_MEAN'][i])
        HPDCI68_mode_zmin, HPDCI68_mode_zmax = HPDCI_from_quantiles(quantiles, conf=0.68, zinside=d['Z_MODE'][i])
        d['Z_MODE_ERR'][i] = 0.5 * (HPDCI68_mode_zmax - HPDCI68_mode_zmin)
        d['Z_MEAN_ERR'][i] = zmean_err_from_quantiles(quantiles)
        HPDCI68_zmin, HPDCI68_zmax = HPDCI_from_quantiles(quantiles, conf=0.68, zinside=None)
        HPDCI95_zmin, HPDCI95_zmax = HPDCI_from_quantiles(quantiles, conf=0.95, zinside=None)
        d['Z_MIN_HPDCI68'][i] = HPDCI68_zmin
        d['Z_MAX_HPDCI68'][i] = HPDCI68_zmax
        d['Z_MIN_HPDCI95'][i] = HPDCI95_zmin
        d['Z_MAX_HPDCI95'][i] = HPDCI95_zmax

    # Create a list of all columns for the new table, starting with original ones
    final_cols = []
    new_col_names = d.keys()
    for col in original_columns:
        # Add original columns unless they are being replaced by a new one
        if col.name not in new_col_names:
            final_cols.append(col)

    # Add all the new columns we just calculated
    for name, array in d.items():
        if array.dtype == np.float32:
            format_str = 'E'
        elif array.dtype == np.int16:
            format_str = 'I'
        final_cols.append(fits.Column(name=name, format=format_str, array=array))

    new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)
    new_hdu.header['HISTORY'] = f'Computed point estimates from column: {args.pdfcol}'
    print(f"Writing point estimates to: {args.output}")
    new_hdu.writeto(args.output, overwrite=True)
    print('Done.')

# --- Main Entry Point and Parser Configuration ---
def main():
    parser = argparse.ArgumentParser(
        description='Compress or decompress PDFs in a FITS file using the coldpress algorithm.'
    )
    # This is the key: creating subparsers for the commands
    subparsers = parser.add_subparsers(dest='command', required=True,
                                       help='Available commands')

    # --- Parser for the "encode" command ---
    parser_encode = subparsers.add_parser('encode', help='Compress PDFs into coldpress format.')
    parser_encode.add_argument('input', metavar='input.fits', type=str, help='name of input FITS catalog')
    parser_encode.add_argument('output', metavar='output.fits', type=str, help='name of output FITS catalog')
    parser_encode.add_argument('--pdfcol', type=str, nargs='?', default='PDF', help='name of column containing PDFs to compress')
    parser_encode.add_argument('--out_pdfcol', type=str, nargs='?', default='coldpress_PDF', help='name of output column with cold-pressed PDF')
    parser_encode.add_argument('--zmin', type=float, nargs='?', default=0., help='zmin of input PDFs')
    parser_encode.add_argument('--zmax', type=float, required=True, help='zmax of input PDFs')
    parser_encode.add_argument('--length', type=int, nargs='?', default=80, help='length of cold-pressed PDF in bytes (must be multiple of 4)')
    parser_encode.add_argument('--validate', action='store_true', default=False, help='verify accuracy of recovered quantiles')
    parser_encode.add_argument('--tolerance', type=float, nargs='?', default=0.001, help='maximum shift in redshift allowed for quantiles')
    parser_encode.add_argument('--keep-orig', action='store_true', help='Keep the original PDF column in the output file.')
    parser_encode.set_defaults(func=encode_logic) # This links the 'encode' command to its function

    # --- Parser for the "decode" command ---
    parser_decode = subparsers.add_parser('decode', help='Decompress PDFs from coldpress format.')
    parser_decode.add_argument('input', type=str, help='name of input FITS catalog')
    parser_decode.add_argument('output', type=str, help='name of output FITS catalog')
    parser_decode.add_argument('--pdfcol', type=str, nargs='?', default='coldpress_PDF', help='name of column containing compressed PDFs')
    parser_decode.add_argument('--out_pdfcol', type=str, nargs='?', default='PDF_decoded', help='name of output column for extracted PDFs')
    parser_decode.add_argument('--zmin', type=float, default=0., help='zmin of output PDFs (default: 0.0)')
    parser_decode.add_argument('--zmax', type=float, required=True, help='zmax of output PDFs')
    parser_decode.add_argument('--zstep', type=float, required=True, help='zstep of output PDFs')
    parser_decode.add_argument('--force-range', action='store_true', help='force binning to the range given by [zmin,zmax] even if some PDFs are truncated')

    parser_decode.set_defaults(func=decode_logic) # This links the 'decode' command to its function

    # --- Parser for the "measure" command ---
    parser_measure = subparsers.add_parser('measure', help='Compute point estimates from compressed PDFs.')
    parser_measure.add_argument('input', type=str, help='name of input FITS table containing compressed PDFs')
    parser_measure.add_argument('output', type=str, help='name of output FITS table with new point estimate columns')
    parser_measure.add_argument('--pdfcol', type=str, nargs='?', default='coldpress_PDF',
                                help='name of column containing cold-pressed PDFs')
                                
    parser_measure.set_defaults(func=measure_logic)

    # Parse the arguments and call the function that was set by set_defaults
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
