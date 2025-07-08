#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
from astropy.io import fits
# Use relative imports as this file is inside the coldpress package
from . import (
    cdf_to_pdf,
    encode_from_histograms,
    decode_to_histograms,
    decode_quantiles,
    zmode_from_quantiles,
    zmedian_from_quantiles,
    zmean_from_quantiles,
    zrandom_from_quantiles,
    zmean_err_from_quantiles,
    odds_from_quantiles,
    HPDCI_from_quantiles,
    reconstruct_pdf_from_quantiles)

# --- Logic for the 'encode' command ---
def encode_logic(args):
    import time
    """Contains all the logic from the original coldpress_encode.py script."""
    if args.length % 4 != 0:
        print(f"Error: Packet length (--length) must be a multiple of 4, but got {args.length}.", file=sys.stderr)
        sys.exit(1)

    print(f"Opening input file: {args.input}")

    # Open the input file ONCE and do everything inside the 'with' block
    with fits.open(args.input) as h:
        # Get the header and all necessary columns while the file is open
        header = h[1].header
        if args.binned is None:
            samples = h[1].data[args.samples]
            print(f"Generating quantiles from random redshift samples and compressing into {args.length}-byte packets...")
        else:
            PDF = h[1].data[args.binned]
            zvector = np.linspace(args.zmin, args.zmax, PDF.shape[1])
            cratio = PDF.shape[1]*PDF.itemsize/args.length
            print(f"Compressing PDFs into {args.length}-byte packets (compression ratio: {cratio:.2f})...")

        orig_cols = list(h[1].columns)

        # Now, perform all the processing that depends on the input data        
        start = time.process_time()
        if args.binned is None:
            coldpress_PDF = encode_from_samples(samples, packetsize=args.length, ini_quantiles=args.length-9,
                                                   validate=args.validate, tolerance=args.tolerance)        
        else:
            coldpress_PDF = encode_from_histograms(PDF, zvector, packetsize=args.length, ini_quantiles=args.length-9,
                                                   validate=args.validate, tolerance=args.tolerance)
        end = time.process_time()
        cpu_seconds = end - start
        print(f"{coldpress_PDF.shape[0]} PDFs cold-pressed in {cpu_seconds:.6f} CPU seconds")

        # Create the new column for the output
        nints = args.length // 4
        new_col = fits.Column(name=args.out_encoded, format=f'{nints}J', array=coldpress_PDF)

        # Manipulate the list of columns
        # First, remove the output column if it already exists to prevent duplicates
        final_cols = [c for c in orig_cols if c.name != args.out_encoded]

        # Conditionally remove the original PDF column if --keep-orig is NOT set
        if args.binned is None:
            orig_column = args.samples
        else:
            orig_column = args.binned  
              
        if args.keep_orig:
            print(f"Including column '{orig_column}' in output FITS table.")
        else:
            final_cols = [c for c in final_cols if c.name != orig_column]
            print(f"Excluding column '{orig_column}' from output FITS table.")

        # Add the new compressed column to the final list
        final_cols.append(new_col)

        # Build the new HDU. This will now work because the file is still open
        # and astropy can read the data from the original columns.
        new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)

    # Add history and write the new HDU to the output file
    if args.binned is None:
        new_hdu.header.add_history(f'PDFs from samples in column {args.samples} cold-pressed as {args.out_encoded}')
    else:       
        new_hdu.header.add_history(f'Binned PDFs in column {args.binned} cold-pressed as {args.out_encoded}')
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
        coldpress_PDF = h[1].data[args.encoded]
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
        new_col = fits.Column(name=args.out_binned, format=f'{zvsize}E', array=decoded_PDF)

        # Manipulate the list of columns
        final_cols = [c for c in orig_cols if c.name != args.out_binned]
        final_cols.append(new_col)

        # Build the new HDU. This now works because the input file is still open.
        new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)
    
    # Add history and write the new HDU to the output file
    new_hdu.header.add_history(f'PDFs in column {args.encoded} extracted as {args.out_binned}')
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

    qcold = data[args.encoded]
    Nsources = qcold.shape[0]

    # Define all possible quantities and their dependencies.
    # FITS column names are case-insensitive, but we'll use uppercase for consistency.
    all_quantities = {
        'Z_MODE', 'Z_MEAN', 'Z_MEDIAN', 'Z_RANDOM', 'Z_MODE_ERR', 'Z_MEAN_ERR',
        'ODDS_MODE', 'ODDS_MEAN', 'Z_MIN_HPDCI68', 'Z_MAX_HPDCI68',
        'Z_MIN_HPDCI95', 'Z_MAX_HPDCI95'
    }
    dependencies = {
        'Z_MODE_ERR': ['Z_MODE'],
        'ODDS_MODE': ['Z_MODE'],
        'ODDS_MEAN': ['Z_MEAN']
    }

    # Determine which quantities to compute based on user input
    requested_q = {q.upper() for q in args.quantities}

    if 'ALL' in requested_q:
        q_to_compute = all_quantities
    else:
        # Validate that all requested quantities are known
        unknown_q = requested_q - all_quantities
        if unknown_q:
            print(f"Error: Unknown quantities specified: {', '.join(unknown_q)}", file=sys.stderr)
            print(f"Available quantities: {', '.join(sorted(list(all_quantities)))}", file=sys.stderr)
            sys.exit(1)
        q_to_compute = requested_q

    # Determine the full set of internal calculations needed, including dependencies
    internal_calcs = set(q_to_compute)
    for q in q_to_compute:
        if q in dependencies:
            internal_calcs.update(dependencies[q])
    
    print(f"Will compute: {', '.join(sorted(list(q_to_compute)))}")

    # Initialize dictionaries to hold new columns only for requested quantities
    d = {}
    for q_name in q_to_compute:
        d[q_name] = np.full(Nsources, np.nan, dtype=np.float32)

    valid = np.any(qcold != 0, axis=1)
               
    valid_indices = np.where(valid)[0]
    print(f"Calculating point estimates for {len(valid_indices)} valid sources...")
    
    for i in valid_indices:
        quantiles = decode_quantiles(qcold[i].tobytes())
        temp_results = {} # To store intermediate calculations for this source

        # --- Perform internal calculations ---
        # The order matters for dependencies.
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
            temp_results['ODDS_MODE'] = odds_from_quantiles(quantiles, temp_results['Z_MODE'], odds_window=args.odds_window)
        if 'ODDS_MEAN' in internal_calcs:
            temp_results['ODDS_MEAN'] = odds_from_quantiles(quantiles, temp_results['Z_MEAN'], odds_window=args.odds_window)
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

        # --- Assign computed values to the final arrays ---
        for q_name in q_to_compute:
            if q_name in temp_results:
                d[q_name][i] = temp_results[q_name]

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
    new_hdu.header['HISTORY'] = f'Computed point estimates from column: {args.encoded}'
    print(f"Writing point estimates to: {args.output}")
    new_hdu.writeto(args.output, overwrite=True)
    print('Done.')

# --- Logic for the 'plot' command ---
def plot_logic(args):
    """Logic to plot PDFs from compressed data."""
    # This import is local to the function to avoid making matplotlib
    # a hard dependency for users who only use encode/decode/measure.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for the plot command.", file=sys.stderr)
        sys.exit(1)

    print(f"Opening input file: {args.input}")
    with fits.open(args.input) as h:
        data = h[1].data

    qcold = data[args.pdfcol]

    # Determine which sources to plot
    if args.all:
        indices_to_plot = range(len(data))
        print(f"Plotting all {len(indices_to_plot)} sources...")
    else:
        if 'ID' not in data.columns.names:
            print(f"Error: --id specified, but no 'ID' column found in {args.input}", file=sys.stderr)
            sys.exit(1)
        source_ids = list(args.id)
        indices_to_plot = np.where(np.isin(data['ID'], source_ids))[0]
        if len(indices_to_plot) != len(source_ids):
            print("Warning: Some specified IDs were not found in the file.", file=sys.stderr)
        print(f"Found {len(indices_to_plot)} of {len(source_ids)} specified IDs to plot.")

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    for i in indices_to_plot:
        source_id = data['ID'][i] if 'ID' in data.columns.names else f"row_{i}"
        
        if not np.any(qcold[i] != 0):
            print(f"Skipping source {source_id}: No valid PDF data.")
            continue

        quantiles = decode_quantiles(qcold[i].tobytes())
        
        plt.figure(figsize=(8, 6))

        if args.method == 'steps' or args.method == 'all':
            z_steps, p_steps = reconstruct_pdf_from_quantiles(quantiles)
            plt.step(z_steps[:-1], p_steps, where='post', label='steps')

        if args.method == 'spline' or args.method == 'all':
            zvector = np.linspace(quantiles[0],quantiles[-1],500)
            pdf = cdf_to_pdf(quantiles, zvector=zvector, method='spline')
            plt.plot(zvector, pdf, label='spline')            

        plt.xlabel('Redshift (z)')
        plt.ylabel('Probability Density P(z)')
        plt.title(f'Reconstructed PDF for Source {source_id}')
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()

        output_filename = os.path.join(args.outdir, f"pdf_{source_id}.{args.format.lower()}")
        plt.savefig(output_filename)
        plt.close()
        print(f"Saved plot to {output_filename}")

# --- Logic for the 'check' command ---
def check_logic(args):
    """Logic to check binner or compressed PDFs."""
    print(f"Opening input file: {args.input}")
    with fits.open(args.input) as h:
        data = h[1].data
        header = h[1].header
        original_columns = data.columns
        if args.list is not None:
            ID = data[args.idcol]
        
        # Check PDFs
        if args.binned is None:
            samples = data[args.samples]
            invalid = (np.sum(~np.isfinite(samples),axis=1) > 0) # samples considered invalid if it has any non-finite elements 
            v = ~invalid # sources with valid samples
            unresolved = (np.max(samples[v],axis=1)-np.min(samples[v],axis=1) == 0) # all samples have the same value
        
        else:
            PDF = data[args.binned]
            invalid = (np.sum(~np.isfinite(PDF),axis=1) > 0) | (np.nanmin(PDF,axis=1) < 0) | (np.nanmax(PDF,axis=1) == 0.) # PDF is invalid if it has any non-finite or negative elements or if all elements are 0
            v = ~invalid # sources with valid samples
            unresolved = (np.sum(PDF[v],axis=1)-np.max(PDF[v],axis=1) == 0) # only one non-zero element in PDF
        
            threshold = args.truncation_threshold * np.max(PDF[v],axis=1)
            truncated = ((PDF[v,0] > threshold) | (PDF[v,-1] > threshold)) # PDF has too much weight on one or both extremes. Probably it is truncated.
    
    # Write summary report
    if args.binned is None:
        print(f'Column {args.samples} contains {samples.shape[0]} sampled PDFs, each containing {samples.shape[1]} random redshift samples.')
    else:    
        print(f'Column {args.binned} contains {PDF.shape[0]} binned PDFs, each containing {PDF.shape[1]} redshift bins.')
    
    Ninvalid = len(invalid[invalid])
    print(f"{Ninvalid} PDFs have been flagged as 'invalid'")
    Nunresolved = len(unresolved[unresolved])
    print(f"{Nunresolved} PDFs have been flagged as 'unresolved'")
    Ntruncated = len(truncated[truncated])
    print(f"{Ntruncated} PDFs have been flagged as 'truncated'")
        
    if args.list:
        print('List of source IDs with flagged issues in their PDFs:')
        for source in ID[invalid]:
            print(f"{source}  invalid")
        for i, source in enumerate(ID[v]):
            tag = ""
            if unresolved[i]:
                tag += " unresolved"
            if truncated[i]:
                tag += " truncated"
                    
            if tag != "":
                print(f"ID = {i}:{tag}")
    
    if args.output:                                
        # Write flags to output table
        d = {}
        d['Z_FLAGS'] = np.zeros(invalid.shape[0], dtype=np.int16)
        d['PDF_invalid'] = invalid
        d['PDF_unresolved'] = np.zeros(invalid.shape[0],dtype=bool)
        d['PDF_unresolved'][v] = unresolved
        d['Z_FLAGS'][d['PDF_invalid']] = 1
        d['Z_FLAGS'][d['PDF_unresolved']] += 2
        if args.binned is not None:
            d['PDF_truncated'] = np.zeros(invalid.shape[0],dtype=bool)
            d['PDF_truncated'][v] = truncated
            d['Z_FLAGS'][d['PDF_truncated']] += 4

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
            elif array.dtype == bool:
                format_str = 'L'    
            final_cols.append(fits.Column(name=name, format=format_str, array=array))

        new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)
        new_hdu.header['HISTORY'] = f'Added flags columns indicating issues in the PDFs: {new_col_names}'
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
    parser_encode.add_argument('input', metavar='input.fits', type=str, help='Name of input FITS catalog.')
    parser_encode.add_argument('output', metavar='output.fits', type=str, help='Name of output FITS catalog.')
    format_group = parser_encode.add_mutually_exclusive_group(required=True)
    format_group.add_argument('--binned', type=str, help='Name of input column containing binned PDFs.')
    format_group.add_argument('--samples', type=str, help='Name of input column containing a set of random samples from the PDFs.')
    parser_encode.add_argument('--out-encoded', type=str, nargs='?', default='coldpress_PDF', help='Name of output column containing the cold-pressed PDFs.')
    parser_encode.add_argument('--zmin', type=float, help='Redshift of the first bin (required with --binned).')
    parser_encode.add_argument('--zmax', type=float, help='Redshift of the last bin (required with --binned).')
    parser_encode.add_argument('--length', type=int, nargs='?', default=80, help='Length of cold-pressed PDFs in bytes (must be multiple of 4).')
    parser_encode.add_argument('--validate', action='store_true', default=False, help='Verify accuracy of recovered quantiles.')
    parser_encode.add_argument('--tolerance', type=float, nargs='?', default=0.001, help='Maximum shift tolerated for the redshift of the quantiles.')
    parser_encode.add_argument('--keep-orig', action='store_true', help='Include the original input column with binned PDFs or samples in the output file.')

    parser_encode.set_defaults(func=encode_logic) # This links the 'encode' command to its function

    # --- Parser for the "decode" command ---
    parser_decode = subparsers.add_parser('decode', help='Extract PDFs previously encoded with ColdPress.')
    parser_decode.add_argument('input', type=str, help='Name of input FITS catalog')
    parser_decode.add_argument('output', type=str, help='Name of output FITS catalog')
    parser_decode.add_argument('--encoded', type=str, nargs='?', default='coldpress_PDF', help='Name of column containing cold-pressed PDFs.')
    parser_decode.add_argument('--out-binned', type=str, nargs='?', default='PDF_decoded', help='Name of output column for extracted binned PDFs.')
    parser_decode.add_argument('--zmin', type=float, help='Redshift of the first bin (required with --out-binned).')
    parser_decode.add_argument('--zmax', type=float, help='Redshift of the last bin (required with --out-binned).')
    parser_decode.add_argument('--zstep', type=float, help='Width of the redshift bins (required with --out_binned).')
    parser_decode.add_argument('--force-range', action='store_true', help='Force binning to the range given by [zmin,zmax] even if some PDFs are truncated.')

    parser_decode.set_defaults(func=decode_logic) # This links the 'decode' command to its function

    # --- Parser for the "measure" command ---
    parser_measure = subparsers.add_parser('measure', help='Compute point estimates from compressed PDFs.')
    parser_measure.add_argument('input', type=str, help='Name of input FITS table containing cold-pressed PDFs.')
    parser_measure.add_argument('output', type=str, help='Name of output FITS table containing point estimates from the PDFs.')
    parser_measure.add_argument('--encoded', type=str, nargs='?', default='coldpress_PDF', help='Name of column containing cold-pressed PDFs.')
    parser_measure.add_argument('--quantities', type=str, nargs='+', default=['all'], help='List of quantities to measure from the PDFs. Default: "all". Choose any or all of these: Z_MODE Z_MEAN Z_MEDIAN Z_RANDOM Z_MODE_ERR Z_MEAN_ERR ODDS_MODE ODDS_MEAN Z_MIN_HPDCI68 Z_MAX_HPCI68 Z_MIN_HPCI95 Z_MAX_HPDCI95 ALL')
    parser_measure.add_argument('--odds-window', type=float, default=0.03, help='Half-width of the integration window for odds calculation (default: 0.03).')                              

    parser_measure.set_defaults(func=measure_logic)

    # --- Parser for the "plot" command ---
    parser_plot = subparsers.add_parser('plot', help='Reconstruct and plot PDFs encoded with ColdPress.')
    parser_plot.add_argument('input', type=str, help='Name of input FITS table containing cold-pressed PDFs.')
    plot_group = parser_plot.add_mutually_exclusive_group(required=True)
    plot_group.add_argument('--id', nargs='+', type=str, help='List of ID(s) of the source(s) to plot.')
    plot_group.add_argument('--plot-all', action='store_true', help='Plot PDFs for all the sources in the file.')
    parser_plot.add_argument('--idcol', type=str, help='Name of input column containing source IDs.')
    parser_plot.add_argument('--encoded', type=str, nargs='?', default='coldpress_PDF',
                                help='Name of input column containing cold-pressed PDFs.')
    parser_plot.add_argument('--outdir', type=str, default='.',
                                help='Output directory for plot files (default: current directory).')
    parser_plot.add_argument('--format', type=str, default='png',
                                help='Output format for plots (e.g., png, pdf, jpg; default: png).')
    parser_plot.add_argument('--method', type=str, default='all', choices=['steps', 'spline', 'all'],
                                help='PDF reconstruction method for plots (default: all).')

    parser_plot.set_defaults(func=plot_logic)
    
    # --- Parser for the "check" command ---
    parser_check = subparsers.add_parser('check', help='Check the PDFs for issues and flag them.')
    parser_check.add_argument('input', type=str, help='Name of input FITS catalog.')
    parser_check.add_argument('output', type=str, nargs='?', help='(Optional) name of output FITS catalog.')
    check_group = parser_check.add_mutually_exclusive_group(required=True)
    check_group.add_argument('--binned', type=str, help='Name of input column containing binned PDFs.')
    check_group.add_argument('--encoded', type=str, help='Name of input column containing cold-pressed PDFs.')
    parser_check.add_argument('--truncation-threshold', type=float, default=0.05, help='Threshold value for PDF truncation detection (default: 0.05)')
    parser_check.add_argument('--list', action='store_true', help='List ID and flags of all flagged PDFs.')
    parser_check.add_argument('--idcol', type=str, help='Name of input column containing source IDs (required with --list).')

    parser_check.set_defaults(func=check_logic)

    # Parse the arguments and call the function that was set by set_defaults
    args = parser.parse_args()
    
    # post-validation of arguments
    if args.command == 'encode':
        # Enforce zmin/zmax with --pdfcol, forbid with --samples
        if args.binned:
            if args.zmin is None or args.zmax is None:
                parser.error('--zmin and --zmax are required when encoding from binned PDFs (--binned)')
        else:
            # samples mode
            if args.zmin is not None or args.zmax is not None:
                parser.error('--zmin and --zmax can only be used with binned PDFs (--binned), not random samples (--samples)')

    if args.command == 'check':
        if args.list:
            if args.idcol is None:
                parser.error('--idcol is required when listing sources with flagged issues in their PDFs ( --list)')
                
    # Call the selected command function
    args.func(args)

if __name__ == '__main__':
    main()