#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
from astropy.io import fits

# Use relative imports from the new submodules
from . import __version__
from .encode import encode_from_binned, encode_from_samples
from .decode import decode_to_binned, decode_quantiles, quantiles_to_binned
from .stats import measure_from_quantiles, ALL_QUANTITIES
from .utils import step_pdf_from_quantiles, plot_from_quantiles

# --- Logic for the 'info' command ---
def info_logic(args):
    """Logic to display information about a FITS file."""
    try:
        with fits.open(args.input) as h:
            if args.hdu >= len(h):
                print(f"Error: HDU {args.hdu} not found. File has {len(h)} HDUs (0 to {len(h)-1}).", file=sys.stderr)
                sys.exit(1)
                
            hdu = h[args.hdu]
            
            print(f"Inspecting '{args.input}'...")
            
            if not hdu.is_image:
                print(f"HDU {args.hdu} (Name: '{hdu.name}')")
                print(f"  Rows: {hdu.header['NAXIS2']}")
                print(f"  Columns: {len(hdu.columns)}")
                print("  --- Column Details ---")
                for col in hdu.columns:
                    print(f"    - Name: {col.name:<20} Format: {col.format}")
            else:
                print(f"HDU {args.hdu} is an Image HDU (Name: '{hdu.name}')")
                print(f"  Dimensions: {hdu.shape}")
                print(f"  Data Type (BITPIX): {hdu.header['BITPIX']}")

            if args.header:
                print("\n--- FITS Header ---")
                print(repr(hdu.header))

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


# --- Logic for the 'encode' command ---
def encode_logic(args):
    import time
    if args.length % 4 != 0:
        print(f"Error: Packet length (--length) must be a multiple of 4, but got {args.length}.", file=sys.stderr)
        sys.exit(1)

    print(f"Opening input file: {args.input}")

    with fits.open(args.input) as h:
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

        start = time.process_time()
        if args.binned is None:
            coldpress_PDF = encode_from_samples(samples, packetsize=args.length, ini_quantiles=args.length-9,
                                                   validate=args.validate, tolerance=args.tolerance)        
        else:
            coldpress_PDF = encode_from_binned(PDF, zvector, packetsize=args.length, ini_quantiles=args.length-9,
                                                   validate=args.validate, tolerance=args.tolerance)
        end = time.process_time()
        cpu_seconds = end - start
        print(f"{coldpress_PDF.shape[0]} PDFs cold-pressed in {cpu_seconds:.6f} CPU seconds")

        nints = args.length // 4
        new_col = fits.Column(name=args.out_encoded, format=f'{nints}J', array=coldpress_PDF)

        final_cols = [c for c in orig_cols if c.name != args.out_encoded]

        if args.binned is None:
            orig_column = args.samples
        else:
            orig_column = args.binned  
              
        if args.keep_orig:
            print(f"Including column '{orig_column}' in output FITS table.")
        else:
            final_cols = [c for c in final_cols if c.name != orig_column]
            print(f"Excluding column '{orig_column}' from output FITS table.")

        final_cols.append(new_col)
        new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)

    if args.binned is None:
        new_hdu.header.add_history(f'PDFs from samples in column {args.samples} cold-pressed as {args.out_encoded}')
    else:       
        new_hdu.header.add_history(f'Binned PDFs in column {args.binned} cold-pressed as {args.out_encoded}')
    print(f"Writing compressed data to: {args.output}")
    new_hdu.writeto(args.output, overwrite=True)
    print('Done.')


# --- Logic for the 'decode' command ---
def decode_logic(args):
    print(f"Opening input file: {args.input}")

    with fits.open(args.input) as h:
        header = h[1].header
        coldpress_PDF = h[1].data[args.encoded]
        orig_cols = list(h[1].columns)

        zvector = np.arange(args.zmin, args.zmax + args.zstep/2, args.zstep)
        zvsize = len(zvector)

        print(f"Decompressing PDFs using {args.method} interpolation of the quantiles...")
        try:
            decoded_PDF = decode_to_binned(coldpress_PDF, zvector, force_range=args.force_range, method=args.method)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Hint: Use the --force-range flag to proceed with truncation at your own risk.", file=sys.stderr)
            sys.exit(1)

        new_col = fits.Column(name=args.out_binned, format=f'{zvsize}E', array=decoded_PDF)
        final_cols = [c for c in orig_cols if c.name != args.out_binned]
        final_cols.append(new_col)
        new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)
    
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
    
    # Argparse now handles validation via the 'choices' parameter.
    # We only need to determine the final list of quantities to compute.
    q_to_compute = {q.upper() for q in args.quantities}
    if 'ALL' in q_to_compute:
        q_to_compute = ALL_QUANTITIES
    
    print(f"Will compute: {', '.join(sorted(list(q_to_compute)))}")

    d = {}
    for q_name in q_to_compute:
         d[q_name] = np.full(Nsources, np.nan, dtype=np.float32)

    valid = np.any(qcold != 0, axis=1)
    valid_indices = np.where(valid)[0]
    print(f"Calculating point estimates for {len(valid_indices)} valid sources...")
    
    for i in valid_indices:
        quantiles = decode_quantiles(qcold[i].tobytes())
        # The API function call now passes the list determined by argparse
        results = measure_from_quantiles(
            quantiles,
            quantities_to_measure=list(q_to_compute),
            odds_window=args.odds_window
        )
        for q_name, value in results.items():
            d[q_name][i] = value

    final_cols = []
    for col in original_columns:
        if col.name not in q_to_compute:
            final_cols.append(col)

    for name, array in d.items():
        format_str = 'E' if array.dtype == np.float32 else 'I'
        final_cols.append(fits.Column(name=name, format=format_str, array=array))

    new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)
    new_hdu.header['HISTORY'] = f'Computed point estimates from column: {args.encoded}'
    print(f"Writing point estimates to: {args.output}")
    new_hdu.writeto(args.output, overwrite=True)
    print('Done.')


# --- Logic for the 'plot' command ---
def plot_logic(args):
    """Logic to plot PDFs from compressed data."""
    print(f"Opening input file: {args.input}")
    with fits.open(args.input) as h:
        try:
            import matplotlib
        except ImportError:
            print("Error: matplotlib is required for the plot command.", file=sys.stderr)
            sys.exit(1)
        data = h[1].data

    qcold = data[args.encoded]

    if args.quantities:
        all_cols_upper = {c.upper() for c in data.columns.names}
        for q_col in args.quantities:
            if q_col.upper() not in all_cols_upper:
                print(f"Error: Quantity column '{q_col}' not found in FITS table.", file=sys.stderr)
                sys.exit(1)

    if args.plot_all:
        indices_to_plot = range(len(data))
        print(f"Plotting all {len(indices_to_plot)} sources...")
    elif args.first is not None:
        if args.first <= len(data):
            indices_to_plot = range(args.first)
            print(f"Plotting first {len(indices_to_plot)} sources...")
        else:
            print(f"Warning: first {args.first} PDFs were requested but file contais only {len(data)}. Will plot all of them.")
            indices_to_plot = range(len(data))
    else:    
        if args.idcol not in data.columns.names:
            print(f"Error: --id specified, but no '{args.idcol}' column found in {args.input}", file=sys.stderr)
            sys.exit(1)
        source_ids = list(args.id)
        id_column_as_str = data[args.idcol].astype(str)
        source_ids_as_str = np.asarray(source_ids, dtype=str)
        indices_to_plot = np.where(np.isin(id_column_as_str, source_ids_as_str))[0]

        if len(indices_to_plot) != len(source_ids):
            print("Warning: Some specified IDs were not found in the file.", file=sys.stderr)
        print(f"Found {len(indices_to_plot)} of {len(source_ids)} specified IDs to plot.")

    os.makedirs(args.outdir, exist_ok=True)

    for i in indices_to_plot:
        source_id_val = data[args.idcol][i] if (args.idcol is not None) and (args.idcol in data.columns.names) else f"row_{i}"
        
        if not np.any(qcold[i] != 0):
            print(f"Skipping source {source_id_val}: No valid PDF data.")
            continue

        quantiles = decode_quantiles(qcold[i].tobytes())
        
        markers_to_plot = {}
        if args.quantities:
            for q_col in args.quantities:
                actual_col_name = next((c for c in data.columns.names if c.upper() == q_col.upper()), None)
                if actual_col_name:
                    markers_to_plot[q_col] = data[actual_col_name][i]

        output_filename = os.path.join(args.outdir, f"pdf_{source_id_val}.{args.format.lower()}")
        
        plot_from_quantiles(
            quantiles,
            output_filename=output_filename,
            source_id=source_id_val,
            method=args.method,
            markers=markers_to_plot
        )
        print(f"Saved plot to {output_filename}")

# --- Logic for the 'check' command ---
def check_logic(args):
    print(f"Opening input file: {args.input}")
    with fits.open(args.input) as h:
        data = h[1].data
        header = h[1].header
        original_columns = data.columns
        if args.list:
            ID = data[args.idcol]
        
        if args.binned is None:
            samples = data[args.samples]
            invalid = (np.sum(~np.isfinite(samples),axis=1) > 0)
            v = ~invalid
            unresolved = (np.max(samples[v],axis=1)-np.min(samples[v],axis=1) == 0)
        else:
            PDF = data[args.binned]
            invalid = (np.sum(~np.isfinite(PDF),axis=1) > 0) | (np.nanmin(PDF,axis=1) < 0) | (np.nanmax(PDF,axis=1) == 0.)
            v = ~invalid
            unresolved = (np.sum(PDF[v],axis=1)-np.max(PDF[v],axis=1) == 0)
            threshold = args.truncation_threshold * np.max(PDF[v],axis=1)
            truncated = ((PDF[v,0] > threshold) | (PDF[v,-1] > threshold))
    
    if args.binned is None:
        print(f'Column {args.samples} contains {samples.shape[0]} sampled PDFs, each containing {samples.shape[1]} random redshift samples.')
    else:    
        print(f'Column {args.binned} contains {PDF.shape[0]} binned PDFs, each containing {PDF.shape[1]} redshift bins.')
    
    print(f"{np.sum(invalid)} PDFs have been flagged as 'invalid'")
    print(f"{np.sum(unresolved)} PDFs have been flagged as 'unresolved'")
    if args.binned is not None:
        print(f"{np.sum(truncated)} PDFs have been flagged as 'truncated'")
        
    if args.list:
        print('List of source IDs with flagged issues in their PDFs:')
        for source in ID[invalid]:
            print(f"{source}  invalid")
        for i, source in enumerate(ID[v]):
            tag = ""
            if unresolved[i]:
                tag += " unresolved"
            if args.binned is not None and truncated[i]:
                tag += " truncated"
            if tag != "":
                print(f"ID = {source}:{tag}")
    
    if args.output:                                
        d = {}
        d['Z_FLAGS'] = np.zeros(len(invalid), dtype=np.int16)
        d['PDF_invalid'] = invalid
        d['PDF_unresolved'] = np.zeros(len(invalid),dtype=bool)
        d['PDF_unresolved'][v] = unresolved
        d['Z_FLAGS'][d['PDF_invalid']] = 1
        d['Z_FLAGS'][d['PDF_unresolved']] += 2
        if args.binned is not None:
            d['PDF_truncated'] = np.zeros(len(invalid),dtype=bool)
            d['PDF_truncated'][v] = truncated
            d['Z_FLAGS'][d['PDF_truncated']] += 4

        final_cols = []
        new_col_names = d.keys()
        for col in original_columns:
            if col.name not in new_col_names:
                final_cols.append(col)

        for name, array in d.items():
            if array.dtype == np.float32: format_str = 'E'
            elif array.dtype == np.int16: format_str = 'I'
            elif array.dtype == bool: format_str = 'L'    
            final_cols.append(fits.Column(name=name, format=format_str, array=array))

        new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)
        new_hdu.header['HISTORY'] = f'Added flags columns indicating issues in the PDFs: {list(new_col_names)}'
        print(f"Writing point estimates to: {args.output}")
        new_hdu.writeto(args.output, overwrite=True)
        print('Done.')

# --- Main Entry Point and Parser Configuration ---
def main():
    parser = argparse.ArgumentParser(description='Compression, analysis, and visualization of redshift PDFs.')
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Parser for the "info" command ---
    parser_info = subparsers.add_parser('info', help='Display information about a FITS file HDU.')
    parser_info.add_argument('input', metavar='input.fits', type=str, help='Name of the input FITS file.')
    parser_info.add_argument('--hdu', type=int, default=1, help='HDU to inspect (default: 1).')
    parser_info.add_argument('--header', action='store_true', help='Print the full FITS header.')
    parser_info.set_defaults(func=info_logic)

    # --- Parser for the "encode" command ---
    parser_encode = subparsers.add_parser('encode', help='Compress PDFs into coldpress format.')
    parser_encode.add_argument('input', metavar='input.fits', type=str, help='Name of input FITS catalog.')
    parser_encode.add_argument('output', metavar='output.fits', type=str, help='Name of output FITS catalog.')
    format_group = parser_encode.add_mutually_exclusive_group(required=True)
    format_group.add_argument('--binned', type=str, help='Name of input column containing binned PDFs.')
    format_group.add_argument('--samples', type=str, help='Name of input column containing a set of random samples from the PDFs.')
    parser_encode.add_argument('-o', '--out-encoded', type=str, nargs='?', default='coldpress_PDF', help='Name of output column containing the cold-pressed PDFs.')
    parser_encode.add_argument('--zmin', type=float, help='Redshift of the first bin (required with --binned).')
    parser_encode.add_argument('--zmax', type=float, help='Redshift of the last bin (required with --binned).')
    parser_encode.add_argument('--length', type=int, nargs='?', default=80, help='Length of cold-pressed PDFs in bytes (must be multiple of 4).')
    parser_encode.add_argument('--validate', action='store_true', default=False, help='Verify accuracy of recovered quantiles.')
    parser_encode.add_argument('--tolerance', type=float, nargs='?', default=0.001, help='Maximum shift tolerated for the redshift of the quantiles.')
    parser_encode.add_argument('--keep-orig', action='store_true', help='Include the original input column with binned PDFs or samples in the output file.')
    parser_encode.set_defaults(func=encode_logic)

    # --- Parser for the "decode" command ---
    parser_decode = subparsers.add_parser('decode', help='Extract PDFs previously encoded with ColdPress.')
    parser_decode.add_argument('input', type=str, help='Name of input FITS catalog')
    parser_decode.add_argument('output', type=str, help='Name of output FITS catalog')
    parser_decode.add_argument('--encoded', type=str, nargs='?', default='coldpress_PDF', help='Name of column containing cold-pressed PDFs.')
    parser_decode.add_argument('-o', '--out-binned', type=str, nargs='?', default='PDF_decoded', help='Name of output column for extracted binned PDFs.')
    parser_decode.add_argument('--zmin', type=float, help='Redshift of the first bin.')
    parser_decode.add_argument('--zmax', type=float, help='Redshift of the last bin.')
    parser_decode.add_argument('--zstep', type=float, help='Width of the redshift bins.')
    parser_decode.add_argument('--force-range', action='store_true', help='Force binning to the range given by [zmin,zmax] even if some PDFs are truncated.')
    parser_decode.add_argument('--method', type=str, nargs='?', default='linear', choices=['linear','spline'], help='Interpolation method for reconstruction of the binned PDF (default: linear).')
    parser_decode.set_defaults(func=decode_logic)

    # --- Parser for the "measure" command ---
    parser_measure = subparsers.add_parser('measure', help='Compute point estimates from compressed PDFs.')
    parser_measure.add_argument('input', type=str, nargs='?', help='Name of input FITS table containing cold-pressed PDFs.')
    parser_measure.add_argument('output', type=str, nargs='?', help='Name of output FITS table containing point estimates measured on the PDFs.')
    parser_measure.add_argument('--encoded', type=str, nargs='?', default='coldpress_PDF', help='Name of column containing cold-pressed PDFs.')
    choices_list = sorted(list(ALL_QUANTITIES) + ['ALL'])
    parser_measure.add_argument('--quantities', type=str, nargs='+', default=['all'], choices=choices_list, metavar='QUANTITY', help='List of quantities to measure from the PDFs (default: all).')
    parser_measure.add_argument('--odds-window', type=float, default=0.03, help='Half-width of the integration window for odds calculation.')
    parser_measure.add_argument('--list-quantities', action='store_true', help='List all available quantities and their descriptions.')
                              
    parser_measure.set_defaults(func=measure_logic)

    # --- Parser for the "plot" command ---
    parser_plot = subparsers.add_parser('plot', help='Reconstruct and plot PDFs encoded with ColdPress.')
    parser_plot.add_argument('input', type=str, help='Name of input FITS table containing cold-pressed PDFs.')
    plot_group = parser_plot.add_mutually_exclusive_group(required=True)
    plot_group.add_argument('--id', nargs='+', type=str, help='List of ID(s) of the source(s) to plot.')
    plot_group.add_argument('--first', metavar='N', type=int, help='plot PDFs for the first N sources in the file.')
    plot_group.add_argument('--plot-all', action='store_true', dest='plot_all', help='Plot PDFs for all the sources in the file.')
    parser_plot.add_argument('--idcol', type=str, nargs='?', help='Name of input column containing source IDs.')
    parser_plot.add_argument('--encoded', type=str, nargs='?', default='coldpress_PDF', help='Name of input column containing cold-pressed PDFs.')
    parser_plot.add_argument('--outdir', type=str, default='.', help='Output directory for plot files.')
    parser_plot.add_argument('--format', type=str, default='png', help='Output format for plots.')
    parser_plot.add_argument('--method', type=str, default='all', choices=['steps', 'spline', 'all'], help='PDF reconstruction method for plots.')
    parser_plot.add_argument('--quantities', nargs='+', type=str, help='List of FITS columns to overplot as vertical lines.')
    parser_plot.set_defaults(func=plot_logic)
    
    # --- Parser for the "check" command ---
    parser_check = subparsers.add_parser('check', help='Check the PDFs for issues and flag them.')
    parser_check.add_argument('input', type=str, help='Name of input FITS catalog.')
    parser_check.add_argument('output', type=str, nargs='?', help='(Optional) name of output FITS catalog.')
    check_group = parser_check.add_mutually_exclusive_group(required=True)
    check_group.add_argument('--binned', type=str, help='Name of input column containing binned PDFs.')
    check_group.add_argument('--samples', type=str, help='Name of input column containing redshift samples.')
    parser_check.add_argument('--truncation-threshold', type=float, default=0.05, help='Threshold value for PDF truncation detection.')
    parser_check.add_argument('--list', action='store_true', help='List ID and flags of all flagged PDFs.')
    parser_check.add_argument('--idcol', type=str, help='Name of input column containing source IDs (required with --list).')
    parser_check.set_defaults(func=check_logic)

    args = parser.parse_args()
    
    if args.command == 'encode':
        if args.binned and (args.zmin is None or args.zmax is None):
            parser.error('--zmin and --zmax are required when encoding from binned PDFs (--binned)')
        if args.samples and (args.zmin is not None or args.zmax is not None):
            parser.error('--zmin and --zmax can only be used with binned PDFs (--binned), not random samples (--samples)')

    if args.command == 'check' and args.list and args.idcol is None:
        parser.error('--idcol is required when listing sources with flagged issues (--list)')

    if args.command == 'measure' and args.list_quantities:
        print("Available quantities for the 'measure' command:")
        from .stats import QUANTITY_DESCRIPTIONS
        for name, desc in QUANTITY_DESCRIPTIONS.items():
            print(f"  {name:<15} {desc}")
        sys.exit(0) # Exit after printing the list

    if args.command == 'measure':
        if not args.input or not args.output:
            parser_measure.error("the following arguments are required: input, output")

    args.func(args)

if __name__ == '__main__':
    main()