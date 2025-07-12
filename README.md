# ColdPress

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**A toolkit for the efficient compression and analysis of redshift PDFs.**

The **coldpress** Python package implements the *ColdPress* algorithm for fast and efficient compression of probability density functions (PDFs) into a compact, fixed-size encoding. This is ideal for storing millions of redshift PDFs in large astronomical databases.

The *ColdPress* algorithm works by computing the redshifts *z*<sub>*i*</sub> that correspond to the quantiles *q*<sub>*i*</sub> of the cumulative distribution function (CDF) and encoding the differences *∆*<sub>*i*</sub> = *z*<sub>*i*</sub> - *z*<sub>*i*-1</sub> using (most often) a single byte.

> [!NOTE]
> The details of the algorithm and a performance comparison with alternative methods are presented in [this research note](https://iopscience.iop.org/article/10.3847/2515-5172/adeca6).

The CDF is obtained by integrating the probability density *P*(*z*). 
**coldpress** accepts as input the probability density *P*(*z*) in two common formats:

1.  **Binned PDF:** An array containing the values of *P(z)* at regular intervals from *z*<sub>min</sub> to *z*<sub>max</sub>. This is the typical output of most SED-fitting photo-z codes.
2.  **PDF from random samples:** An array of random redshift values drawn from the underlying probability distribution *P(z)*. This is the natural output of photo-z codes that use Monte Carlo methods.

Using the compressed representation of the CDF, **coldpress** can perform multiple tasks, including generating binned PDFs in a new grid, creating random samples, measuring statistics (mode, mean, confidence intervals, etc.), and plotting the reconstructed PDF.

## Installation

The **coldpress** package requires Python 3.8 or newer. The main dependencies are **numpy** and **astropy**. **matplotlib** is required for the `plot` command, and **scipy** is required for the `spline` interpolation method.

You can install **coldpress** directly from GitHub using `pip`:

```bash
pip install git+https://github.com/ahc-photoz/coldpress-project.git
```

## Usage

You can interact with **coldpress** in two main ways:

1.  **As a Python Module:** For maximum versatility, `import coldpress` directly into your Python scripts to access its API.

2.  **As a Command-Line Tool:** For working with FITS tables, the `coldpress` command provides a powerful interface.

    To see the main help message and available commands, run:
    
    ```bash
    coldpress --help
    ```
    To see the specific options for any command, such as `encode`, run:
    
    ```bash
    coldpress encode --help
    ```

## Quick Start

This section demonstrates a typical workflow using the `coldpress` command-line tool. We will inspect a FITS table, compress the PDFs it contains, measure key statistics, and plot the results.

### The Data

We will use a sample of 1,000 redshift PDFs from the Hyper Suprime-Cam Subaru Strategic Program (HSC-SSP) Public Data Release 3 ([Aihara et al. 2022](https://ui.adsabs.harvard.edu/abs/2022PASJ...74..247A/abstract)). The PDFs were generated with the **Mizuki** photometric redshift code ([Tanaka 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...801...20T/abstract)).

> [!NOTE]
> The full HSC-SSP PDR3 photo-z catalogs are available at the [official data release site](https://hsc-release.mtk.nao.ac.jp/doc/index.php/photometric-redshifts__pdr3/).

For this example, you can download a small sample file directly from this repository:

```bash
wget https://raw.githubusercontent.com/ahc-photoz/coldpress-project/main/examples/hsc_sample.fits
```

### 1. Inspect the File with `coldpress info`

First, view the contents of the FITS table to understand its structure:

```bash
coldpress info hsc_sample.fits 
```
```
Inspecting 'hsc_sample.fits'...
HDU 1 (Name: 'DATA')
  Rows: 1000
  Columns: 2
  --- Column Details ---
    - Name: ID                   Format: 1K
    - Name: PDF                  Format: 701E
```
The `PDF` column contains the probability density *P(z)* sampled in 701 bins. To find the corresponding redshift for each bin, we inspect the FITS header:

```bash
coldpress info hsc_sample.fits --header | grep -E 'Z_MIN|Z_MAX|DELTA_Z'
```
```
Z_MIN   =                   0. / Redshift of the first bin
Z_MAX   =                   7. / Redshift of the last bin
DELTA_Z =                 0.01 / Redshift bin width
```
This shows that the `PDF` column samples *P(z)* from z=0 to z=7 in steps of 0.01.

### 2. Compress PDFs with `coldpress encode`
To compress the PDFs, we provide the input and output filenames, the redshift range, and the name of the column containing the PDFs.

```bash
coldpress encode hsc_sample.fits hsc_sample_encoded.fits --zmin 0 --zmax 7 --binned PDF
```
```
Opening input file: hsc_sample.fits
Compressing PDFs into 80-byte packets (compression ratio: 35.05)...
1000 PDFs cold-pressed in 0.170699 CPU seconds
Excluding column 'PDF' from output FITS table.
Writing compressed data to: hsc_sample_encoded.fits
Done.
```
> [!IMPORTANT]
> The `--binned PDF` option tells **coldpress** that the `PDF` column contains binned PDFs. If your column contained random samples, you would use `--samples` instead.

By default, the original `PDF` column is removed. To keep it, add the `--keep-orig` flag. The compressed data is saved in a new column named `coldpress_PDF`.

### 3. Measure Statistics with `coldpress measure`
While a full PDF is comprehensive, point estimates like the mode or median are often more convenient. **coldpress** can measure many common statistics directly from the compressed data.

To see a list of all available quantities and their descriptions, run:

```bash
coldpress measure --list-quantities 
```
To calculate a few key statistics and save them to a new file, run:

```bash
coldpress measure hsc_sample_encoded.fits hsc_sample_measured.fits --quantities Z_MEDIAN Z_MODE ODDS_MODE
```
```
Opening input file: hsc_sample_encoded.fits
Will compute: ODDS_MODE, Z_MEDIAN, Z_MODE
Calculating point estimates for 1000 valid sources...
Writing point estimates to: hsc_sample_measured.fits
Done.
```
> [!TIP]
> Use `--quantities ALL` to compute all available statistics at once.

### Visualizing the PDFs with `coldpress plot`

You can quickly visualize any PDF directly from its compressed representation using the `plot` command. 

To plot the first ten PDFs in the table:

```bash
coldpress plot hsc_sample_encoded.fits --first 10 
```
This command saves the plots as PNG files in the current directory.

To plot the PDF for a specific source, use the `--id` and `--idcol` flags:

```bash 
coldpress plot hsc_sample_encoded.fits --idcol ID --id 73979566133084512
```

![Example of PDF visualized with coldpress plot](examples/pdf_73979566133084512.png)

> [!IMPORTANT]
> You don't need to decompress the PDFs into a new file before plotting. The `plot` command decodes them on the fly.

To reconstruct the continuous PDF from a discrete set of quantiles, **coldpress** must interpolate the CDF. It supports two methods:

* **Linear (`steps`):** A linear interpolation of the CDF results in a constant *P(z)* between quantiles, which is rendered as a step function.
* **Monotonic Cubic Spline (`spline`):** This produces a smooth *P(z)* curve while ensuring the cumulative probability in each inter-quantile interval is preserved.

By default, both methods are shown. You can choose to display only one using `--method steps` or `--method spline`.

You can also overplot any numerical quantity from the FITS table as a vertical line using the `--quantities` flag followed by the relevant column names.

```bash
coldpress plot hsc_sample_measured.fits --quantities Z_MODE Z_MEDIAN --idcol ID --id 73979566133084645
```
![Example of PDF with quantities marked](examples/pdf_73979566133084907.png)

> [!TIP] Use `--format JPEG` or `--format PDF` to save the figures in JPEG or PDF format. Use `--outdir <DIRECTORY>` to specify a different output directory.


### 5. Decompress PDFs with `coldpress decode`

For cases where you need the PDF in a standard binned format for other software, the `decode` command reconstructs the histogram on any grid you define.

For example, to reconstruct the PDFs in a finer grid using monotonic spline interpolation:

```bash
coldpress decode hsc_sample_encoded.fits hsc_sample_decoded.fits --zmin 0 --zmax 7 --zstep 0.005 --method spline
```
> [!WARNING]
> If a decoded PDF has non-zero probability outside the range you specify, `coldpress` will raise a truncation error. Use the `--force-range` flag to allow truncation. 

## Contributing

Contributions in the form of bug reports, patches, and feature requests are welcome. 

## Citation

If you use `coldpress` in your research, please acknowledge **coldpress** in your publications and cite the research note where **coldpress** is described:

> Hernán-Caballero, A. 2025, Res. Notes AAS, 9, 170.
> DOI: 10.3847/2515-5172/adeca6


You can use the following BibTeX entry:

```bibtex
@ARTICLE{<ColdPressRN>,
   author = {{Hern\'an-Caballero, A.}},
    title = "{ColdPress: Efficient Quantile-based Compression of Photometric Redshift PDFs}",
  journal = {Research Notes of the AAS},
     year = {2025},
   volume = {9},
      eid = {170},
      doi = {10.3847/2515-5172/adeca6},
   archivePrefix = {arXiv},
   eprint = {},
}
```

## License
This project is licensed under the GNU General Public License v3.0. See the `COPYING` file for more details.