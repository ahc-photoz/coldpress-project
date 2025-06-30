# ColdPress

[](https://www.gnu.org/licenses/gpl-3.0)

**Efficient compression for redshift probability density functions**

This module provides functions to encode and decode redshift probability density
functions (PDFs) into a compact, fixed-size byte representation suitable for
efficient storage in databases. 
It works by computing the redshifts {z\_i} that correspond to the quantiles {q\_i}
of the cumulative distribution function (CDF) and encoding the differences
∆i = z\_i - z\_{i-1} using (mostly) a single byte.

This package provides functions to encode and decode 1D probability density functions (PDFs), such as photometric redshift estimations, into a compact, fixed-size byte representation. This is ideal for efficient storage in large databases. The compression works by converting the PDF to its cumulative (CDF) form and storing a compressed representation of its quantiles.

## Installation

ColdPress works with Python 3.8 or newer. The only major dependencies are NumPy and AstroPy, which will be installed automatically if not already available.

You can install `coldpress` directly from GitHub using `pip`:

```bash
pip install git+https://github.com/ahc-photoz/coldpress-project.git
```
## Usage

You can interact with `coldpress` in two main ways:

1.  **As a Python Module:** For maximum versatility, import `coldpress` directly into your Python scripts to access all its functions.
    ```python
    import coldpress
    ```

2.  **As a Command-Line Tool:** If you just want to compress or decompress redshift PDFs in a FITS table, you can use the `coldpress` command-line tool.

    To see the main help message and available commands (`encode`, `decode`), run:
    ```bash
    coldpress --help
    ```
    To see the specific options for a command, run:
    ```bash
    coldpress encode --help
    ```

## Example

This example demonstrates how to compress a FITS file containing redshift PDFs and then decompress it back.

### The Data

We will use a sample of 1,000 redshift PDFs from the **Hyper Suprime-Cam Subaru Strategic Program (HSC-SSP) Public Data Release 3**. The PDFs were generated using the `Mizuki` photometric redshift code.

> The full HSC-SSP PDR3 photo-z catalogs are available from the [official data release site](https://hsc-release.mtk.nao.ac.jp/doc/index.php/photometric-redshifts__pdr3/). Please be aware that the full datasets are distributed in very large tarballs.

For this example, you can download a small, pre-made sample file directly from this repository.

**1. Download the sample data:**
```bash
wget https://raw.githubusercontent.com/ahc-photoz/coldpress-project/main/examples/hsc_sample.fits
```

**2. Important Parameters:**
The `coldpress` tool needs to know the redshift range of the PDFs. For the `Mizuki` code in HSC PDR3, this range is `z = 0.0` to `z = 7.0`.

### Compression and Decompression

**1. Compress the PDFs:**
This command will take the `PDF` column from the input file, compress it, and save the result to a new file.
```bash
coldpress encode hsc_sample.fits hsc_sample_compressed.fits --zmax 7.0
```

**2. Decompress the PDFs:**
This command will take the compressed data and recover the original PDF, saving it to a new FITS file. We specify the desired grid for the output PDF with `--zstep`.
```bash
coldpress decode hsc_sample_compressed.fits hsc_sample_recovered.fits --zmax 7.0 --zstep 0.01
```

You can now use a FITS viewer or a Python script to compare the PDFs in `hsc_sample.fits` and `hsc_sample_recovered.fits` to verify the process.


## Citation

If you use `coldpress` in your research, please cite the research note where it is described. This is the most important way to support the project's development.

> Hernán-Caballero, A. 2025, Research Notes of the AAS, \<Volume\>, \<ID\>.
> DOI: \<Your DOI Here\>
> arXiv: \<Your arXiv ID Here\>

You can use the following BibTeX entry:

```bibtex
@ARTICLE{<YourBibtexKey>,
   author = {{<Author(s)>}},
    title = "{<Title of Your Research Note>}",
  journal = {Research Notes of the AAS},
     year = <Year>,
   volume = <Volume>,
      eid = {<ID>},
      doi = {<Your DOI Here>},
   archivePrefix = {arXiv},
   eprint = {<Your arXiv ID Here>},
}
```

## License

This project is licensed under the GNU General Public License v3.0. See the [COPYING](https://www.google.com/search?q=COPYING) file for more details.
