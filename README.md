# Archetypal Analysis for Population Genetics
This tool estimates genetic clusters by using genomic data in an unsupervised, computationally efficient manner, as described in [Gimbernat-Mayol et al, PLOS COMPUTATIONAL BIOLOGY](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010301). The approach combines singular value decomposition (SV) with Archetypal Analysis to perform fast and accurate genetic clustering by first reducing the dimensionality of the space of genomic sequences. Each sequence is described as a convex combination (admixture) of archetypes (cluster representatives) in the reduced dimensional space. Visualizations of compositional plots are available by mapping the admixture coefficients to a simplex. See the example below of genetic data plotted in a PCA, with its corresponding 8-archetype compositional plot:
![PCA and Compositional plot](https://github.com/AI-sandbox/archetypal-analysis/blob/master/pca-polygon.jpg)

## Software requirements

The package has been tested on both Linux (Ubuntu 18.04.5 LTS) and MacOS (BigSur 11.2.3, Intel).

We recommend creating a fresh Python 3.6 environment using `virtualenv` (or `conda`), and then installing the package there. As an example, for `virtualenv`, one should launch the following commands:

```console
> virtualenv --python=python3.6 ~/venv/aa_env
> source ~/venv/aa_env/bin/activate
(aa_env) > pip3 install -e .
```

## Installation Guide

The package can be easily installed using `pip` on the root directory:

```console
(aa_env) > pip3 install -e .
```

## Usage 

```console
> archetypal-analysis -i INPUT_FILE -o OUTPUT_FILE -k K
```

### Arguments

- `-i`/`--input_file`: defines the input file / path. File must be in VCF, BED, PGEN or NPY format. If format is NPY, the data is assumed to be already projected.
- `-o`/`--output_file`: defines the output file / path. File name does not need any extensions.
- `-k`/`--n_archetypes`: defines the number of archetypes.
- `--tolerance`: defines when to stop optimization.
- `--max_iter`: defines the maximum number of iterations.
- `--random_state`: defines the random seed number for initialization. No effect if "furthest_sum" is selected.                   
- `-C`/`--constraint_coef`: constraint coefficient to ensure that the summation of alfa's and beta's equals to 1. C is conisdered to be inverse of M^2 in the original paper.
- `--initialize`: defines the initialization method to guess initial archetypes.
- `-dr`/`--dim_reduction`: defines the dimensionality reduction technique to project the input data. Accepted=['PCA', 'MDS', 'UMAP', 'TSNE']. Default='PCA'.

## Using Plink2 binary files (.pgen)

If the data format that you will be working on is _Plink2 Binary Files (.pgen, .psam, .pvar)_ then you also need to install the package `pgenlib`. This package is not available in PyPi, but is included in the [plink repository](https://github.com/chrchang/plink-ng/tree/master/2.0/Python). Installation instructions can be found in the [corresponding `README.md` file](https://github.com/chrchang/plink-ng/blob/master/2.0/Python/ReadMe.md). While you will need to clone the whole repository, you can remove it after installing the package, unless you plan to work with it.

## Plotting
For the plotting functionally, you should use archetypal-plot instead of archetypal analysis in bash. 

```console
archetypal-plot -i [Q_FILE_OUTPUT_PATH] -p [PLOT_TYPE] [OTHER_PLOT_OPTIONS]
```
**Parameters:**
- `Q_FILE_OUTPUT_PATH`: Path where the archetypal analysis result was saved.
- `PLOT_TYPE`: The type of plot you want. Options include:
  - `bar_simple`: Simple matplotlib bar plot.
  - `bar_html`: Interactive HTML bar plot.
  - `plot_simplex`: Simplex plot using matplotlib.
  - `plot_simplex_html`: Interactive HTML simplex plot.
  - `bar_labeled`: Labeled bar plot.

**Other Plot Options:**

- `supplement`: Path for any supplementary description data. The data file should have no column name or index, and values should be separated by spaces.
- `sorted`: If passed, it sorts the data by the maximum archetype component.
- `dataTitle`: Title for your plot.
- `dpi_num`: Quality of the saved plot graphic. 

For example:

```console
archetypal-plot -i data -p bar_simple
```

## License

**NOTICE**: This software is available for use free of charge for academic research use only. Academic users may fork this repository and modify and improve to suit their research needs, but also inherit these terms and must include a licensing notice to that effect. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as "academic research" should contact the authors for a separate license. This applies to this repository directly and any other repository that includes source, executables, or git commands that pull/clone this repository as part of its function. Such repositories, whether ours or others, must include this notice.

## Cite

When using this software, please cite the following paper (currently pre-print):

```{tex}
@article {Gimbernat-Mayol2021.11.28.470296,
	author = {Gimbernat-Mayol, Julia and Montserrat, Daniel Mas and Bustamante, Carlos D. and Ioannidis, Alexander G.},
	title = {Archetypal Analysis for Population Genetics},
	elocation-id = {2021.11.28.470296},
	year = {2021},
	doi = {10.1101/2021.11.28.470296},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {The estimation of genetic clusters using genomic data has application from genome-wide association studies (GWAS) to demographic history to polygenic risk scores (PRS) and is expected to play an important role in the analyses of increasingly diverse, large-scale cohorts. However, existing methods are computationally-intensive, prohibitively so in the case of nationwide biobanks. Here we explore Archetypal Analysis as an efficient, unsupervised approach for identifying genetic clusters and for associating individuals with them. Such unsupervised approaches help avoid conflating socially constructed ethnic labels with genetic clusters by eliminating the need for exogenous training labels. We show that Archetypal Analysis yields similar cluster structure to existing unsupervised methods such as ADMIXTURE and provides interpretative advantages. More importantly, we show that since Archetypal Analysis can be used with lower-dimensional representations of genetic data, significant reductions in computational time and memory requirements are possible. When Archetypal Analysis is run in this fashion, it takes several orders of magnitude less compute time than the current standard, ADMIXTURE. Finally, we demonstrate uses ranging across datasets from humans to canids.Author summary This work introduces a method that combines the singular value decomposition (SVD) with Archetypal Analysis to perform fast and accurate genetic clustering by first reducing the dimensionality of the space of genomic sequences. Each sequence is described as a convex combination (admixture) of archetypes (cluster representatives) in the reduced dimensional space. We compare this interpretable approach to the widely used genetic clustering algorithm, ADMIXTURE, and show that, without significant degradation in performance, Archetypal Analysis outperforms, offering shorter run times and representational advantages. We include theoretical, qualitative, and quantitative comparisons between both methods.Competing Interest StatementCDB is CEO of Galatea Bio Inc.},
	URL = {https://www.biorxiv.org/content/early/2021/11/29/2021.11.28.470296},
	eprint = {https://www.biorxiv.org/content/early/2021/11/29/2021.11.28.470296.full.pdf},
	journal = {bioRxiv}
}
```

