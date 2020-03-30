# Time series ordinal classification via shapelets

[![DOI](https://zenodo.org/badge/213915623.svg)](https://zenodo.org/badge/latestdoi/213915623)

## Algorithms included

This repo includes different shapelet quality measures trying to boost the order information by refining the quality measure. These are:

* Ordinal Fisher.
* Pearson's correlation coefficient.
* Spearman's correlation coefficient.

## Installation

### Dependencies

This repo basically requires:

 * Python (>= 3.6.8)
 * NumPy (>= 1.16.4)
 * SciPy (>= 1.3.0)
 * sktime (>= 0.3.0)
 * Pandas (>=0.24.2)
 * Scikit-learn (>=0.21.2)

 Moreover, the Ordinal Regression and Classification Algorithms repository [ORCA](https://github.com/ayrna/orca) should also be installed.

### Compilation

To install the requirements, use:

    pip install -r requirements.txt

Note that the MATLAB engine for python should be installed following these [instructions](https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). Moreover, ORCA can be installed from its [GitHub repo](https://github.com/ayrna/orca).

## Development

Contributions are welcome. Pull requests are encouraged to be formatted according to [PEP8](https://www.python.org/dev/peps/pep-0008/), e.g., using [yapf](https://github.com/google/yapf).

## Usage

Follow the example exposed in main.py, use:

    python main.py -t path_to_datasets/ -p path_to_save_intermediate_files -r path_to_save_results

Note that the `path_to_datasets/` should contain the datasets downloaded from [UEA TSC repo](http://www.timeseriesclassification.com/).

Moreover, extra parameters can be set inside `main.py`.

## Citation

Information about the citation will be exposed soon.

## Contributors

#### TSOC via shapelets

* David Guijo-Rubio ([@dguijo](https://github.com/dguijo))
* Pedro Antonio Gutiérrez ([@pagutierrez](https://github.com/pagutierrez))
* Anthony Bagnall ([@TonyBagnall](https://github.com/TonyBagnall))
