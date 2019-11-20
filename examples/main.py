# -*- coding: utf-8 -*-
import argparse
import sys
import TSOC.utils.experiments as exp
import numpy as np
import matlab
import itertools as it

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", type=str, default="/home/david/TSOC/datasets/", help="Path to the datasets")
parser.add_argument("--res", "-r", type=str, default="/home/david/TSOC/results/", help="Path to save the results")
args = parser.parse_args()


def comparison_experiments(data_dir, res_dir):
    complete_classifiers = [
        #"kdlor",
        #"pom",
        "svorim",
    ]

    small_datasets = [
        "toy",
        #"tormentas",
    ]

    params = {
        "kdlor": {'C': np.logspace(-3, 3, 7),
                  'k': np.logspace(-3, 3, 7),
                  'u': np.logspace(-3, 3, 7)},
        "pom": [],
        "svorim": {'C': np.logspace(-3, 3, 7),
                   'k': np.logspace(-3, 3, 7)},
    }

    params = {
        "pom": None,
        "svorim": {'C': [1],
                   'k': [10, 100]},
    }

    for d in small_datasets:
        for c in complete_classifiers:
            print('Algorithm:', c, '\tDataset:', d)
            try:
                if params[c] is not None:
                    params_names = sorted(params[c])
                    combinations = it.product(*(params[c][param] for param in params_names))
                    params_c = list(combinations)
                else:
                    params_names = []
                    params_c = []
                exp.run_experiment(data_dir, res_dir, c, d, params_c, params_names, overwrite=True)
            except:
                print('\n\n FAILED: ', sys.exc_info()[0], '\n\n')


if __name__ == "__main__":
    comparison_experiments(str(args.data), str(args.res))
