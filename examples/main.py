# -*- coding: utf-8 -*-
import argparse
import sys
import TSOC.utils.experiments as exp

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
        "kdlor": {'C': 10.0, 'k': 0.1, 'u': 0.01},
        "pom": [],
        "svorim": {'C': 10.0, 'k': 0.1},
    }

    for d in small_datasets:
        for c in complete_classifiers:
            print('Algorithm:', c, '\tDataset:', d)
             #try:
            exp.run_experiment(data_dir, res_dir, c, d, params[c], overwrite=True)
        #    except:
            #print('\n\n FAILED: ', sys.exc_info()[0], '\n\n')


if __name__ == "__main__":
    comparison_experiments(str(args.data), str(args.res))
