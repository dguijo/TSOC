# -*- coding: utf-8 -*-
import argparse
import sys
import TSOC.utils.experiments as exp
from TSOC.utils.shapelets import writeShapeletsToCSV
from TSOC.transformers.ordST import ContractedOrdinalShapeletTransform
from sktime.transformers.shapelets import ContractedShapeletTransform
import numpy as np
import os
import itertools as it

from sktime.utils.load_data import load_from_tsfile_to_dataframe as load_ts

from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--timeseriesPath", "-t", type=str, default="/home/david/TSOC/timeseries/", help="Path to time series")
parser.add_argument("--datasetPath", "-p", type=str, default="/home/david/TSOC/datasets/", help="Path to datasets")
parser.add_argument("--datasetName", "-d", type=str, default="DistalPhalanxTW", help="Dataset name")
parser.add_argument("--extractShapelets", "-e", type=bool, default=True, help="Boolean to extract or not the shapelets")
parser.add_argument("--shp", "-s", type=str, default="Ordinal_1", help="Shapelet extraction approach used")
parser.add_argument("--res", "-r", type=str, default="/home/david/TSOC/results/", help="Path to save the results")
args = parser.parse_args()


def comparison_experiments(data_dir, res_dir, data_name, transform):
    complete_classifiers = [
        # Naive approaches
        "svr",
        "svc1va",
        "svc1v1",
        "cssvc",

        # Ordinal decomposition methods
        "svmop",
        # nnop,
        # elmop,
        "opbe",

        # threshold methods
        "pom",
        # nnpom,
        "kdlor",
        "svorex",
        "svorim",
        "svorimlin",
        "redsvm",
        "orboostall",
        "hpold",
    ]
    complete_classifiers = [
        # Naive approaches
        "svr",
        "svc1va",
    ]
    # Commented are non-deterministic.
    params = {
        # Naive approaches
        "svr": {'C': np.logspace(-3, 3, 7), 'k': np.logspace(-3, 3, 7), 'e': np.logspace(-3, 0, 4)},
        "svc1va": {'C': np.logspace(-3, 3, 7), 'k': np.logspace(-3, 3, 7)},
        "svc1v1": {'C': np.logspace(-3, 3, 7), 'k': np.logspace(-3, 3, 7)},
        "cssvc": {'C': np.logspace(-3, 3, 7), 'k': np.logspace(-3, 3, 7)},

        # Ordinal decomposition methods
        "svmop": {'C': np.logspace(-3, 3, 7), 'k': np.logspace(-3, 3, 7)},
        # "nnop": {'iter': [250, 500], 'lambda': [0.01, 0, 1], 'hiddenN': [5, 10, 20, 30, 40, 50]},
        # "elmop": {'hiddenN': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        "opbe": {'C': np.logspace(-3, 3, 7), 'k': np.logspace(-3, 3, 7)},

        # threshold methods
        "pom": [],
        # "nnpom": {'iter': [250, 500], 'lambda': [0.01, 0, 1], 'hiddenN': [5, 10, 20, 30, 40, 50]},
        "kdlor": {'C': np.logspace(-3, 3, 7), 'k': np.logspace(-3, 3, 7), 'u': np.logspace(-6, -2, 5)},
        "svorex": {'C': np.logspace(-3, 3, 7), 'k': np.logspace(-3, 3, 7)},
        "svorim": {'C': np.logspace(-3, 3, 7), 'k': np.logspace(-3, 3, 7)},
        "svorimlin": {'C': np.logspace(-3, 3, 7)},
        "redsvm": {'C': np.logspace(-3, 3, 7), 'k': np.logspace(-3, 3, 7)},
        "orboostall": [],
        "hpold": {'C': np.logspace(-3, 3, 7), 'k': np.logspace(-3, 3, 7)},
    }

    params = {
        # Naive approaches
        "svr": {'C': np.logspace(-1, 0, 2), 'k': np.logspace(-1, 0, 2), 'e': np.logspace(-1, 0, 2)},
        "svc1va": {'C': np.logspace(-1, 0, 2), 'k': np.logspace(-1, 0, 2)},
        "svc1v1": {'C': np.logspace(-1, 0, 2), 'k': np.logspace(-1, 0, 2)},
        "cssvc": {'C': np.logspace(-1, 0, 2), 'k': np.logspace(-1, 0, 2)},

        # Ordinal decomposition methods
        "svmop": {'C': np.logspace(-1, 0, 2), 'k': np.logspace(-1, 0, 2)},
        # "nnop": {'iter': [250, 500], 'lambda': [0.01, 0, 1], 'hiddenN': [5, 10, 20, 30, 40, 50]},
        # "elmop": {'hiddenN': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        "opbe": {'C': np.logspace(-1, 0, 2), 'k': np.logspace(-1, 0, 2)},

        # threshold methods
        "pom": [],
        # "nnpom": {'iter': [250, 500], 'lambda': [0.01, 0, 1], 'hiddenN': [5, 10, 20, 30, 40, 50]},
        "kdlor": {'C': np.logspace(-1, 0, 2), 'k': np.logspace(-1, 0, 2), 'u': np.logspace(-3, -2, 2)},
        "svorex": {'C': np.logspace(-1, 0, 2), 'k': np.logspace(-1, 0, 2)},
        "svorim": {'C': np.logspace(-1, 0, 2), 'k': np.logspace(-1, 0, 2)},
        "svorimlin": {'C': np.logspace(-1, 0, 2)},
        "redsvm": {'C': np.logspace(-1, 0, 2), 'k': np.logspace(-1, 0, 2)},
        "orboostall": [],
        "hpold": {'C': np.logspace(-1, 0, 2), 'k': np.logspace(-1, 0, 2)},
    }

    for c in complete_classifiers:
        print('Ordinal Classification Algorithm: ' + str(c))
        try:
            if params[c] is not None:
                params_names = sorted(params[c])
                combinations = it.product(*(params[c][param] for param in params_names))
                params_c = list(combinations)
            else:
                params_names = []
                params_c = []
            exp.run_experiment(data_dir, res_dir, c, data_name, transform, params_c, params_names, overwrite=True)
        except:
            print('\n\n FAILED: ', sys.exc_info()[0], '\n\n')


def shapelet_extraction(timeseries_dir, data_dir, data_name, shp_type):
    trainX, trainY = load_ts(timeseries_dir + data_name + '/' + data_name + '_TRAIN.ts')
    testX, testY = load_ts(timeseries_dir + data_name + '/' + data_name + '_TEST.ts')

    # Encoding of the labels from 1 to num_classes of the dataset.
    le = LabelEncoder()
    trainY = le.fit_transform(trainY)
    testY = le.transform(testY)
    trainY = trainY + 1
    testY = testY + 1
    if shp_type == "Standard":
        shp = ContractedShapeletTransform(time_limit_in_mins=0.5, random_state=0)
    elif shp_type == "Ordinal_1":
        shp = ContractedOrdinalShapeletTransform(time_limit_in_mins=0.5, random_state=0)
    else:
        shp = ContractedShapeletTransform(time_limit_in_mins=0.5, random_state=0)
    shp.fit(trainX, trainY)

    shapelets = shp.get_shapelets()
    writeShapeletsToCSV(shapelets, 60, data_dir + '/' + data_name + '/' + data_name + '_shapelets.csv')

    for i in range(70, 110, 10):
        directory = data_dir + '/' + data_name + '/' + 'transform_' + str(i) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        shp.shapelets = shapelets[:int((len(shapelets)*i)/100)]
        train_transform = shp.transform(trainX)
        test_transform = shp.transform(testX)
        train_transform['label'] = trainY
        test_transform['label'] = testY

        train_transform.to_csv(directory + '/' + data_name + '_train.0', header=None, index=None, sep=' ')
        test_transform.to_csv(directory + '/' + data_name + '_test.0', header=None, index=None, sep=' ')


if __name__ == "__main__":
    print("Dataset: " + args.datasetName)
    print("Shapelet Extraction Procedure: " + args.shp)

    final_dataset_path = args.datasetPath + args.shp
    final_results_path = args.res + args.shp + '/'

    if args.extractShapelets:
        shapelet_extraction(args.timeseriesPath, final_dataset_path, args.datasetName, args.shp)
    else:
        final_existing_shapelets_path = final_dataset_path + '/' + args.datasetName + '/' + args.datasetName
        if (not os.path.exists(final_existing_shapelets_path + '_shapelets.csv')):
            raise FileNotFoundError("Shapelets and transform not found. Please run with -e True")

    for i in range(70, 110, 10):
        comparison_experiments(final_dataset_path + '/', final_results_path, args.datasetName, i)