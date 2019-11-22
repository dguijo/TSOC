__author__ = "David Guijo-Rubio"

import numpy as np
import csv
import os
from collections import Counter
from sktime.transformers.shapelets import ShapeletTransform, ContractedShapeletTransform, Shapelet


def getShapeletsFromFile(path_to_shapelets, *argv):
    if len(argv) == 0:  # Extract all the shapelets of the file.
        number_of_shapelets = np.inf
    elif len(argv) == 1:  # Extract a fixed number of params.
        number_of_shapelets = argv[0]
    else:
        raise ValueError("Check the params of getShapelets, only 1 extra param is allowed.")

    list_original_shapelets = []
    with open(path_to_shapelets) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif line_count % 2 == 0:
                shp_info_gain = row[0]
                shp_series_id = row[1]
                shp_start_pos = row[2]
            elif line_count % 2 == 1:
                row2 = list(filter(None, row))
                shp_data = [float(i) for i in row2]
                shp_length = len(shp_data)
                list_original_shapelets.append(Shapelet(shp_series_id, shp_start_pos, shp_length, shp_info_gain,
                                                        np.array([shp_data])))
                if len(list_original_shapelets) == number_of_shapelets:
                    break
            line_count += 1

    csv_file.close()

    return list_original_shapelets


def getTransform(X, list_shapelets):
    if list_shapelets is None:
        raise Exception("Fit not called yet or no shapelets were generated")

    X = np.array([[X.iloc[r, c].values for c in range(len(X.columns))] for r in range(len(X))])

    output = np.zeros([len(X), len(list_shapelets)], dtype=np.float32, )

    # for the i^th series to transform
    for i in range(0, len(X)):
        this_series = X[i]

        # get the s^th shapelet
        for s in range(0, len(list_shapelets)):
            # find distance between this series and each shapelet
            min_dist = np.inf
            this_shapelet_length = list_shapelets[s].length

            for start_pos in range(0, len(this_series[0]) - this_shapelet_length + 1):
                comparison = ContractedShapeletTransform.zscore(
                    this_series[:, start_pos:start_pos + this_shapelet_length])

                dist = np.linalg.norm(list_shapelets[s].data - comparison)
                dist = dist * dist
                dist = 1.0 / this_shapelet_length * dist
                min_dist = min(min_dist, dist)

                output[i][s] = min_dist
    return output


def writeTransformToCSV(X, y, list_shapelets, path, train_or_test="train"):
    """Transforms X according to the extracted shapelets (self.shapelets)

            Parameters
            ----------
            X : pandas DataFrame
                The input dataframe to transform
            y : list
                The input labels
            list_shapelets: original or tunned list of shapelets
                the list of shapelets to be printed to the csv
            path: str
                where to include the file
            train_or_test: train or test
                whether the transformation is for train o test.
            Returns
            -------
            output : pandas DataFrame
                The transformed dataframe in tabular format.
            """

    output = getTransform(X, list_shapelets)

    file_name = path + train_or_test + ".arff"
    # Create directory in case it doesn't exists
    directory = '/'.join(file_name.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_shapelets = output.shape[1]
    unique_labels = np.unique(y).tolist()

    with open(file_name, 'w+') as f:
        # Headers
        f.write("@Relation " + train_or_test + '\n\n')
        for i in range(num_shapelets):
            f.write("@attribute Shapelet_" + str(i) + " numeric\n")
        f.write("@attribute target " + str(unique_labels).replace('[', '{').replace(']', '}\n'))
        f.write("\n@data\n")
        for i in range(output.shape[0]):
            f.write(",".join(map(str, output[i][:])) + "," + str(y[i]) + "\n")
    f.close()
    return output


def writeShapeletsToCSV(shapelets, time, file_name):
    """ A simple function to save the shapelets obtained in csv format
    Parameters
    ----------
    shapelets: array-like
        The shapelets obtained for a dataset
    time: fload
        The time spent obtaining shapelets
    file_name: string
        The directory to save the set of shapelets
    """

    # Create directory in case it doesn't exists
    directory = '/'.join(file_name.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, 'w+') as f:
        # Number of shapelets and time extracting

        f.write("InformationGain,SeriesID,StartingPos,Length,NumShapelets: " + str(len(shapelets)) +
                ",TimeExtraction: " + str(time) + "\n")
        for i, j in enumerate(shapelets):
            f.write(str(j.info_gain) + "," + str(j.series_id) + "," + str(j.start_pos) + "," + str(j.length) + "\n")
            f.write(",".join(map(str, j.data[0])) + "\n")
    f.close()


def updateIG(X, y, shp):
    sz = X.shape[2]

    class_counts = dict(Counter(y))
    binary_ig_this_class_count = class_counts[y[int(shp.series_id)]] - 1
    binary_ig_other_class_count = len(y) - binary_ig_this_class_count - 1
    orderline = []
    for i in range(len(X)):
        if i == int(shp.series_id):
            continue
        if y[i] == y[int(shp.series_id)]:
            binary_class_identifier = 1  # positive for this class
        else:
            binary_class_identifier = -1  # negative for any other class
        bsf_dist = np.inf
        for start in range(0, sz - shp.length + 1):
            comparison = ShapeletTransform.zscore(X[i][:, start: start + shp.length])
            dist = np.linalg.norm(shp.data - comparison)
            bsf_dist = min(dist * dist, bsf_dist)
        orderline.append((bsf_dist, binary_class_identifier))
        orderline.sort()
    return ShapeletTransform.calc_binary_ig(orderline, binary_ig_this_class_count, binary_ig_other_class_count)
