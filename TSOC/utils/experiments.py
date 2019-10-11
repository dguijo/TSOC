import pandas as pd
import matlab.engine
import matlab
import numpy as np
import os
import time
import pickle


def set_classifier(cls_name, eng):
    if cls_name.lower() == 'kdlor':  # KDLOR
        return eng.KDLOR('kernelType', 'rbf', 'optimizationMethod', 'quadprog')
    elif cls_name.lower() == 'pom':  # POM
        return eng.POM()
    elif cls_name.lower() == 'svorim':  # SVORIM
        return eng.SVORIM()
    else:
        return 'UNKNOWN CLASSIFIER'


def load_datasets(data_dir, dset_name, rtn_format='matlab'):
    # Read the train and test files
    train = pd.read_csv(data_dir + dset_name + '/train_' + dset_name + '.0', sep=' ', header=None)
    test = pd.read_csv(data_dir + dset_name + '/test_' + dset_name + '.0', sep=' ', header=None)

    data = {'train': {'patterns': train.iloc[:, :-1], 'targets': train.iloc[:, -1]},
            'test': {'patterns': test.iloc[:, :-1], 'targets': test.iloc[:, -1]}}

    if rtn_format == 'matlab':
        # Divide into X and Y
        trainX = matlab.double(train.iloc[:, :-1].values.tolist())
        trainY = matlab.double([[i] for i in train.iloc[:, -1].values.tolist()])
        testX = matlab.double(test.iloc[:, :-1].values.tolist())
        testY = matlab.double([[i] for i in test.iloc[:, -1].values.tolist()])

        tr = {'patterns': trainX, 'targets': trainY}
        te = {'patterns': testX, 'targets': testY}

        return tr, te, data

    return data['train'], data['test'], data


def save_results(res_dir, cls_name, dset_name, classifier_info, metrics):

    # Define the dir where all the results-related stuff is saved
    output_path = str(res_dir) + str(cls_name) + '/' + str(dset_name) + '/'
    metrics_path = output_path + 'Metrics/'
    predictions_path = output_path + 'Predictions/'
    hyperparams_path = output_path + 'Hyperparams/'
    guess_path = output_path + 'Guess/'
    model_path = output_path + 'Model/'

    for i in [output_path, metrics_path, predictions_path, hyperparams_path, guess_path, model_path]:
        try:
            os.makedirs(i)
        except os.error:
            pass  # raises os.error if path already exists

    # Include metrics and times in the metrics file
    file = open(metrics_path + 'metrics.csv', 'w')
    file.write(str(dset_name) + ',' + str(cls_name) + '\n')

    for i in metrics.keys():
        if i != 'cm':
            file.write(str(i) + ',' + str(metrics[i]) + '\n')

    file.write('\n')
    file.write('trainTime,' + str(classifier_info['trainTime']) + '\n')
    file.write('testTime,' + str(classifier_info['testTime']) + '\n')

    file.close()

    # Save confusion matrix in the confmat.csv filenp.array(classifier_info['predictedTrain']._data)
    np.savetxt(metrics_path + 'confmat.csv', metrics['cm'], delimiter=',')

    # Save params used in the hyperparams file
    file = open(hyperparams_path + 'params.csv', 'w')
    file.write(str(dset_name) + ',' + str(cls_name) + '\n')

    try:
        for i in classifier_info['model']['parameters'].keys():
            file.write(str(i) + ',' + str(classifier_info['model']['parameters'][i]) + '\n')
    except KeyError:
        pass

    file.close()

    # Save predictions in two separates files, one for train and one for test
    np.savetxt(predictions_path + 'train.csv', np.array(classifier_info['predictedTrain']._data), delimiter=',',
               fmt='%d')
    np.savetxt(predictions_path + 'test.csv', np.array(classifier_info['predictedTest']._data), delimiter=',', fmt='%d')

    # Save projections in two separates files, one for train and one for test
    np.savetxt(guess_path + 'train.csv', np.array(classifier_info['projectedTrain']._data), delimiter=',')
    np.savetxt(guess_path + 'test.csv', np.array(classifier_info['projectedTest']._data), delimiter=',')

    # Save other important information in a pickle

    with open(model_path + '/classifier_info.p', 'wb') as handle:
        pickle.dump(classifier_info['model'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    # to unpickle (matlab must be imported beforehand)
    # with open('filename.pickle', 'rb') as handle:
    # var = pickle.load(handle)


def run_experiment(data_dir, res_dir, cls_name, dset_name, cls_params, param_names, overwrite):
    if not overwrite:
        full_path = str(res_dir) + str(cls_name) + '/Predictions/' + str(dset_name) + '/metrics.csv'
        if os.path.exists(full_path):
            print(full_path + ' already exists and overwrite set to false')
            return

    eng = matlab.engine.start_matlab()  # Para tenerlo en el background matlab.engine.start_matlab(background=True)

    #eng.warning('off', 'all')

    eng.addpath('../orca-master/src/Algorithms/')
    eng.addpath('../orca-master/src/Measures/')
    eng.addpath('../TSOC/utils/')

    train, test, original_data = load_datasets(data_dir, dset_name, 'matlab')

    classifier = set_classifier(cls_name, eng)
    print(matlab.double(cls_params))
    best_params = eng.crossvalide(classifier, train, 2.0, matlab.double(cls_params), param_names)
    print(best_params)

    # The Matlab object should be the first param.
    classifier_info = eng.fitpredict(classifier, train, test, best_params)

    cm = eng.confusionmat(test['targets'], classifier_info['predictedTest'])

    metrics = {'cm': np.array(cm._data).reshape(cm.size[::-1]).T, 'ccr': eng.CCR.calculateMetric(cm),
                  'mae': eng.MAE.calculateMetric(cm), 'amae': eng.AMAE.calculateMetric(cm),
                  'wkappa': eng.Wkappa.calculateMetric(cm), 'ms': eng.MS.calculateMetric(cm),
                  'gm': eng.GM.calculateMetric(cm)}

    eng.quit()

    save_results(res_dir, cls_name, dset_name, classifier_info, metrics)