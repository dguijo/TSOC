import pandas as pd
import matlab.engine
import matlab
import numpy as np

# Read the train and test files
train = pd.read_csv('../orca-master/exampledata/1-holdout/toy/matlab/train_toy.0', sep=' ', header=None)
test = pd.read_csv('../orca-master/exampledata/1-holdout/toy/matlab/test_toy.0', sep=' ', header=None)

original_data = {"train": {"patterns": train.iloc[:, :-1], "targets": train.iloc[:, -1]},
                 "test": {"patterns": test.iloc[:, :-1], "targets": test.iloc[:, -1]}}

# Divide into X and Y
trainX = matlab.double(train.iloc[:, :-1].values.tolist())
trainY = matlab.double([[i] for i in train.iloc[:, -1].values.tolist()])
testX = matlab.double(test.iloc[:, :-1].values.tolist())
testY = matlab.double([[i] for i in test.iloc[:, -1].values.tolist()])

eng = matlab.engine.start_matlab()  # Para tenerlo en el background matlab.engine.start_matlab(background=True)

eng.addpath('../orca-master/src/Algorithms/')
eng.addpath('../orca-master/src/Measures/')

kdlorAlgorithm = eng.KDLOR('kernelType', 'rbf', 'optimizationMethod', 'quadprog')

tr = {"patterns": trainX, "targets": trainY}
te = {"patterns": testX, "targets": testY}
param = eng.struct('C', 10.0, 'k', 0.1, 'u', 0.01)

# The Matlab object should be the first param.
classifier_info = eng.fitpredict(kdlorAlgorithm, tr, te, param)

cm = eng.confusionmat(te["targets"], classifier_info["predictedTest"])

metrics = {"cm": np.array(cm._data).reshape(cm.size[::-1]).T, "ccr": eng.CCR.calculateMetric(cm),
              "mae": eng.MAE.calculateMetric(cm), "amae": eng.AMAE.calculateMetric(cm),
              "wkappa": eng.Wkappa.calculateMetric(cm), "ms": eng.MS.calculateMetric(cm),
              "gm": eng.GM.calculateMetric(cm)}

eng.quit()


print(metrics)