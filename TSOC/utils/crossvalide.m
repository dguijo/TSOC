function [ bestParam ] = crossvalide(algorithmObj, train, kFold, params, param_names)
%CROSSVALIDE Fucntion to perform automatic crossvalidation based on the train set
    CVO = cvpartition(train.targets,'KFold',kFold);
    bestAccuracy = 0;

    for i=1:length(params)
        param = struct();
        for j=1:length(param_names)
            param = setfield(param, string(param_names(j)), params(i, j));
        end
        accuracy = 0;
        for ff = 1:CVO.NumTestSets
            trIdx = CVO.training(ff);
            teIdx = CVO.test(ff);
            trainCV.patterns = train.patterns(trIdx, :);
            trainCV.targets = train.targets(trIdx, :);
            testCV.patterns = train.patterns(teIdx, :);
            testCV.targets = train.targets(teIdx, :);
            info = algorithmObj.fitpredict(trainCV, testCV, param);
            accuracy = accuracy + CCR.calculateMetric(testCV.targets, info.predictedTest);
        end
        accuracy = accuracy / kFold;
        disp(param)
        fprintf('SVORIM --> Accuracy: %f, Minimum Sensitivity: %f\n', accuracy, MS.calculateMetric(testCV.targets,info.predictedTest));
        if accuracy > bestAccuracy
            bestAccuracy = accuracy;
            bestParam = param;
        end
    end
end

%    for C=10.^(-3:1:3)
%        for k=10.^(-3:1:3)
%            accuracy = 0;
%            param = struct('C',C,'k',k);
%            for ff = 1:CVO.NumTestSets
%                trIdx = CVO.training(ff);
%                teIdx = CVO.test(ff);
%                trainCV.patterns = train.patterns(trIdx,:);
%                trainCV.targets = train.targets(trIdx,:);
%                testCV.patterns = train.patterns(teIdx,:);
%                testCV.targets = train.targets(teIdx,:);
%                info = algorithmObj.fitpredict(trainCV, testCV, param);
%                accuracy = accuracy + CCR.calculateMetric(testCV.targets,info.predictedTest);
%            end
%            accuracy = accuracy / kFold;
%            fprintf('SVORIM C %f, k %f --> Accuracy: %f, Minimum Sensitivity: %f\n', C, k, accuracy, MS.calculateMetric(testCV.targets,info.predictedTest));
%            if accuracy > bestAccuracy
%                bestAccuracy = accuracy;
%                bestParam = param;
%            end
%        end
%    end
%end

