function [bestParam] = crossvalide(algorithmObj, train, kFold, params, param_names)
%CROSSVALIDE Function to perform automatic crossvalidation based on the train set
    CVO = cvpartition(train.targets,'KFold',kFold);
    bestmetric = 0;
    bestParam = [];
    for i=1:length(params)
        param = struct();
        for j=1:length(param_names)
            param = setfield(param, string(param_names(j)), params(i, j));
        end
        metric = 0;
        for ff = 1:CVO.NumTestSets
            trIdx = CVO.training(ff);
            teIdx = CVO.test(ff);
            trainCV.patterns = train.patterns(trIdx, :);
            trainCV.targets = train.targets(trIdx, :);
            testCV.patterns = train.patterns(teIdx, :);
            testCV.targets = train.targets(teIdx, :);
            info = algorithmObj.fitpredict(trainCV, testCV, param);
            metric = metric + (1/(1+AMAE.calculateMetric(testCV.targets, info.predictedTest)));
        end
        metric = metric / kFold;
        %fprintf('metric: %f, Minimum Sensitivity: %f\n', accuracy, MS.calculateMetric(testCV.targets,info.predictedTest));
        %disp(param)
        if metric > bestmetric
            bestmetric = metric;
            bestParam = param;
        end
    end
end