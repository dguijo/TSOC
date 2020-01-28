function [bestParam] = crossvalide(algorithmObj, train, kFold, params, param_names)
%CROSSVALIDE Function to perform automatic crossvalidation based on the train set
    CVO = cvpartition(train.targets,'KFold',kFold);
    %disp(CVO);
    bestmetric = Inf;
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
            %disp("He llegado hasta el fitpredict");
            %save('train.mat', 'trainCV', '-mat');
            %save('test.mat', 'testCV', '-mat');
            info = algorithmObj.fitpredict(trainCV, testCV, param);
            %disp("He pasado el fitpredict");
            metric = metric + AMAE.calculateMetric(testCV.targets, info.predictedTest);
        end
        metric = metric / kFold;
        %fprintf('metric: %f\n', metric);
        %disp(param);
        if metric < bestmetric
            bestmetric = metric;
            bestParam = param;
        end
    end
    %disp("-----------------------------------------------------")
end
