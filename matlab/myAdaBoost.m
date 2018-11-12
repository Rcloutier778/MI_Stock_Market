function [classifiers, errors,pred] = myAdaBoost(TrainXdata,TrainGT,adaboost_numFeatures,TestXdata,TestGT)
%  function [classifiers, errors,pred] = myAdaBoost(TrainXdata,TrainGT,adaboost_numFeatures,TestXdata,TestGT)
%  Implements the AdaBoost algorithm for decsion stump trees
% Input:
%   TrainXdata- the X values of input training samples, nxD
%   TrainGT- the ground truth for each input training samples, nx1
%   adaboost_numFeatures- number of thresholds to solve for
%   TestXdata- the X values of input testing samples, nxD
%   TestGT- the ground truth for each input testing samples, nx1
% Output:
%   classifiers- vector of classifiers, each being a struct containing the
%                    fields alpha, feature, performance, polarity,and
%                    and thresh
%   errors- AdaBoost classification errors for test and train set
%   pred- AdaBoost classification result for test and train set
%
%  CMPE-677, Machine Intelligence
%  Base code by R. Ptucha, Andrew Gallagher 2014
%  Rochester Institute of Technology


%for every feature, find the best threshold
D = size(TrainXdata,2);  %number of features , fn
n = size(TrainXdata,1);  %number of samples, fs
nlist = [1:n]'; 
  

%Each sample gets a weight r
weights = TrainGT.*0+1./n;   %equal weights to start

for i = 1:1:adaboost_numFeatures
    %each iteration creates a classifier. 
    h1=decisionStumpW(weights,TrainXdata,TrainGT)    ;                %train base learner
    [errorAmt, alpha] = AdaBoostError(weights, h1,TrainXdata, TrainGT); %determine alpha
    h1.alpha = alpha;
    [newWeights,zz] = AdaBoostUpdateWeights(weights,h1,TrainXdata,TrainGT);    %update the weights
    h1.z = zz;                                                  %theoretical error bound
    classifiers{i} = h1;
    
    %Document the performance
    trainPred = AdaBoostClassifier(classifiers,TrainXdata);
    trainErr(i)= sum(trainPred~=TrainGT)/n;
    testPred = AdaBoostClassifier(classifiers,TestXdata);
    testErr(i)= sum(testPred~=TestGT)/size(TestXdata,1);
    if(i==1) errBound(i)= zz;
    else errBound(i) = zz*errBound(i-1);
    end
    
    weights = newWeights;
end

errors.train = trainErr;
errors.test = testErr;
errors.eb = errBound;

pred.train = trainPred;
pred.test = testPred;

