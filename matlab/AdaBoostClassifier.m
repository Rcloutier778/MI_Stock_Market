function [predict] = AdaBoostClassifier( classifier,localXdata)
% function [pred] = AdaBoostClassifier( classifier,localXdata)
%  Compute AdaBoost Classification across a set of training samples
% Input:
%   classifier- a struct containing the fields feature, thresh, and
%                  polarity, one entry per threshold
%   localXdata- the X values of input sample, nxD
% Output:
%   predict- prediction class [-1,+1] of each input sample
%
%  CMPE-677, Machine Intelligence
%  Base code by R. Ptucha, Andrew Gallagher 2014
%  Rochester Institute of Technology

%for every feature, find the best threshold
D = size(localXdata,2);  			%number of features 
n = size(localXdata,1);  			%number of samples
nh = length(classifier);             %the number of classifiers.
totalPred = zeros(n,1); 

%for each classifier, make a guess on the input X data
for i = 1:1:nh
   % apply the classifier
   % extract the current iterations classifier
   cc = classifier{i};
    

    %Make a prediction for each feature, threshold value, and polarity
    % The equation here is identical to to AdaBoostError.m, but now we also
    % need to multiply by the alpha value in our classifier entry
    predict = cc.alpha*(2*(localXdata(:,cc.feature) < cc.thresh) -1)*cc.polarity;
    
    
    %keep running sum of predict
    %totalPred = <insert code here>
    totalPred = totalPred+predict;
end

% now convert threshold to [-1 1]
% Convert totalPred positive values to +1 and negative totalPred values to -1
for i = 1:length(predict)
    if predict(i) > 0
        predict(i) = 1;
    else
        predict(i) = -1;
    end
end