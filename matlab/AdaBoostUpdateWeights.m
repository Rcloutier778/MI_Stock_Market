function [newweights,zt] = AdaBoostUpdateWeights(inweights, classifier, localXdata, localGT)
%function [newweights,zt] = AdaBoostUpdateWeights(inweights, classifier, localXdata, localGT)
%  Update weights for each iteration of AdaBoost
% Input:
%   inweights- one weight per input sample, nx1
%   classifier- a struct containing the fields feature, thresh, and polarity
%   localXdata- the X values of input sample, nxD
%   localGT- the ground truth for each input sample, nx1
% Output:
%   newweights- updated weights, one weight per input sample
%   zt- sum of newweights
%
%  CMPE-677, Machine Intelligence
%  Base code by R. Ptucha, Andrew Gallagher 2014
%  Rochester Institute of Technology


%get the weighting of the classifier
%call AdaBoostError to get alpha and predict
[err, alpha,predict] = AdaBoostError( inweights, classifier, localXdata, localGT);


%calculate new weights
newweights = inweights.*exp(-alpha.*predict.*localGT);


%normalize new weights
zt = sum(newweights); 
newweights = newweights./zt; 