function [errorAmt, alpha,predict] = AdaBoostError(weights, classifier,localXdata, localGT)
% function [errorAmt, alpha] = AdaBoostError(weights, classifier,localXdata, localGT)
%  Compute AdaBoost Error across a set of training samples
% Input:
%   weights- one weight per input sample, nx1
%   classifier- a struct containing the fields feature, thresh, and polarity
%   localXdata- the X values of input sample, nxD
%   localGT- the ground truth for each input sample, nx1
% Output:
%   errorAmt- current error
%   alpha- AdaBoost alpha value
%   predict- prediction class [-1,+1] of each input sample
%
%  CMPE-677, Machine Intelligence
%  Base code by R. Ptucha, Andrew Gallagher 2014
%  Rochester Institute of Technology

%make a predition using the classifier on our data
%predict = (2*(Xdata(:,feature) < T) -1)*polarity
predict = (2*(localXdata(:,classifier.feature) < classifier.thresh) -1)*classifier.polarity;

%form a [0,1] vector mistakes, mistakes will have one entry for each
%training sample
mistakes = (predict~=localGT);   			% 1 for mistakes

%weigh these mistakes
wmistakes = mistakes.*weights; 			%weighted mistakes

%sum of weighted mistakes
errsum = sum(wmistakes); 

%relative error
errorAmt = errsum/sum(weights); 

%alpha as per AdaBoost
alpha = 0.5*log((1-errorAmt)./errorAmt);



