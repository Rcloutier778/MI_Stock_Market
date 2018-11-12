function [classifier, performance] = decisionStumpW(weights,features, classes)
% input:
% weights. n x 1 matrix of weights
% features. n x r matrix, where each of n rows is a data sample with r
% feature values
% classes. nx1 class values. I will force to -1 and 1 if they aren't
% already
%   output:
%   classifier a structure indicating the selected feature, the threshold
%   and the classes (for above and below the threshold
% perf is the performance of the classifier. 
%
%  CMPE-677, Machine Intelligence
%  Base code by Andrew Gallagher 2014
%  Rochester Institute of Technology
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  NEED TO ADD THE 
%%%%%%  SUPPORT FOR COUNTING THE WEIGHTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mc = min(classes);
xc = max(classes)-mc; 
classes = 2*(classes-mc)/xc-1; %-1 and 1
ccc = classes;
ccc(ccc==-1)=0;    %now 0 and 1. 
nz = sum(ccc==0); %number of class 0
no = sum(ccc==1); %number of class 1 

%for every feature, find the best threshold
fn = size(features,2);  %number of features 
fs = size(features,1);  %number of samples
nlist = [1:1:fs]'; 

for i = 1:1:fn
   % i
    %for each feature
    [tc,ii] =sort(features(:,i));  
    labs = ccc(ii); %sorted labels 
    wts  = weights(ii); 
    %see how well the classification went...
    % quickly count the number of correct classifications
    % for each threshold... 
    % do a cumulative sum. 
    cc = cumsum(labs.*wts) ;  %cumulative score for classifying 1s
    zz = flipud(cumsum(flipud((1-labs).*wts))); %cumulative score for classifying zeros
    bz = zz(1); % score for classifying zeros, threshold to the left of the first datapt. 
    zz = [zz(2:end); 0];
    
    
    %%%%% classify as 1 below the threshold!
    %numCorrect= cc + nz + (cc - nlist);%number correct as a function of threshold
    numCorrect= cc + zz;%number correct as a function of threshold
    
    % will need to change to accomadate the weights... 
    % now pick the best... I want the classification that is the furthest
    % from 50%!! 
    nc = abs(numCorrect-(cc(end)+ bz)/2);
    indicator = ([diff(tc); 1]>0); %% shows where categories end
    nc = nc.*indicator; 
    
   %% [tc cc zz numCorrect nc]
    %labs
    %numCorrect
    %nc
    [mmm,iii] = max(nc); % pick out the most extreme
    flag(i) = sign(numCorrect(iii) - (cc(end)+ bz)/2);  % tells the classification direction
    %numCorrect(iii)
    if(iii<fs)
        thresh(i) = .5*(tc(iii)+ tc(iii+1));
    else 
        thresh(i) = tc(iii) + .0001; 
    end
    
    % check the endpoints to see if all should be classified one way...
    all1 = cc(end); 
    all0 = zz(end); 
    if(all1>mmm+.5)
        %classify everything as 1. 
        thresh(i) = tc(end) + .0001;
        flag(i) = 1; 
        mmm = all1-.5; 
    elseif (all0>mmm+.5)
        thresh(i) = tc(1) +.0001;
        flag(i) = -1;
        mmm = all0-.5; 
    end
   % mmm
    perf(i) = mmm+.5; 
end
%[flag;
%perf;
%thresh]
% now finding the best classifier is easy: 
[mmm,iii] = max(perf); 
%%% iii is the best classifier. 
classifier.feature  = iii;
classifier.thresh   = thresh(iii); 
classifier.polarity = flag(iii); 
classifier.performance =perf(iii); 
performance = perf(iii); 