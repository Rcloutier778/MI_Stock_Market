close all ; clear all; clc;
%Download hwk7files_forMycourses.zip and place them into an appropriate
%hwk7 directory. Update the two paths below for your machine
nntraintool('close')
warning('off','all')
warning
cd 'D:\Machine Intelligence\Term Project\matlab'
addpath 'D:\Machine Intelligence\HW6\libsvm-3.18\windows'
% We will use mnist hand written digits, '0' through '9'
files={'AMD','BAC','CVX','DOW','DUK','GOOGL','JPM','M','NASDAQ','PYPL','QQQ','S&P','V','VZ','XOM'};

%results={'Stock','SVM-Linear','SVM-Poly','SVM-Radial','SVM-sigmoid','nnet1','nnet2','ff1','ff2','ff3','lm','bfg','rp','cgb','cgf','cgp','oss','gdx'};
results={'Stock','SVM-Linear','SVM-Poly','SVM-Radial','SVM-sigmoid',...
    'LogRegress_lambda=0','LR_lambda=1','LR_lambda=0.1',...
    'LR_lambda=0.01','BaggedTree',...
    'nnet1','nnet2','ff1','ff2','ff3','nnet3','nnet4','ff4','ff5'};

parfor f = 1:size(files,2)
    str=['data/',files{f},'.txt'];
    data = load(str);

    X = data(:,[1,2,3,4,5,6]);
    y = data(:,7);

    n = size(X, 1); %number of samples = 5000, 500 from each class
    D = size(X,2); %number of dimensions/sample. 20x20=400
    C = length(unique(y)); %number of classes, class values are 1...10, where 10 is a digit '0'
    
    options=[];
    accuracies={files{f}};
    
    options.confusionMatrix = 'False';
    options.numberOfFolds = 1;
    
    options.method = 'SVM';
    options.svm_t = 0; %linear
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    options.method = 'SVM';
    options.svm_t = 1;%poly
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    options.method = 'SVM';
    options.svm_t = 2;%radial basis
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    options.method = 'SVM';
    options.svm_t = 3;%sigmoid
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
        
    
    options.method = 'LogisticRegression';
    options.lambda=0;
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    options.method = 'LogisticRegression';
    options.lambda=1;
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    options.method = 'LogisticRegression';
    options.lambda=0.1;
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    options.method = 'LogisticRegression';
    options.lambda=0.01;
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    options.method = 'BaggedTree';
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
        
    options.method = 'nnet';
    options.nnet_hiddenLayerSize = 25;
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    options.method = 'nnet';
    options.nnet_hiddenLayerSize = [25 10];
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    options.method = 'feedForward';
    options.nnet_hiddenLayerSize = 25;
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    options.method = 'feedForward';
    options.nnet_hiddenLayerSize = [25 10];
    accuracies = [accuracies, classify677_hwk7(X,y,options)];

    options.method = 'feedForward';
    options.nnet_hiddenLayerSize = [25 10 4];
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    
    
    options.method = 'nnet';
    options.nnet_hiddenLayerSize = round((size(X,1)+2)/2);
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    options.method = 'nnet';
    options.nnet_hiddenLayerSize = [round((size(X,1)+2)/2), round((size(X,1)+2)/5)];
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    
    options.method = 'feedForward';
    options.nnet_hiddenLayerSize = round((size(X,1)+2)/2);
    accuracies = [accuracies, classify677_hwk7(X,y,options)];


    options.method = 'feedForward';
    options.nnet_hiddenLayerSize = [round((size(X,1)+2)/2), round((size(X,1)+2)/5)];
    accuracies = [accuracies, classify677_hwk7(X,y,options)];
    
    
    options.method = 'feedForward';
    options.nnet_algo = 'trainlm';
    options.nnet_hiddenLayerSize = round((size(X,1)+2)/2);
    accuracies = [accuracies, classify677_hwk7(X,y,options)];

    options.method = 'feedForward';
    options.nnet_algo = 'trainbfg';
    options.nnet_hiddenLayerSize = round((size(X,1)+2)/2);
    accuracies = [accuracies, classify677_hwk7(X,y,options)];

    options.method = 'feedForward';
    options.nnet_algo = 'trainrp';
    options.nnet_hiddenLayerSize = round((size(X,1)+2)/2);
    accuracies = [accuracies, classify677_hwk7(X,y,options)];

    options.method = 'feedForward';
    options.nnet_algo = 'traincgb';
    options.nnet_hiddenLayerSize = round((size(X,1)+2)/2);
    accuracies = [accuracies, classify677_hwk7(X,y,options)];

    options.method = 'feedForward';
    options.nnet_algo = 'traincgf';
    options.nnet_hiddenLayerSize = round((size(X,1)+2)/2);
    accuracies = [accuracies, classify677_hwk7(X,y,options)];

    options.method = 'feedForward';
    options.nnet_algo = 'traincgp';
    options.nnet_hiddenLayerSize = round((size(X,1)+2)/2);
    accuracies = [accuracies, classify677_hwk7(X,y,options)];

    options.method = 'feedForward';
    options.nnet_algo = 'trainoss';
    options.nnet_hiddenLayerSize = round((size(X,1)+2)/2);
    accuracies = [accuracies, classify677_hwk7(X,y,options)];

    options.method = 'feedForward';
    options.nnet_algo = 'traingdx';
    options.nnet_hiddenLayerSize = round((size(X,1)+2)/2);
    accuracies = [accuracies, classify677_hwk7(X,y,options)];


    
    results(f+1,:) = accuracies;
    
end

max=zeros(1,size(results,2));
nmax={'Highest:'};
avg={'Average:'}
for i = 2:size(results,2)
    tavg=0;
    for k=2:size(results,1)
       tavg=tavg+cell2mat(results(k,i));
       if cell2mat(results(k,i)) > max(i)
           max(i)=cell2mat(results(k,i));
           nmax(i)=results(k,1);
       end
    end
    avg{1,i}=tavg/(size(results,1)-1);
    
end
tmax=mat2cell(max,1,size(results,2));
results(size(results,1)+1,:)=nmax;
tlen=size(results,1)+1;
for i = 2:size(results,2)
    results(tlen,i)=num2cell(max(1,i));
end

results(size(results,1)+1,:)=avg;


results















