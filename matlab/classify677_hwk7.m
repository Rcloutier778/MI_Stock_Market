function accuracy =  classify677_hwk7(X,y,options)
%function [confusionMatrix,accuracy] =  classify677_hwk7(Xdata,y,options)
% Input:
%      Xdata: A vector of feature, nxD, one set of attributes for each
%                sample (n samples x D dimensional data)
%      y: n x 1 ground truth labeling
%      options.method = {'LogisticRegression', 'KNN', 'SVM','ClassificationTree',
%                                 'BaggedTree', 'Boosting','Adaboost','nnet'}
%      options.numberOfFolds- the number of folds for k-fold cross validation
%      options.lambda - regularization lamda used in LogisticRegression
%      options.knn_k - number of nearest neighbors in KNN
%      options.adaboost_numFeatures- number of decision splits used in boosting
% Output:
%      confusionMatrix- an overall confusion matrix
%      accuracy- an overall accuracy value
%
%  CMPE-789, Machine Intelligence
%  R. Ptucha, 2014
%  Rochester Institute of Technology

[n,D] = size(X);
if length(y) ~= n
    error('X is nxD, and y is nx1');
end
numClasses = length(unique(y));
C = numClasses;

if (~exist('options','var'))
   options = [];
end

if  isfield(options,'method')
    method=options.method;
else
    method='KNN';
end
    
if strcmp(options.method,'LogisticRegression')
    Xdata = [ones(n, 1) X];
else
    Xdata = X;
end

if  isfield(options,'numberOfFolds')
   numberOfFolds=options.numberOfFolds;
else
    numberOfFolds =2;
end


rng(2000);  %random number generator seed so results are repeatable
%Generate a fold value for each training sample
CVindex = crossvalind('Kfold',y, numberOfFolds);
i=1;  %this is for easier debugging....

for i = 1:numberOfFolds
    
    if (numberOfFolds == 1)  %then test = train set- be careful here!
        %Get train and test index values
        %TestIndex = find(CVindex == i);
        %TrainIndex = find(CVindex == i);
        splitIndex = round(size(Xdata,1)*0.8);
        
        %for train and test partitions
        TrainXdata = Xdata(1:splitIndex,:);
        TrainGT =y(1:splitIndex);
        TestXdata = Xdata(splitIndex:size(y,1),:);
        TestGT = y(splitIndex:size(y,1));
    else
        %Get train and test index values
        TestIndex = find(CVindex == i);
        TrainIndex = find(CVindex ~= i);
        %for train and test partitions
        TrainXdata = Xdata(TrainIndex,:);
        TrainGT =y(TrainIndex);
        TestXdata = Xdata(TestIndex,:);
        TestGT = y(TestIndex);
    end
    
    
    
    %
    %build the model using TrainXdata and TrainGT
    %test the built model using TestXdata
    %
    switch method
        case 'LogisticRegression'
            
            if  isfield(options,'lambda')
                lambda = options.lambda;
            else
                lambda = 0;
            end
            
            % for Logistic Regression, we need to solve for theta
            % Initialize fitting parameters
            all_theta = zeros(numClasses, size(Xdata, 2));
            
            for c=1:numClasses
                % Set Initial theta
                initial_theta = zeros(size(Xdata, 2), 1);
                % Set options for fminunc
                opts = optimset('GradObj', 'on', 'MaxIter', 50);
                
                % Run fmincg to obtain the optimal theta
                % This function will return theta and the cost
                [theta] = ...
                    fmincg (@(t)(costFunctionLogisticRegression(t, TrainXdata, (TrainGT == c), lambda)), ...
                    initial_theta, opts);
                
                all_theta(c,:) = theta;
            end          
            
            % Using TestDataCV, compute testing set prediction using
            % the model created
            % for Logistic Regression, the model is theta
            all_pred = sigmoid(TestXdata*all_theta');
            [maxVal,maxIndex] = max(all_pred,[],2);
            TestDataPred=maxIndex;
            
        case 'KNN'
            if  isfield(options,'knn_k')
                knn_k = options.knn_k;
            else
                knn_k = 1;
            end
            [idx, dist] = knnsearch(TrainXdata,TestXdata,'k',knn_k);
            nnList=[];
            for i=1:knn_k
                nnList = [nnList TrainGT(idx(:,i))];
            end
            TestDataPred=mode(nnList')';
            
         case 'SVM'
             %Note- this is libsvm not the built-in svm functions to matlab
             if  isfield(options,'svm_t')
                svm_t = options.svm_t;
             else
                 svm_t = 0;
             end
             if  isfield(options,'svm_c')
                 svm_C = options.svm_c;
             else
                 svm_C = 1.0;
             end
             
             if  isfield(options,'svm_g')
                 svm_g = options.svm_g;
             else
                 svm_g = 1/n; %auto in python
             end
%             -s svm_type : set type of SVM (default 0)
%                 0 -- C-SVC
%                 1 -- nu-SVC
%                 2 -- one-class SVM
%                 3 -- epsilon-SVR
%                 4 -- nu-SVR
%             -t kernel_type : set type of kernel function (default 2)
%                 0 -- linear: u'*v
%                 1 -- polynomial: (gamma*u'*v + coef0)^degree
%                 2 -- radial basis function: exp(-gamma*|u-v|^2)
%                 3 -- sigmoid: tanh(gamma*u'*v + coef0)
%             -d degree : set degree in kernel function (default 3)
%             -g gamma : set gamma in kernel function (default 1/num_features)
%             -r coef0 : set coef0 in kernel function (default 0)
%             -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
%             -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
%             -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
%             -m cachesize : set cache memory size in MB (default 100)
%             -e epsilon : set tolerance of termination criterion (default 0.001)
%             -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
%             -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
%             -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
               

             %-m = 200 MB
             %tol = 0.000001
             [TrainXdataNorm, mu, sigma] = featureNormalize(TrainXdata,0,0);
             eval(['model = svmtrain(TrainGT,TrainXdataNorm,''-t  ' num2str(svm_t)  ' -c ' num2str(svm_C) ' -g ' num2str(svm_g) ' -e 0.0000001 -m 200 '' );']);
             
             [TestXdataNorm, mu, sigma] = featureNormalize(TestXdata,mu, sigma);
             TestDataPred = svmpredict( TestGT, TestXdataNorm, model, '-q');

             
         case 'ClassificationTree'
            tree = ClassificationTree.fit(TrainXdata,TrainGT);
            TestDataPred = predict(tree,TestXdata);
            
         case 'BaggedTree'
            rng(2000);  %random number generator seed
            t = ClassificationTree.template('MinLeaf',1);
            bagtree = fitensemble(TrainXdata,TrainGT,'Bag',10,t,'type','classification');
            TestDataPred = predict(bagtree,TestXdata);  %really should test with a test set here
        
        case 'ConcreteChallenge'
            rng(2000);
            %templateTree regression lsboost resample
            t = RegressionTree.template('MinLeaf',1);
            learnrate = 0.0245;
            samples=10;%41;
            resample='off';%'on';
            bagtree = fitensemble(TrainXdata,TrainGT,'LSBoost',samples,t,'type','regression','LearnRate',learnrate, 'Resample',resample);
            TestDataPred = predict(bagtree,TestXdata);  %really should test with a test set here
            
        case 'Adaboost'
            if  isfield(options,'adaboost_numFeatures')
                adaboost_numFeatures = options.adaboost_numFeatures;
            else
                adaboost_numFeatures = round(n/2);
            end
            
            if numClasses ~= 2
                error('Adaboost only works with two class data');
            end
            %change class labels to -1 and +1
            yList = unique(TrainGT);
            if yList(1) ~= -1
                TrainGT(TrainGT==yList(1))=-1;
                TrainGT(TrainGT==yList(2))= 1;
                TestGT(TestGT==yList(1))=-1;
                TestGT(TestGT==yList(2))= 1;
            end
            
           [classifiers, errors,pred] = myAdaBoost(TrainXdata,TrainGT,adaboost_numFeatures,TestXdata,TestGT);
           allTrain(i,:) = errors.train;
           allTest(i,:) = errors.test;
           allEB(i,:) = errors.eb;
           
           TestDataPred = pred.test;
           TestDataPred(TestDataPred==1)=yList(2);
           TestDataPred(TestDataPred==-1)=yList(1);

        case 'nnet'
            if  isfield(options,'nnet_hiddenLayerSize')
                hiddenLayerSize=options.nnet_hiddenLayerSize;
            else
                hiddenLayerSize =10;
            end

            %Convert X and y data into Matlab nnet format:
            inputs = TrainXdata';%%
            
            %Convert to one-hot encoding ground truth values
            targets = zeros(C,size(TrainGT,1));%%
            for ii=1:size(TrainGT,1)
                targets(TrainGT(ii),ii)=1;
            end
            
            % Create a Pattern Recognition Network
            setdemorandstream(2014784333);   %seed for random number generator
            net = patternnet(hiddenLayerSize);
            
            % Set up Division of Data for Training, Validation, Testing
            net.divideParam.trainRatio = 0.8;  %note- splits are done in a random fashion
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0.0;
            net.trainParam.showCommandLine=false;
            net.trainParam.show = NaN;
            
            % Train the Network
            [net,tr] = train(net,inputs,targets);%%  %return neural net and a training record 
            % plotperform(tr); %shows train, validation, and test per epoch
            
            %Convert X and y test data into Matlab nnet format:
            inputsTest = TestXdata';
            
            testY = net(inputsTest);
            TestDataPred=vec2ind(testY)';
        case 'feedForward'
            if  isfield(options,'nnet_hiddenLayerSize')
                hiddenLayerSize=options.nnet_hiddenLayerSize;
            else
                hiddenLayerSize =10;
            end

            %Convert X and y data into Matlab nnet format:
            inputs = TrainXdata';%%
            
            %Convert to one-hot encoding ground truth values
            targets = zeros(C,size(TrainGT,1));%%
            for ii=1:size(TrainGT,1)
                targets(TrainGT(ii),ii)=1;
            end
            
            % Create a Pattern Recognition Network
            setdemorandstream(2014784333);   %seed for random number generator
            if  isfield(options,'nnet_algo')
                algo=options.nnet_algo;
            else
                algo ='trainscg';
            end
            net = feedforwardnet(hiddenLayerSize,algo);
            
            % Set up Division of Data for Training, Validation, Testing
            net.divideParam.trainRatio = 0.8;  %note- splits are done in a random fashion
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0.0;
            net.trainParam.show = NaN;
            
            % Train the Network
            [net,tr] = train(net,inputs,targets,'UseGPU','yes');%%  %return neural net and a training record 
            % plotperform(tr); %shows train, validation, and test per epoch
            
            %Convert X and y test data into Matlab nnet format:
            inputsTest = TestXdata';
            
            testY = net(inputsTest,'UseGPU','yes');
            TestDataPred=vec2ind(testY)';
        otherwise
            error('Unknown classification method')
    end    
    if (numberOfFolds == 1)
        predictionLabels(:,:) =double(TestDataPred);
    else
        predictionLabels(TestIndex,:) =double(TestDataPred);
    end
end

if (numberOfFolds == 1)
    confusionMatrix = confusionmat(TestGT,predictionLabels);
else
    confusionMatrix = confusionmat(y,predictionLabels);
end
accuracy = sum(diag(confusionMatrix))/sum(sum(confusionMatrix))*100;


%fprintf(sprintf('%s: Accuracy = %6.2f%%%% \n',method,accuracy*100));


if  isfield(options,'confusionMatrix')
    if strcmp(options.confusionMatrix,'True')
        fprintf('Confusion Matrix:\n');
        [r c] = size(confusionMatrix);
        for i=1:r
            for j=1:r
                fprintf('%6d ',confusionMatrix(i,j));
            end
            fprintf('\n');
        end
    end
end



if strcmp(options.method,'Adaboost')
    meanTest = mean(allTest);
    meanTrain = mean(allTrain);
    meanEB  = mean(allEB);
    figure
    hold on
    x = 1:1:adaboost_numFeatures;
    plot(x, meanEB,'k:',x,meanTest,'r-',x,meanTrain,'b--','LineWidth',2);
    legend('ErrorBound','TestErr','TrainErr','Location','Best');
    xlabel 'iteration (number of Classifiers)'
    ylabel 'error rates (50 trials)'
    title 'AdaBoost Performance on Bupa'
    %print -dpng hwk5_11.png
end