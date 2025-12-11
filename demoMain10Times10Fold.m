clearvars;
                                                                      
dataname='Hepatitis';
opt.algorithm='optimize_WU';
load(fullfile('dataset', [dataname, '.mat']));
data=[X, Y];     
[n, d] = size(X);

if strcmp(opt.algorithm , 'optimize_WU')
    base = regexprep(dataname, '\d+$', '');
    k_dataname = ['k_', base];
end

all_indices = cell(10, 1);
all_fea_w = cell(10, 10);
all_para = cell(10, 10);
all_matrics = cell(10, 10); 
all_acc = zeros(10, 10);
all_macro_precision = zeros(10, 10);
all_macro_recall = zeros(10, 10);
all_macro_f1 = zeros(10, 10);

if strcmp(opt.algorithm , 'optimize_WU')
    idx_data =load(fullfile('dataset', ['all_indices_',base, '.mat']));
    if isfield(idx_data, 'all_indices')
        all_indices = idx_data.all_indices;
    else
        error('file indices_%s.mat no all_indices', dataname);
    end
end
for t=1:10
    SelectFeaNum=zeros(d,1);%Count the number of times each feature is selected in the 10-fold cross-validation.
    no_select_num = 0;

    if strcmp(opt.algorithm , 'optimize_WU')
        indices = all_indices{t};
    else
        indices=crossvalind('Kfold',size(data,1),10);
    end
    all_indices{t,1}=indices;
    for k=1:10
        opt.k = k;
        testnum=(indices==k);
        trainnum=~testnum;
        if strcmp(opt.algorithm , 'optimize_WU')
            k_dataname = ['k_', base,'_ind','_',num2str(t),'_',num2str(k)];
            load(fullfile('dataset\nkernel', [k_dataname, '.mat']));
            K_train = k_X;
        end
        X_test=X(testnum==1,:);
        X_train=X(trainnum==1,:); 
        Y_test=Y(testnum==1,:);
        Y_train=Y(trainnum==1,:);

        if strcmp(opt.algorithm , 'optimize_WU')
            opt.r = 5;
            opt.lambda1 = 1000;
            opt.lambda2 = 1;
            opt.lambda3 = 100;
            opt.lrU = 1e-4;
            opt.tol = 1e-5;                                    
            opt.max_iter = 100;
            opt.percent = 0.6;                       
            D.X_train = X_train;
            D.K_train = K_train;
            result = chooseFeatureSelectAlgorithm(D,opt);
            para.opt = opt;                                    
            para.U = result.U;
            para.obj_values = result.obj_values;
            normW = sqrt(sum((result.fea_w).^2, 2));
            [T_Weight, T_sorted_features] = sort(normW, 'descend');
            Num_SelectFeaLY = floor(opt.percent *d);
            SelectFeaIdx = T_sorted_features(1:Num_SelectFeaLY);   
        elseif strcmp(opt.algorithm , 'HQUFS')
            opt.lambda = 0.1;
            opt.maxIter = 200;
            opt.tol     = 1e-5;
            opt.ridge0  = 1e-3;
            opt.percent = 0.6;
            result=chooseFeatureSelectAlgorithm(X_train,opt);
            para.opt = opt;
            normW = sqrt(sum((result.fea_w).^2, 2));
            [T_Weight, T_sorted_features] = sort(normW, 'descend');
            Num_SelectFeaLY = floor(opt.percent *d);
            SelectFeaIdx = T_sorted_features(1:Num_SelectFeaLY);
        elseif strcmp(opt.algorithm , 'EWMC')
            opt.r = 5;
            opt.gamma = 20;
            opt.eta = 0.1;
            opt.epsW = 1e-5;
            opt.maxOuterIters = 10;
            opt.maxInnerItersM = 50;
            opt.baseImputations = {'EM','KNN','SVD'};
            opt.kKNN = 5;
            opt.percent = 0.6;
            opt.verbose = true;
            D.X_train = X_train;
            D.X_test = X_test;
            D.Y_train = Y_train;
            D.featureType = featureType;
            result=chooseFeatureSelectAlgorithm(D,opt);
            para.opt = opt;
            [T_Weight, T_sorted_features] = sort(result.fea_w,'descend');
            Num_SelectFeaLY = floor(opt.percent*d);
            SelectFeaIdx = T_sorted_features(1:Num_SelectFeaLY);      
        else
            D.X_train = X_train;
            D.Y_train = Y_train;
            result=chooseFeatureSelectAlgorithm(D,opt);
            para.opt = opt;
            SelectFeaIdx = find(result.fea_w == 1);
        end

        all_fea_w{t,k} = result.fea_w;
        all_para{t,k} = para;
        if ~isempty(SelectFeaIdx)  
            SelectFeaNum(SelectFeaIdx)=SelectFeaNum(SelectFeaIdx)+1;
            X_trainwF = X_train(:,SelectFeaIdx);
            X_testwF = X_test(:,SelectFeaIdx); 

            Learn = templateSVM('KernelFunction', 'rbf', 'PolynomialOrder', [], 'KernelScale', 'auto', 'BoxConstraint', 1, 'Standardize', true);
            model = fitcecoc(X_trainwF, Y_train, 'Learners', Learn);
            predictedLabels = predict(model, X_testwF);
            metrics = EvaluationMetrics(predictedLabels, Y_test);
            all_matrics{t, k} = metrics;
            all_acc(t,k) = metrics.accuracy;
            all_macro_precision(t,k) = metrics.macro_precision;
            all_macro_recall(t,k) = metrics.macro_recall;
            all_macro_f1(t,k) = metrics.macro_f1;
        else
            all_acc(t,k) = NaN; 
            all_macro_precision(t,k) = NaN;
            all_macro_recall(t,k) = NaN;
            all_macro_f1(t,k)= NaN;
        end
    end
end

total_acc = nanmean(all_acc(:));
total_macro_precision = nanmean(all_macro_precision(:));
total_macro_recall = nanmean(all_macro_recall(:));
total_macro_f1 = nanmean(all_macro_f1(:));
[order_select_num,order_select_id] = sort(SelectFeaNum,'descend');
fea_w_1_1= all_fea_w{1,1};
save(['resultrbf\',char(dataname),'_svm_',char(opt.algorithm),'_best_result_',num2str(total_acc),'_',num2str(total_macro_precision),'_',num2str(total_macro_recall),'_',num2str(total_macro_f1),'.mat'],'all_indices', 'fea_w_1_1', 'all_para', 'all_acc', 'all_macro_precision', 'all_macro_recall', 'all_macro_f1');



