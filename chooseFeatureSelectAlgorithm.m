function  [result] = chooseFeatureSelectAlgorithm(data,opt)

currentPath = pwd;          
%% start classification
switch opt.algorithm
    case 'optimize_WU'
        addpath([currentPath,'\model\FS']);
        [result.fea_w, result.U, result.obj_values] = optimize_WU(data.X_train, data.K_train, opt.r, opt.lambda1, opt.lambda2, opt.lambda3, opt.lrU, opt.max_iter, opt.tol);
        result.K = result.U * (result.U');
        rmpath([currentPath,'\model\FS']);   
    case 'EQI-BGWO'
        addpath([currentPath,'\model\EQI-BGWO']);
        X = data.X_train;
        X(isnan(X)) = 0; 
        [result.fitness, result.fea_w] = main(X, data.Y_train);
        rmpath([currentPath,'\model\EQI-BGWO']);
    case 'QSIFS'
        addpath([currentPath,'\model\QSIFS']);
        X = data.X_train;
        X(isnan(X)) = 0; 
        [result.fea_w] = main(X, data.Y_train);
        rmpath([currentPath,'\model\QSIFS']);
    case 'HQUFS'
        addpath([currentPath,'\model\HQUFS']);   
        [result.fea_w, result.v] = hq_ufs_incomplete(data, opt);     
        rmpath([currentPath,'\model\HQUFS']);
    case 'EWMC'
        addpath([currentPath,'\model\EWMC']);  
        [result.Z,result.fea_w,result.history,result.G,result.H] = ewmc_mu_reliefA_strict(data.X_train, data.Y_train, data.featureType, opt);    
        rmpath([currentPath,'\model\EWMC']);
     
end

