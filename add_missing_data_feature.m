% data: Original Dataset (n ¡Á d)
% sample_ratio: Missing sample proportion, e.g., 0.05 indicates 5%
% feature_ratio: Proportion of missing features per missing sample, e.g., 0.2 indicates 20% of features missing
dataname = 'warpAR10P';     % colon warpAR10P Yale
sample_ratio = [0.1, 0.15, 0.2];     % Fixed missing sample proportion
feature_ratio_list = [0.1, 0.15, 0.2];   % Different missing feature proportions to generate

for fr = feature_ratio_list
    load(fullfile('dataset', [dataname, '.mat']));
    [n, d] = size(X);

    X_missing = X;

    % Precisely select columns
    num_cols_missing = max(1, round(d * fr));
    cols_missing = randperm(d, num_cols_missing);

    % Precisely select rows
    num_rows_missing = max(1, round(n * fr));
    rows_missing = randperm(n, num_rows_missing);

    % Precise missing matrix
    X_missing(rows_missing, cols_missing) = NaN;

    X = X_missing;
    try
        save(fullfile('dataset', ...
             [dataname, num2str(fr*100), '.mat']), ...
             'X', 'Y', 'featureType');
    catch ME
        disp(ME.message)
    end

    fprintf('Saved dataset with feature_ratio = %.2f\n', fr);

end