function [W, U, obj_values] = optimize_WU(X, K, r, lambda1, lambda2, lambda3, lrU, max_iter, tol)
% Alternating optimization for W and U
%   Minimize over W,U:
%     tr[ U U' - 2 U U' W + W' U U' W ] + lambda1 || M .* (K - U U') ||_F^2
%     + lambda2 * ||W||_{2,1} + lambda3 * ||U||_F^2
%   s.t. 0 <= (U U')_ij <= 1, diag(UU') = I
%
% INPUT:
%   K         - d x d symmetric kernel matrix
%   M         - d x d mask matrix (same size as K)
%   lambda1   - weight for mask term
%   lambda2   - group-lasso weight for W (||W||_{2,1})
%   lambda3   - Frobenius weight for U
%   r         - embedding dimension (number of cols of U)
%   max_iter: Maximum number of iterations
%   tol: Convergence threshold

    [n, d] = size(X);
    nan_sample_idx = find(any(isnan(X), 1));
    M = ones(d, d);
    M(nan_sample_idx, :)=0;
    M(:, nan_sample_idx)=0;    
    
    % 1. Top-r eigenvectors initialization
    [U_eig,S_eig] = eigs(K, r, 'LM');  % K is symmetric real matrix
    S_eig = diag(S_eig);
    % 2. Construct initial U
    U = U_eig * diag(sqrt(S_eig));   % Multiply each column by corresponding sqrt(eigenvalue)
    % 3. Row normalization to make diag(UU') = 1
    row_norms = sqrt(sum(U.^2, 2));
    U = bsxfun(@rdivide, U, max(row_norms,1e-12));
    
    proj_type = 'strict';  %simplified strict 
    
    % init W (simple)
    W = eye(d);

    obj_values = zeros(max_iter, 1);
   
    for iter = 1:max_iter
        % Step 1: Update W
        rownormsW = sqrt(sum(W.^2, 2));
        D = diag(1 ./ (2*rownormsW + 1e-8));  % IRLS weight diag
        % Solve linear system: (K + lambda2 * D) * W = K
        % -> W = (UU^T + lambda2 * D) \ UU^T
        A = U*U' + lambda2 * D + 1e-12 * eye(d);
        % ensure symmetry / positive-definiteness numerically
        %A = (A + A')/2 + 1e-12 * eye(d);
        % Solve for W (d x d right-hand side)
        W = A \ (U*U');
      
        % Step 2: Update U
        % Precompute A = I - (W + W') + W*W'
        A_mat = eye(d) - (W + W') + (W * W');
        R = M .* ( U * U' - K);
        
        grad_U = 2 * (A_mat * U) + 4 * lambda1 * (R * U) + 2 * lambda3 * U;

        %lrU = 1e-3;  % Learning rate
        U = U - lrU * grad_U;
        
        % ---------------- 3. Projection ----------------
        switch proj_type
            case 'simplified'
                % Simplified engineering version: Non-negative + row normalization
                U = max(U, 0);
                row_norms = sqrt(sum(U.^2,2));
                U = bsxfun(@rdivide, U, max(row_norms,1e-12));

                % Optional: Ensure PSD
                G = U*U';
                if any(eig(G) < 0)
                    [Q,S] = eigs(G, r, 'LA');  % Take top r eigenvectors
                    S = diag(real(S));
                    S(S<0) = 0;
                    U = Q * diag(sqrt(S));
                    row_norms = sqrt(sum(U.^2,2));
                    U = bsxfun(@rdivide, U, max(row_norms,1e-12));
                end

            case 'strict'
                % Strict mathematical version
                U = project_U_strict(U, r, 1);

            otherwise
                error('Unknown projection type');
        end
    
        % Step 3: Calculate objective function value
        UUt = U*U';
        term1 = trace(UUt - 2*UUt*W + W'*(UUt)*W);
        term2 = lambda1 * norm(M .* (K - UUt), 'fro')^2;
        term3 = lambda2 * sum(sqrt(sum(W.^2,2)));
        term4 = lambda3 * norm(U, 'fro')^2;

        obj = term1 + term2 + term3 + term4;
        obj_values(iter) = obj;

        % Convergence check
        if iter > 1 && abs(obj_values(iter) - obj_values(iter-1))/obj_values(iter-1) < tol
            obj_values = obj_values(1:iter);   
            fprintf('Converged at iteration %d\n', iter);
            break;
        end
    end
 end

%Dykstra's alternating projection algorithm
%Alternate the following 3 steps
%Project onto box constraints (0 ≤ G_ij ≤ 1)
%Project onto PSD constraints (G ? 0)
%Project onto diag(G)=1
function U = project_U_strict(U, r, max_iter)
% Strict mathematical projection
% Input:
%   U        - d x r current U
%   r        - embedding dimension
%   max_iter - internal iteration count (Dykstra style)
% Output:
%   U        - projected U, satisfying G PSD, diag(G)=1, 0<=G_ij<=1

if nargin < 3
    max_iter = 50;
end

d = size(U,1);

for k = 1:max_iter
    % 1. Calculate Gram matrix
    G = U * U';
    
    % 2. Element-wise truncation to 0~1
    G(G<0) = 0;
    G(G>1) = 1;
    
    % 3. PSD projection
    [V,D] = eig(G);
    D = diag(D);
    D(D<0) = 0;
    G = V * diag(D) * V';
    
    % 4. Keep diagonal = 1
    diag_scale = diag(1 ./ sqrt(diag(G) + 1e-12));
    G = diag_scale * G * diag_scale;
    
    % 5. Low-rank decomposition from Gram back to U
    [Q,S] = eigs(G, r, 'LM');   % Take top-r
    S = diag(real(S));
    S(S<0) = 0;
    U = Q * diag(sqrt(S));
end

% 6. Final row normalization
row_norms = sqrt(sum(U.^2, 2));
U = bsxfun(@rdivide, U, max(row_norms, 1e-12));
end
