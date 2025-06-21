function L = mycholAAt(ATT, m)
% Perform Cholesky factorization of ATT = A*A'
% Input:
%   ATT - symmetric matrix A*A'
%   m   - dimension of ATT
% Output:
%   L structure containing Cholesky factor and solver metadata

% --- Heuristic: choose sparse or dense Cholesky ---
if nnz(ATT) < 0.2 * m^2
    use_sparse = true;
else
    use_sparse = false;
end

% --- Sparse Cholesky via MATLAB's built-in method ---
if use_sparse
    [L.R, L.p, L.perm] = chol(sparse(ATT), 'vector');  % returns upper triangular R
    L.Rt = L.R';  % transpose for backward solve
    L.matfct_options = 'spcholmatlab';
else
    if issparse(ATT)
        ATT = full(ATT);  % convert to dense
    end
    L.perm = 1:m;
    [L.R, indef] = chol(ATT);  % dense Cholesky
    L.matfct_options = 'chol';
end
end
