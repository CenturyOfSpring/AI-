function [x, supps, hist, k] = EM(X, SIGMA, m, options)
% EM: Expectation-Maximization algorithm for NPMLE with unknown support
% Inputs:
%   X ∈ ℝ^{n×d}       - Observations
%   SIGMA ∈ ℝ^{n×d}   - Diagonal covariances
%   m                - Number of grid points
%   options          - Struct (optional) with fields:
%                      'supps_initial', 'stoptol'
% Outputs:
%   x     - Estimated weights (m×1)
%   supps - Estimated support points (m×d)
%   hist  - Struct with log-likelihood history
%   k     - Iteration count

fprintf('\n----------------- EM algorithm --------------------\n');
maxiter = 100;
stoptol = 1e-4;
[n, d] = size(X);
x = ones(m, 1) / m;

% Initialize support points
if m < n
    supps = X(randperm(n, m), :);
else
    supps = X;
end
if nargin > 3
    if isfield(options, 'supps_initial'), supps = options.supps_initial; end
    if isfield(options, 'stoptol'), stoptol = options.stoptol; end
end

Sigma = reshape(SIGMA, d, n)';
inv_Sigma = 1 ./ Sigma;
inv_SigmaX = inv_Sigma .* X;

for k = 1:maxiter
    L = likelihood_matrix(X, supps, SIGMA);
    Lx = L * x;
    gamma_hat = (L .* x') ./ (Lx * ones(1, m));
    supps = (gamma_hat' * inv_SigmaX) ./ (gamma_hat' * inv_Sigma);
    x = sum(gamma_hat)' / n;
    obj = mean(log(Lx));
    hist.obj(k) = obj;
    fprintf('iter = %3d, log-likelihood = %5.8e\n', k, obj);
    if k > 1 && (obj - hist.obj(k - 1)) < stoptol
        break;
    end
end
end
