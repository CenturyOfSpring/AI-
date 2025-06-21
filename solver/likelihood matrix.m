function [L, rowmax, removeind] = likelihood_matrix(X, U, SIGMA, normalizerows, restrict_dist)
% Constructs the likelihood matrix L(i,j) = N(X(i)|U(j),SIGMA_i)
% Inputs:
%   X             - n x d matrix of observations
%   U             - m x d matrix of grid/support points
%   SIGMA         - either:
%                     (1) d x d matrix (shared covariance)
%                     (2) 1 x d x n (individual diagonal variances per obs)
%   normalizerows - 1: normalize each row by its maximum
%                   0: no normalization
%   restrict_dist - 1: remove rows with all L_ij near zero
% Outputs:
%   L             - n_eff x m likelihood matrix (may remove some rows)
%   rowmax        - maximum value per row before normalization
%   removeind     - indices of rows removed (if restrict_dist is set)

% --- Initialization ---
if nargin < 4, normalizerows = 0; end
if nargin < 5, restrict_dist = 0; end
[n, d] = size(X);
m = size(U,1);
tiny = restrict_dist * 1e-9 + ~restrict_dist * 1e-150;

L = zeros(n, m);
rowmax = zeros(n,1);
removeind = [];
cnt = 0;

% --- Process each observation ---
for i = 1:n
    x_i = X(i, :);
    
    % Get per-observation SIGMA
    if ndims(SIGMA) == 3
        sigma_i = diag(SIGMA(1,:,i));  % diagonal covariance
    elseif ismatrix(SIGMA)
        sigma_i = SIGMA;               % shared covariance
    else
        error('Invalid SIGMA format.');
    end

    % Compute likelihood row
    l_row = mvnpdf(x_i - U, [], sigma_i);
    maxval = max(l_row);
    
    % Keep or discard the row
    if maxval > tiny
        cnt = cnt + 1;
        rowmax(cnt) = maxval;
        if normalizerows
            L(cnt, :) = max(l_row, tiny) / maxval;
        else
            L(cnt, :) = max(l_row, tiny);
        end
    else
        removeind = [removeind; i];
    end
end

% Truncate L and rowmax if some rows removed
L(cnt+1:end,:) = [];
rowmax(cnt+1:end) = [];

end
