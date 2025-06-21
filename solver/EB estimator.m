function theta_hat = EB_estimator(L, x, U)
% EB_estimator: Generalized Maximum Likelihood Empirical Bayes estimator
% Inputs:
%   L ∈ ℝ^{n×m} - Likelihood matrix
%   x ∈ ℝ^m     - Weights (x ≥ 0, sum(x) = 1)
%   U ∈ ℝ^{m×d} - Grid/support points
% Output:
%   theta_hat ∈ ℝ^{n×d} - Estimated parameters

Lx = L * x;                     % Compute marginal likelihood
[n, ~] = size(L);
d = size(U, 2);
theta_hat = zeros(n, d);

for i = 1:n
    theta_hat(i, :) = (x' .* L(i, :)')' * U / Lx(i);
end
end
