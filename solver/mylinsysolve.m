function [prox_y, M_y, prox_prime, prox_prime_minus] = prox_h(y, sigma)
% Compute the proximal operator of h(z) = (-1/n) * sum(log(z_j))
% along with Moreau envelope and derivative information
% Input:
%   y     - input vector in R^n
%   sigma - positive scalar (quadratic regularization weight)
% Output:
%   prox_y           - proximal point minimizing h(z) + (sigma/2)||z - y||^2
%   M_y              - Moreau envelope value at y
%   prox_prime       - diagonal of Jacobian of prox operator
%   prox_prime_minus - complement: 1 - prox_prime

n = length(y);

% Compute square root term inside the proximal formula
sqrt_term = sqrt(y.^2 + 4 / (sigma * n));

% Compute the proximal mapping elementwise
prox_y = 0.5 * (sqrt_term + y);

% Ensure positivity to avoid log-domain errors
if any(prox_y <= 0)
    fprintf('\nWarning: log undefined for non-positive values. Adjusting.\n');
    prox_y = prox_y + 1e-30;
end

% Compute the Moreau envelope value at y
M_y = (sigma / 2) * norm(prox_y - y)^2 - mean(log(prox_y));

% Compute derivative of proximal operator
tmp = y ./ sqrt_term;
prox_prime = (1 + tmp) / 2;
prox_prime_minus = (1 - tmp) / 2;

end
