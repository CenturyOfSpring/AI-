function [X, theta, SIGMA] = generate_observation(n, fig_option, sigma_option, d)
% Generate synthetic observations and latent variables for NPMLE testing
% Input:
%   n            - number of samples
%   fig_option   - integer for specifying θ_i pattern (e.g., circle, triangle, etc.)
%   sigma_option - 1: identity covariance; 2: heteroscedastic diagonal
%   d            - dimension (default = 2)
% Output:
%   X      - observed data (n-by-d)
%   theta  - true signal θ_i (n-by-d)
%   SIGMA  - 3D array of covariance values if heteroscedastic (1*d*n)

if nargin < 4 || d < 2
    d = 2;
end

% Generate Sigma
if nargin < 3 || sigma_option == 1
    SIGMA = eye(d);
else
    SIGMA = rand([1 d n]) * 2 + 1;  % Diagonal entries ∈ [1,3]
end

% Generate theta based on fig_option
switch fig_option
    case 1  % Two concentric circles
        theta = [2 * [cos(2*pi*rand(n/2,1)), sin(2*pi*rand(n/2,1))];
                 6 * [cos(2*pi*rand(n/2,1)), sin(2*pi*rand(n/2,1))]];
    case 2  % Triangle path
        p = [-3 0; 0 6; 3 0];
        segments = floor(n/3);
        theta = [p(1,:) + (p(2,:) - p(1,:)) .* rand(segments,1);
                 p(2,:) + (p(3,:) - p(2,:)) .* rand(segments,1);
                 p(3,:) + (p(1,:) - p(3,:)) .* rand(n - 2*segments,1)];
    case 3  % Digit "8" with two circles
        theta = zeros(n,d);
        t1 = 2 * pi * rand(floor(n/2),1);
        t2 = 2 * pi * rand(n - floor(n/2),1);
        theta(1:floor(n/2),:) = 3 * [cos(t1), sin(t1)];
        theta(floor(n/2)+1:end,:) = [0 6] + 3 * [cos(t2), sin(t2)];
    case 4  % Letter "A"
        points = [-4 -6; -2 0; 0 6; 2 0; 4 -6];
        theta = [];
        for i = 1:4
            theta = [theta; points(i,:) + (points(i+1,:) - points(i,:)) .* rand(floor(n/5),1)];
        end
        theta = [theta; points(4,:) + (points(2,:) - points(4,:)) .* rand(n - 4*floor(n/5),1)];
    case 5  % Single circle
        theta = 6 * [cos(2*pi*rand(n,1)), sin(2*pi*rand(n,1))];
    case 6  % Zero signal
        theta = zeros(n,d);
    case 7  % Discrete atoms at 0, (6,0), (0,6)
        atoms = [0 0; 6 0; 0 6];
        theta = repmat(atoms, ceil(n/3), 1);
        theta = theta(1:n,:);
    case 8  % θ_i ∼ N(0,Σ_i)
        theta = mvnrnd(zeros(1,d), SIGMA, n);
    case 9  % 6 atom mixture
        atoms = [0 0; 6 0; -6 0; 0 6; 6 6; -6 6];
        theta = repmat(atoms, ceil(n/6), 1);
        theta = theta(1:n,:);
    otherwise
        error('Unknown fig_option');
end

% Generate observations with noise
X = mvnrnd(theta, SIGMA);

end
