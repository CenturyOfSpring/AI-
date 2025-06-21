function [X, theta, SIGMA] = generate_observation(n, fig_option, sigma_option, d)
% Generate synthetic observations X in R^d based on predefined geometric structures
% Inputs:
%   n           - Number of observations
%   fig_option  - Structure type (1~9), determines theta distribution
%   sigma_option - 1: Identity covariance; 2: diag(ai,bi), ai,bi ~ U[1,3]
%   d           - Dimension (default = 2)
% Outputs:
%   X       - n-by-d observation matrix
%   theta   - n-by-d true signal matrix
%   SIGMA   - Covariance structure

if nargin < 4 || d < 2
    d = 2;
end

if nargin < 3 || sigma_option == 1
    SIGMA = eye(d);
else
    SIGMA = rand([1 d n]) * 2 + 1; % Uniform(ai,bi) ∈ [1,3]
end

switch fig_option
    case 1  % Two concentric circles
        r1 = 2; r2 = 6;
        n1 = round(n / 2); n2 = n - n1;
        theta = zeros(n, d);
        theta(1:n1, :) = r1 * [cos(2*pi*rand(n1,1)), sin(2*pi*rand(n1,1))];
        theta(n1+1:n, :) = r2 * [cos(2*pi*rand(n2,1)), sin(2*pi*rand(n2,1))];
    
    case 2  % Triangle path
        p1 = [-3 0]; p2 = [0 6]; p3 = [3 0];
        n1 = floor(n/3); n2 = n1; n3 = n - n1 - n2;
        theta = [p1 + rand(n1,1).*(p2-p1);
                 p2 + rand(n2,1).*(p3-p2);
                 p3 + rand(n3,1).*(p1-p3)];

    case 3  % Figure 8 (two circles)
        r = 3; c1 = [0 0]; c2 = [0 6];
        n1 = round(n/2); n2 = n - n1;
        theta = zeros(n, d);
        theta(1:n1,:) = c1 + r * [cos(2*pi*rand(n1,1)), sin(2*pi*rand(n1,1))];
        theta(n1+1:n,:) = c2 + r * [cos(2*pi*rand(n2,1)), sin(2*pi*rand(n2,1))];

    case 4  % Letter A path (broken into 5 segments)
        p = [-4 -6; -2 0; 0 6; 2 0; 4 -6];
        n_seg = floor(n/5); remainder = n - 4*n_seg;
        theta = [p(1,:) + rand(n_seg,1).*(p(2,:) - p(1,:));
                 p(2,:) + rand(n_seg,1).*(p(3,:) - p(2,:));
                 p(3,:) + rand(n_seg,1).*(p(4,:) - p(3,:));
                 p(4,:) + rand(n_seg,1).*(p(5,:) - p(4,:));
                 p(4,:) + rand(remainder,1).*(p(2,:) - p(4,:))];

    case 5  % Single circle
        r = 6;
        t = 2 * pi * rand(n,1);
        theta = r * [cos(t), sin(t)];

    case 6  % All zeros
        theta = zeros(n, d);

    case 7  % Atoms: 0, 6e1, 6e2
        atoms = [0 0; 6 0; 0 6];
        xstar = ones(3,1)/3;
        theta = [];
        for i = 1:3
            count = round(n * xstar(i));
            theta = [theta; repmat(atoms(i,:), count, 1)];
        end
        theta = theta(1:n,:);  % adjust if over-rounded

    case 8  % Gaussian prior for theta
        theta = mvnrnd(zeros(d,1), SIGMA, n);

    case 9  % 6-atom uniform discrete prior (e.g. star layout)
        r = 6;
        atoms = [0 0; r 0; -r 0; 0 r; r r; -r r];
        xstar = ones(6,1)/6;
        theta = [];
        for i = 1:6
            count = round(n * xstar(i));
            theta = [theta; repmat(atoms(i,:), count, 1)];
        end
        theta = theta(1:n,:);
end

% Final observations
if size(SIGMA,3) == 1
    X = mvnrnd(theta, SIGMA);
else
    X = zeros(n, d);
    for i = 1:n
        cov_i = diag(SIGMA(1,:,i));
        X(i,:) = mvnrnd(theta(i,:), cov_i);
    end
end
end
