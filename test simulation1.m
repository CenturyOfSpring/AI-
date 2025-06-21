%% Simulation Example 1 (Dimension d = 1)
% This experiment replicates the setup used in:
% - Johnston and Silverman (2004)
% - Brown and Greenshtein (2009)
% - Jiang and Zhang (2009)
% The goal is to recover the underlying sparse signal distribution using NPMLE.

% Add path and set random seed
addpath(genpath(pwd));
rng(1);
clear;

% Experiment parameters
n = 1000;          % Total number of observations
k = 500;           % Number of non-zero signals
mu = 7;            % Mean of non-zero signals
m = 500;           % Number of grid points

% Generate observations: Xi ~ N(θi,1), where θi = mu for i ≤ k, otherwise 0
Y = zeros(n,1);
Y(1:k) = randn(k,1) + mu;
Y(k+1:n) = randn(n - k,1);

% Construct grid points uniformly in the data range
grid_upper = max(Y) + eps;
grid_lower = min(Y) - eps;
U = linspace(grid_lower, grid_upper, m)';
G = Y * ones(1, m) - ones(n,1) * U';
L = normpdf(G);  % Likelihood matrix: L_ij = N(Y_i | U_j, 1)

% Solver options
options.maxiter = 100;
options.stoptol = 1e-6;
options.stopop = 3;
options.printyes = 1;
options.approxL = 1;

% Run DualALM optimization
[obj, x, y, u, v, info, runhist] = DualALM(L, options);

% Plot settings
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
line_width = 2;
marker_size = 4;
font_size = 15;

% 1. True density vs estimated mixture density
figure(1);
is_old_version = verLessThan('matlab', '9.7');
if is_old_version
    subplot(1, 3, 1);
else
    layout = tiledlayout(1, 3, 'Padding', 'none', 'TileSpacing', 'none');
    nexttile;
end
x_vals = (-max(U)-1):0.01:(max(U)+1);
true_density = (k/n)*normpdf(x_vals, mu, 1) + (1 - k/n)*normpdf(x_vals, 0, 1);
plot(x_vals, true_density, 'LineWidth', line_width); hold on;
[Y_sorted, idx] = sort(Y);
plot(Y_sorted, y(idx), ':', 'LineWidth', line_width);
legend('True density $f_{G^*,1}$', 'Estimated $\widehat{f}_{\widehat{G}_n,1}$', ...
       'Location', 'northwest', 'FontSize', font_size);
ylim padded; axis square; box on;

% 2. Estimated prior measure
if is_old_version
    subplot(1, 3, 2);
else
    nexttile;
end
plot(U, x, 'LineWidth', line_width);
legend('NPMLE of prior $\widehat{G}_n$', 'Location', 'northwest', 'FontSize', font_size);
ylim padded; axis square; box on;

% 3. Empirical Bayes estimator via Tweedie’s formula
if is_old_version
    subplot(1, 3, 3);
else
    nexttile;
end
theta_hat = (L * (U .* x)) ./ (L * x);  % Posterior mean estimates
plot(Y_sorted, theta_hat(idx), '-+', 'LineWidth', line_width, 'MarkerSize', marker_size);
legend('Bayes estimator $\widehat{\theta}_i$', 'Location', 'northwest', 'FontSize', font_size);
ylim padded; axis square; box on;

% Adjust figure size
set(gcf, 'Position', [50 50 1800 600]);
