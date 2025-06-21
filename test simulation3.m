%% Simulation 3: High-dimensional Setting (d = 3, 4, ..., 9)
% This script evaluates NPMLE-based Empirical Bayes estimation in high dimensions.
% Different figure options define the distribution of true signals θ_i.
% Observations are generated as X_i ~ N(θ_i, I_d), and the prior is estimated nonparametrically.

% --- Initialization ---
clear; close all;
addpath(genpath(pwd));
rng(1);

% --- Configuration ---
n = 5000;               % Number of observations
m = n;                  % Grid points (default: use all data points)
d = 9;                  % Data dimensionality
method_type = 'PEM';    % Solver type: 'ALM', 'EM', or 'PEM'
fig_option = 7;         % θ_i structure (see below for definitions)

% --- Observation Generation ---
% fig_option:
% 5: θ_i lies on circle in 1st two dims, rest 0
% 6: θ_i = 0
% 7: θ_i ∈ {0, [6 0...], [0 6...]}
% 8: θ_i ~ N(0,I_d)
% 9: θ_i ∈ six symmetric points in R^d
[obs, theta, SIGMA] = generate_observation(n, fig_option, 1, d);

% --- Grid Construction ---
grid = obs(randperm(n, m), :);  % If m == n, use all obs as grid

% --- Likelihood Matrix L ---
[L, ~, removed] = likelihood_matrix(obs, grid, SIGMA, 1);
if m == n  % Normalize rows and zero diagonals to avoid overfitting
    for i = 1:n
        L(i, i) = 0;
        L(i, :) = L(i, :) / max(L(i, :));
    end
end
if ~isempty(removed)
    n = size(L, 1);
end

% --- Solver Execution ---
options.stoptol = 1e-4;
options.printyes = 0;

switch method_type
    case 'ALM'
        options.scaleL = 0; options.approxL = 0; options.printyes = 1;
        tic;
        [~, x, ~, ~, ~, info, ~] = DualALM(L, options);
        runtime = toc;
        L = likelihood_matrix(obs, grid, eye(d), 0);
        llk = mean(log(L * x));
        fprintf('[ALM] Iter = %d, log-likelihood = %.8e\n', info.iter, llk);
    case 'EM'
        SIGMA = ones([1 d n]);
        tic;
        [x, grid, ~, iter] = EM(obs, SIGMA, m, options);
        runtime = toc;
        L = likelihood_matrix(obs, grid, eye(d), 0);
        llk = mean(log(L * x));
        fprintf('[EM] Iter = %d, log-likelihood = %.8e\n', iter, llk);
    case 'PEM'
        SIGMA = ones([1 d n]);
        tic;
        [x, grid, ~, iter] = PEM(obs, SIGMA, m, options);
        runtime = toc;
        L = likelihood_matrix(obs, grid, eye(d), 0);
        llk = mean(log(L * x));
        fprintf('[PEM] Iter = %d, log-likelihood = %.8e\n', iter, llk);
end

% --- Estimation Results ---
theta_hat = EB_estimator(L, x, grid);               % Posterior mean estimates
mse = norm(theta - theta_hat, 'fro')^2 / n;          % Mean squared error
fprintf('MSE = %.6e\n', mse);

% --- Plotting (project to first two dimensions) ---
plot_flags = [1 1 1];  % [True + Raw, True + EB, Grid Mass]
x = x / sum(x);        % Normalize weights

set(groot, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

fs = 20; ms = 4; scl = 50;
xrange = [-9.5 9.5]; yrange = [-9.5 9.5];
xticks_vals = -8:2:8; yticks_vals = -8:2:8;

if fig_option == 7 || fig_option == 9
    xrange = [-3.5 9.5]; yrange = [-3.5 9.5];
    xticks_vals = -2:2:8;
    yticks_vals = -2:2:8;
end

x_txt = xrange(1) + 0.03 * diff(xrange);
y_txt = yrange(2) - 0.06 * diff(yrange);

% Start plotting
figure(1); clf;
use_old = verLessThan('matlab', '9.7');
n_plot = sum(plot_flags);

if ~use_old
    tiledlayout(1, n_plot, 'Padding', 'none', 'TileSpacing', 'none');
end

% Plot 1: Raw data + True signals
if plot_flags(1)
    if use_old, subplot(1, n_plot, 1); else, nexttile; end
    plot(theta(:,1), theta(:,2), 'k.', 'markersize', ms); hold on;
    plot(obs(:,1), obs(:,2), '.', 'color', 'b', 'markersize', ms);
    k1 = plot(1e3,1e3,'k.','markersize',fs);  % dummy for legend
    k2 = plot(1e3,1e3,'.','color','b','markersize',fs);
    legend([k1 k2], {'True Signal', 'Raw Data'}, 'FontSize', fs, 'Location', 'northwest');
    xlim(xrange); ylim(yrange); xticks(xticks_vals); yticks(yticks_vals);
    axis square; box on; grid on; hold off;
end

% Plot 2: EB estimator
if plot_flags(2)
    if use_old, subplot(1, n_plot, 2); else, nexttile; end
    plot(theta(:,1), theta(:,2), 'k.', 'markersize', ms); hold on;
    plot(theta_hat(:,1), theta_hat(:,2), '.', 'color', 'r', 'markersize', ms);
    xlim(xrange); ylim(yrange); xticks(xticks_vals); yticks([]);
    text(x_txt, y_txt, 'Empirical Bayes', 'FontSize', fs);
    axis square; box on; grid on; hold off;
end

% Plot 3: Estimated prior $\hat{G}_n$
if plot_flags(3)
    if use_old, subplot(1, n_plot, 3); else, nexttile; end
    for i = 1:m
        if x(i) > 0
            plot(grid(i,1), grid(i,2), 'k.', 'markersize', x(i) * scl / max(x));
            hold on;
        end
    end
    plot(theta(:,1), theta(:,2), 'k.', 'markersize', ms);
    xlim(xrange); ylim(yrange); xticks(xticks_vals); yticks([]);
    text(x_txt, y_txt, '$\widehat{G}_n$', 'FontSize', fs);
    axis square; box on; grid on; hold off;
end

% Set window size
set(gcf, 'Position', [100 100 500*n_plot 500]);
