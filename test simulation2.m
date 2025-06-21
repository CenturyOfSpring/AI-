%% Simulation 2 (d = 2)
% This experiment estimates the prior distribution of θ_i using NPMLE.
% θ_i are sampled from predefined 2D shapes (e.g., circles, triangle, digit 8, letter A),
% and observations X_i ~ N(θ_i, Σ) are generated accordingly.

% --- Initialization ---
clear; close all;
addpath(genpath(pwd));
rng(1);

% --- Settings ---
n = 5000;               % Number of observations
m = 5000;               % Number of grid points
d = 2;                  % Dimension
method_type = 'ALM';    % Solver: 'ALM', 'EM', or 'PEM'
fig_option = 1;         % Shape of θ_i: 1=circle, 2=triangle, 3=digit 8, 4=letter A
sigma_option = 1;       % Covariance: 1=I, 2=random diag
grid_option = 1;        % Grid strategy: 1=data as grid, 2=subsample, 3=uniform mesh

% --- Generate Observations ---
[obs, theta, SIGMA] = generate_observation(n, fig_option, sigma_option, d);

% --- Construct Grid ---
[grid, m_eff] = select_grid(obs, grid_option, m);

% --- Construct Likelihood Matrix ---
[L, ~, removed] = likelihood_matrix(obs, grid, SIGMA, 1);
if ~isempty(removed)
    n = size(L, 1);
end

% --- Solver ---
options.stoptol = 1e-6;
options.printyes = 1;
switch method_type
    case 'ALM'
        options.scaleL = 0;
        options.approxL = 0;
        tic;
        [~, x, ~, ~, ~, info, ~] = DualALM(L, options);
        runtime = toc;
        L = likelihood_matrix(obs, grid, SIGMA, 0);
        llk = mean(log(L * x));
        fprintf('ALM iter = %d, mean log-likelihood = %.8e\n', info.iter, llk);
    case 'EM'
        options.stoptol = 1e-4;
        tic;
        [x, grid, ~, iter] = EM(obs, SIGMA, m, options);
        runtime = toc;
        L = likelihood_matrix(obs, grid, eye(d), 0);
        llk = mean(log(L * x));
        fprintf('EM iter = %d, mean log-likelihood = %.8e\n', iter, llk);
    case 'PEM'
        options.stoptol = 1e-4;
        options.printyes = 0;
        tic;
        [x, grid, ~, iter] = PEM(obs, SIGMA, m, options);
        runtime = toc;
        L = likelihood_matrix(obs, grid, eye(d), 0);
        llk = mean(log(L * x));
        fprintf('PEM iter = %d, mean log-likelihood = %.8e\n', iter, llk);
end

% --- Estimation and Error ---
theta_hat = EB_estimator(L, x, grid);
mse = norm(theta - theta_hat, 'fro')^2 / n;

% --- Plotting ---
plot_raw = 1;
plot_eb = 1;
plot_grid = 1;
x = x / sum(x);  % Normalize mixing weights

set(groot, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

marker_size = 4;
font_size = 20;
x_range = [-9.5, 9.5];
y_range = [-9.5, 9.5];
x_ticks = -8:2:8;
y_ticks = -8:2:8;

% Axis ranges for different figures
switch fig_option
    case 2
        x_range = [-6, 6];
        y_range = [-3, 9];
        x_ticks = -4:2:4;
        y_ticks = -2:2:8;
    case 3
        y_range = [-6.5, 9.5];
    case 4
        % already set above
end

x_text = x_range(1) + 0.03 * diff(x_range);
y_text = y_range(2) - 0.06 * diff(y_range);

% --- Start Plotting ---
figure(1); clf;
total_plots = plot_raw + plot_eb + plot_grid;
use_old = verLessThan('matlab', '9.7');
tile_layout = ~use_old;

if tile_layout
    tiledlayout(1, total_plots, 'Padding', 'none', 'TileSpacing', 'none');
end

% 1. Raw observations and true signals
if plot_raw
    if use_old
        subplot(1, total_plots, 1);
    else
        nexttile;
    end
    plot(theta(:,1), theta(:,2), 'k.', 'MarkerSize', marker_size); hold on;
    plot(obs(:,1), obs(:,2), '.', 'Color', 'b', 'MarkerSize', marker_size);
    legend({'True Signal', 'Raw Data'}, 'FontSize', font_size, 'Location', 'northwest');
    xlim(x_range); ylim(y_range); xticks(x_ticks); yticks(y_ticks);
    axis square; box on; hold off;
end

% 2. Empirical Bayes estimates
if plot_eb
    if use_old
        subplot(1, total_plots, 1 + plot_raw);
    else
        nexttile;
    end
    plot(theta(:,1), theta(:,2), 'k.', 'MarkerSize', marker_size); hold on;
    plot(theta_hat(:,1), theta_hat(:,2), '.', 'Color', 'r', 'MarkerSize', marker_size);
    xlim(x_range); ylim(y_range); xticks(x_ticks); yticks([]);
    axis square; box on; hold off;
    text(x_text, y_text, 'Empirical Bayes', 'FontSize', font_size);
end

% 3. Estimated prior $\widehat{G}_n$
if plot_grid
    if use_old
        subplot(1, total_plots, total_plots);
    else
        nexttile;
    end
    scale_factor = 50;
    mx = max(x);
    x_thresh = strcmp(method_type, 'EM') * 1e-4 + ~strcmp(method_type, 'EM') * 0;
    mx = strcmp(method_type, 'EM') * 1/20 + ~strcmp(method_type, 'EM') * mx;
    for i = 1:m
        if x(i) > x_thresh
            plot(grid(i,1), grid(i,2), 'k.', 'MarkerSize', x(i) * (scale_factor / mx)); hold on;
        end
    end
    plot(theta(:,1), theta(:,2), 'k.', 'MarkerSize', marker_size);
    xlim(x_range); ylim(y_range); xticks(x_ticks); yticks([]);
    axis square; box on; hold off;
    text(x_text, y_text, '$\widehat{G}_n$', 'FontSize', font_size);
end

% --- Adjust Figure Size ---
set(gcf, 'Position', [50 50 500 * total_plots, 500]);
