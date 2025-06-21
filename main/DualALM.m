function [obj, x, y, u, v, info, runhist] = DualALM(L, options)
% ALM-based solver for dual of NPMLE mixture proportion estimation
% L: likelihood matrix of size n × m
% options: structure containing fields like stoptol, sigma, etc.

% -------- Default Parameters --------
params = struct('stoptol', 1e-6, 'printyes', 1, 'maxiter', 100, ...
                'sigma', 100, 'scaleL', 1, 'approxL', 0, ...
                'approxRank', 30, 'init_opt', 0);
if exist('options','var')
    option_fields = fieldnames(params);
    for f = 1:numel(option_fields)
        if isfield(options, option_fields{f})
            params.(option_fields{f}) = options.(option_fields{f});
        end
    end
end

% -------- Preprocessing --------
[n, m] = size(L);
s = ones(n,1);
if params.scaleL
    s = 1 ./ max(L, [], 2);
    L = s .* L;
end

% -------- Low-Rank Approximation --------
if params.approxL
    [U, S, V] = svds(L, ceil(params.approxRank), 'largest');
    if S(end,end) <= min(10*params.stoptol, 1e-4)
        rank_eff = find(diag(S) < 1e-4, 1);
        if ~isempty(rank_eff)
            U = U(:, 1:rank_eff) * S(1:rank_eff, 1:rank_eff);
            V = V(:, 1:rank_eff);
        end
        LL.times = @(x) U * (V' * x);
        LL.trans = @(y) V * (U' * y);
    else
        LL.times = @(x) L * x;
        LL.trans = @(y) (y' * L)';
    end
else
    LL.times = @(x) L * x;
    LL.trans = @(y) (y' * L)';
end

% -------- Initialization --------
if params.init_opt == 0
    x = ones(m,1) / m;
    y = sum(L,2) / m;
    u = 1 ./ y;
    v = u;
else
    x = 0.5 * params.sigma * ones(m,1);
    y = sum(L,2) / m;
    u = 1 ./ y;
    v = zeros(n,1);
end

% -------- Main Algorithm --------
[~, x, y, u, v, info, runhist] = DualALM_main(LL, params, x, y, u, v);

% -------- Post-processing --------
Lx = L * x;
obj(1) = sum(x) + sum(log(s) - log(Lx)) / n - 1;
obj(2) = sum(log(v)) / n;
y = y ./ s; u = u .* s; v = v .* s;
end
