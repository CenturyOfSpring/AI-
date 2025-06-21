function [x, Ax, resnrm, solve_ok] = psqmr(matvecfname, b, par, x0, Ax0)
% Preconditioned symmetric QMR solver (PSQMR)
% Solves Ax = b where A is defined by a matrix-vector function handle
% Inputs:
%   matvecfname - function handle for A*x
%   b           - right-hand side vector
%   par         - struct of parameters (maxit, tol, etc.)
%   x0          - initial guess
%   Ax0         - optional initial Ax0 = A*x0
% Outputs:
%   x       - solution estimate
%   Ax      - A*x
%   resnrm  - residual norms across iterations
%   solve_ok - exit flag (1: success, 2: breakdown, -1: stagnation, -2: maxiter)

N = length(b);
maxit = max(5000, round(sqrt(N)));
tol = 1e-6 * norm(b);
stagnate_check = 20;
miniter = 0;

if nargin < 4 || isempty(x0), x0 = zeros(N,1); end
if isfield(par, 'maxit'), maxit = par.maxit; end
if isfield(par, 'tol'), tol = par.tol; end
if isfield(par, 'stagnate_check_psqmr'), stagnate_check = par.stagnate_check_psqmr; end
if isfield(par, 'minitpsqmr'), miniter = par.minitpsqmr; end

printlevel = 1;
solve_ok = 1;

x = x0;
Aq = nargin >= 5 ? Ax0 : feval(matvecfname, x0);
r = b - Aq;
resnrm = norm(r);
minres = resnrm;

% Use identity preconditioner by default
q = r;
tau_old = norm(q);
rho_old = r' * q;
theta_old = 0;
d = zeros(N, 1);
res = r;
Ad = zeros(N, 1);

tiny = -1e-30;

for iter = 1:maxit
    Aq = feval(matvecfname, q);
    sigma = q' * Aq;
    
    if abs(sigma) < tiny
        solve_ok = 2;
        if printlevel, fprintf('Breakdown: sigma ≈ 0\n'); end
        break;
    end
    
    alpha = rho_old / sigma;
    r = r - alpha * Aq;
    u = r;
    
    theta = norm(u) / tau_old;
    c = 1 / sqrt(1 + theta^2);
    tau = tau_old * theta * c;
    
    gam = (c^2) * (theta_old^2);
    eta = (c^2) * alpha;
    
    d = gam * d + eta * q;
    x = x + d;
    
    % Residual update and convergence check
    Ad = gam * Ad + eta * Aq;
    res = res - Ad;
    err = norm(res);
    resnrm(iter+1) = err;
    minres = min(minres, err);
    
    if (err < tol) && (iter > miniter) && (b' * x > 0)
        break;
    end
    
    if (iter > stagnate_check) && (iter > 10)
        ratio = resnrm(iter-9:iter+1) ./ resnrm(iter-10:iter);
        if all(ratio > 0.997 & ratio < 1.003)
            if printlevel, fprintf('Stagnation detected\n'); end
            solve_ok = -1;
            break;
        end
    end
    
    rho = r' * u;
    if abs(rho_old) < tiny
        solve_ok = 2;
        if printlevel, fprintf('Breakdown: rho_old ≈ 0\n'); end
        break;
    end
    
    beta = rho / rho_old;
    q = u + beta * q;
    
    rho_old = rho;
    tau_old = tau;
    theta_old = theta;
end

if iter == maxit
    solve_ok = -2;
end

Ax = b - res;

end
