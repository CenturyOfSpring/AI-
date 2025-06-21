function [x, y, u, v, Lx, LTv, par, runhist, info] = MLE_SSNCG(LL, x, y, v, LTv, par, options)
% SSNCG Solver for the dual of NPMLE with augmented Lagrangian
% Inputs: LL (operator), initial x/y/v, parameters and options
% Outputs: solution (x, y, u, v), diagnostics and convergence info

printyes = options.printyes;
tol = options.tol;
sigma = par.sigma;
n = par.n;

% Initialize proximal quantities
v1input = v - (n/sigma) * y;
[prox_v1, M_v1, ~, prox_v1_prime_m] = prox_h(v1input, sigma/(n^2));
v2input = LTv / n + x / sigma - 1;
prox_v2 = max(v2input, 0);
Lprox_v2 = LL.times(prox_v2);

phi = -(M_v1 + (sigma / 2) * norm(prox_v2)^2);
runhist = struct(); breakyes = 0;

for itersub = 1:options.maxitersub
    % Compute gradient
    grad = (sigma / n^2) * (v1input - prox_v1) + (sigma / n) * Lprox_v2;
    normGrad = norm(grad);

    % Check optimality
    priminf = normGrad / norm((sigma/n)*(v1input - prox_v1));
    dualinf = max(norm(LTv - n, inf), norm(prox_v1 - v)) / max(norm(prox_v1), 1e-8);
    runhist.priminf(itersub) = priminf;
    runhist.dualinf(itersub) = dualinf;
    runhist.phi(itersub) = phi;

    if printyes
        fprintf('\n  [%02d] phi = %.2e | primal = %.2e | dual = %.2e', ...
                itersub, phi, priminf, dualinf);
    end

    % Termination condition
    if priminf < tol && itersub > 1
        u = prox_v1;
        x = sigma * prox_v2;
        Lx = sigma * Lprox_v2;
        y = (sigma/n) * (prox_v1 - v1input);
        break;
    end

    % Newton direction (PCG or direct)
    rhs = -grad;
    par.tol = min(1e-2, 1e-1 * priminf);
    [dv, resnrm, solve_ok, par] = Linsolver_MLE(rhs, LL, prox_v1_prime_m, v2input, par);
    iterpsqmr = length(resnrm) - 1;

    if printyes
        fprintf(' | CG iters: %d | res: %.2e', iterpsqmr, resnrm(end));
    end

    % Line search
    LTdv = LL.trans(dv);
    [phi, v1input, prox_v1, prox_v1_prime_m, ...
     v2input, prox_v2, Lprox_v2, alp, iterstep, par] = ...
     findstep(grad, dv, LTdv, LL, phi, v1input, prox_v1, ...
              prox_v1_prime_m, v2input, prox_v2, Lprox_v2, 1e-4, 2, par);

    % Update v and dual objective
    v = v + alp * dv;
    LTv = LTv + alp * LTdv;
    runhist.solve_ok(itersub) = solve_ok;
    runhist.psqmr(itersub) = iterpsqmr;
    runhist.findstep(itersub) = iterstep;

    if alp < 1e-10
        breakyes = 11; break;
    end
end

if itersub == options.maxitersub
    u = prox_v1;
    x = sigma * prox_v2;
    Lx = sigma * Lprox_v2;
    y = (sigma/n) * (prox_v1 - v1input);
end

info.tolCG = sum(runhist.psqmr);
info.breakyes = breakyes;
info.itersub = itersub;
end
