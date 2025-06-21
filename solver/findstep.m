function [phi, v1input, prox_v1, prox_v1_prime_m, ...
          v2input, prox_v2, Lprox_v2, alp, iter, par] = ...
    findstep(Grad, dv, LTdv, LL, phi0, ...
             v1input0, prox_v10, prox_v1_prime_m0, ...
             v2input0, prox_v20, Lprox_v20, tol, options, par)
% findstep: Perform line search to ensure descent on augmented Lagrangian
% Inputs: current gradient, direction, previous proximal terms
% Outputs: updated quantities and step size

sigma = par.sigma;
n = par.n;
maxit = ceil(log(1 / (tol + eps)) / log(2));
c1 = 1e-4;
c2 = 0.9;
g0 = -Grad' * dv;

if g0 <= 0
    if par.printyes
        fprintf('\n Warning: ascent direction encountered, g0 = %.2e\n', g0);
    end
    % Return original values unchanged
    [phi, v1input, prox_v1, prox_v1_prime_m, ...
     v2input, prox_v2, Lprox_v2, alp, iter] = ...
        deal(phi0, v1input0, prox_v10, prox_v1_prime_m0, ...
             v2input0, prox_v20, Lprox_v20, 0, 0);
    return;
end

alp = 1; alpconst = 0.5;
LB = 0; UB = 1;

for iter = 1:maxit
    if iter > 1
        alp = alpconst * (LB + UB);
    end

    % Update proximal values
    v1input = v1input0 + alp * dv;
    v2input = v2input0 + (alp / n) * LTdv;
    [prox_v1, M_v1, ~, prox_v1_prime_m] = prox_h(v1input, sigma / n^2);
    prox_v2 = max(v2input, 0);

    % Augmented Lagrangian (dual)
    phi = -(M_v1 + (sigma / 2) * norm(prox_v2)^2);
    tmp = (sigma / n^2) * (v1input - prox_v1);
    galp = -tmp' * dv - (sigma / n) * (prox_v2' * LTdv);

    % Sufficient decrease and curvature check
    if abs(galp) < c2 * abs(g0) && (phi - phi0 - c1 * alp * g0 > eps)
        if (options == 1) || (options == 2 && abs(galp) < tol)
            if par.printyes
                fprintf(':');
            end
            break;
        end
    end

    % Interval shrinkage
    if sign(galp) ~= sign(g0)
        UB = alp;
    else
        LB = alp;
    end
end

Lprox_v2 = LL.times(prox_v2);
par.count_L = par.count_L + 1;

if par.printyes && iter == maxit
    fprintf(' (line search max iters reached) ');
end
end
