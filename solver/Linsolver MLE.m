function [dv, resnrm, solve_ok, par] = Linsolver_MLE(rhs, LL, prox_v1_prime_m, v2, par)
n = par.n;
m = par.m;
sigma = par.sigma;
J = (v2 > 0);
r = sum(J);
par.r = r;

% Solver selection
if n <= 5000
    solveby = 'pdirect';
elseif r < 2000 || (n > 5000 && r < 5000)
    solveby = 'ddirect';
else
    solveby = 'pcg';
end

rhs = rhs * (n^2 / sigma);
prox_v1_prime_m = prox_v1_prime_m + eps;

switch solveby
    case 'pdirect'
        if par.approxL
            U = LL.U; V = LL.V; VJ = V(J,:);
            LLT = U * (VJ' * VJ) * U';
        else
            LJ = LL.matrix(:, J);
            LLT = LJ * LJ';
        end
        LLT(1:n+1:end) = LLT(1:n+1:end) + prox_v1_prime_m;
        cholLLT = mycholAAt(LLT, n);
        dv = mylinsysolve(cholLLT, rhs);
        resnrm = 0; solve_ok = 1;

    case 'pcg'
        if par.approxL
            U = LL.U; V = LL.V; VJ = V(J,:);
            Afun = @(v) (prox_v1_prime_m .* v + U * (VJ * (v' * U)')' * VJ') * (sigma / n^2);
        else
            LJ = LL.matrix(:, J);
            Afun = @(v) (prox_v1_prime_m .* v + LJ * (LJ' * v)) * (sigma / n^2);
        end
        [dv, ~, resnrm, solve_ok] = psqmr(Afun, rhs, par);

    case 'ddirect'
        if par.approxL
            U = LL.U; V = LL.V; VJ = V(J,:);
            rhstmp = VJ * ((rhs ./ prox_v1_prime_m)' * U)';
            LTL = VJ * (U' * (U ./ prox_v1_prime_m)) * VJ' + eye(r);
            cholLTL = mycholAAt(LTL, r);
            dv = mylinsysolve(cholLTL, rhstmp);
            dv = rhs ./ prox_v1_prime_m - (U * (dv' * VJ)') ./ prox_v1_prime_m;
        else
            LJ = LL.matrix(:, J);
            LJ2 = LJ ./ prox_v1_prime_m;
            rhstmp = LJ2' * rhs;
            LTL = eye(r) + LJ2' * LJ;
            cholLTL = mycholAAt(LTL, r);
            dv = rhs ./ prox_v1_prime_m - LJ2 * (mylinsysolve(cholLTL, rhstmp));
        end
        resnrm = 0; solve_ok = 1;
end
end
