function [obj, x, y, u, v, info, runhist] = DualALM_main(LL, params, x, y, u, v)
% Main ALM loop for dual optimization

% -------- Initialization --------
sigma = params.sigma; n = params.n; m = params.m;
maxiter = params.maxiter; tstart = params.tstart;
stopop = 1; sigmamax = 1e7; sigmamin = 1e-8;
stop = 0; termination = '';

% -------- Initial Feasibility --------
Lx = LL.times(x); obj(1) = sum(x) - sum(log(Lx))/n - 1;
obj(2) = sum(log(u))/n;
relgap = abs(diff(obj)) / (1 + sum(abs(obj)));
Rp = Lx - y;
primfeas = max(norm(Rp)/norm(y), norm(min(x,0))/norm(x));
LTv = LL.trans(v);
dualfeas = max(norm(max(LTv - n,0))/n, norm(u - v)/norm(u));
eta = norm(y - 1./v)/norm(y);
runhist = struct();

% -------- Main Loop --------
for iter = 1:maxiter
    % Call semi-smooth Newton-CG step
    ssncgop = struct('tol', params.stoptol, 'printyes', params.printyes, ...
                     'maxitersub', 20);
    [x, y, u, v, Lx, LTv, ~, ~, infoNCG] = ...
        MLE_SSNCG(LL, x, y, v, LTv, params, ssncgop);
    
    % Update feasibility metrics
    Rp = Lx - y;
    primfeas = max(norm(Rp)/norm(y), norm(min(x,0))/norm(x));
    dualfeas = max(norm(max(LTv - n,0))/n, norm(u - v)/norm(u));
    eta = norm(y - 1./v)/norm(y);
    
    obj(1) = sum(x) - sum(log(Lx))/n - 1;
    obj(2) = sum(log(u))/n;
    relgap = abs(diff(obj)) / (1 + sum(abs(obj)));

    runhist.primobj(iter) = obj(1);
    runhist.dualobj(iter) = obj(2);
    runhist.primfeas(iter) = primfeas;
    runhist.dualfeas(iter) = dualfeas;
    runhist.relgap(iter) = relgap;
    
    if max(primfeas, dualfeas) < params.stoptol && eta < params.stoptol
        termination = 'converged'; break;
    end
    
    if infoNCG.breakyes >= 0
        sigma = max(sigmamin, sigma / 10);
    elseif iter > 1 && runhist.dualfeas(iter)/runhist.dualfeas(iter-1) > 0.6
        sigma = min(sigmamax, sigma * sqrt(3));
    end
end

info = struct('relgap', relgap, 'iter', iter, 'eta', eta, ...
              'termination', termination, 'maxfeas', max(primfeas, dualfeas));
end
