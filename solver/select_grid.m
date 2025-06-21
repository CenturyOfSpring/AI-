function [U, m] = select_grid(X, grid_option, m)
% Select grid points U from data X
% Inputs:
%   X           - n x d matrix of observations
%   grid_option - how to generate grid points
%       1: use all data points
%       2: random subsample of data points
%       3: uniform grid in bounding box
%       4: lin-log mixed grid
%   m           - desired number of grid points (optional, depends on option)
% Output:
%   U - m x d matrix of grid points
%   m - number of grid points used

[n, d] = size(X);

if nargin < 2 || isempty(grid_option)
    grid_option = 1;
end

switch grid_option
    case 1
        U = X;
        m = n;

    case 2
        if nargin < 3 || isempty(m)
            m = round(sqrt(n));
        end
        m = min(m, n);
        idx = randperm(n, m);
        U = X(idx, :);

    case 3
        if nargin < 3 || isempty(m)
            m = n;
        end
        xmin = min(X);
        xmax = max(X);
        xspan = xmax - xmin;
        ratio = xspan(1) / max(xspan(2), eps);
        my = round(sqrt(m / ratio));
        mx = round(sqrt(m * ratio));
        m = mx * my;

        xgrid = linspace(xmin(1), xmax(1), mx);
        ygrid = linspace(xmin(2), xmax(2), my);
        [Xg, Yg] = meshgrid(xgrid, ygrid);
        U = [Xg(:), Yg(:)];

    case 4
        if nargin < 3 || isempty(m)
            m = n;
        end
        xmin = min(X);
        xmax = max(X);
        xspan = xmax - xmin;
        ratio = xspan(1) / max(xspan(2), eps);
        my = round(sqrt(m / ratio));
        mx = round(sqrt(m * ratio));
        m = mx * my;

        xgrid = linspace(xmin(1), xmax(1), mx);
        ygrid = logspace(log10(max(xmin(2), eps)), log10(xmax(2)), my);
        [Xg, Yg] = meshgrid(xgrid, ygrid);
        U = [Xg(:), Yg(:)];

    otherwise
        U = X;
        m = n;
end

end
