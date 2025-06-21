#include "mex.h"
#include <math.h>
#include <matrix.h>

#define SQR(x) ((x)*(x))

double realdot(const double *x, const double *y, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

void subscalarmul(double *x, double alpha, const double *y, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] -= alpha * y[i];
    }
}

void lbsolve(double *y, const double *U, const double *x, int n) {
    for (int k = 0; k < n; ++k, U += n) {
        y[k] = (x[k] - realdot(y, U, k)) / U[k];
    }
}

void ubsolve(double *x, const double *U, int n) {
    const double *u = U + n * n;
    for (int j = n - 1; j >= 0; --j) {
        u -= n;
        x[j] /= u[j];
        subscalarmul(x, x[j], u, j);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 2) {
        mexErrMsgTxt("mextriang requires at least 2 input arguments.");
    }
    if (nlhs > 1) {
        mexErrMsgTxt("mextriang returns only one output.");
    }

    const double *U = mxGetPr(prhs[0]);
    int n = mxGetM(prhs[0]);
    if (mxGetN(prhs[0]) != n || mxIsSparse(prhs[0])) {
        mexErrMsgTxt("U must be dense and square.");
    }

    int mode = (nrhs > 2) ? (int)(*mxGetPr(prhs[2])) : 1;

    const mxArray *bMat = prhs[1];
    if (mxGetM(bMat) * mxGetN(bMat) != n) {
        mexErrMsgTxt("Dimension mismatch between U and b.");
    }

    double *b;
    if (mxIsSparse(bMat)) {
        b = mxCalloc(n, sizeof(double));
        const double *btmp = mxGetPr(bMat);
        const mwIndex *irb = mxGetIr(bMat);
        const mwIndex *jcb = mxGetJc(bMat);
        for (mwIndex k = jcb[0]; k < jcb[1]; ++k) {
            b[irb[k]] = btmp[k];
        }
    } else {
        b = mxGetPr(bMat);
    }

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double *y = mxGetPr(plhs[0]);

    if (mode == 1) {
        memcpy(y, b, sizeof(double) * n);
        ubsolve(y, U, n);
    } else {
        lbsolve(y, U, b, n);
    }

    if (mxIsSparse(bMat)) {
        mxFree(b);
    }
}
