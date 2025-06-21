#include "mex.h"
#include <math.h>
#include <matrix.h>

/*
 * Solve R' * x = b where R is upper-triangular.
 * Usage: x = mexfwsolve(R, b)
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgTxt("mexfwsolve requires exactly 2 input arguments.");
    }
    if (nlhs > 1) {
        mexErrMsgTxt("mexfwsolve returns only one output.");
    }

    const mxArray *RMat = prhs[0];
    const mxArray *bMat = prhs[1];
    if (!mxIsSparse(RMat)) {
        mexErrMsgTxt("R must be sparse.");
    }

    mwSize n = mxGetM(bMat);
    if (mxGetN(RMat) != n || mxGetM(RMat) != n) {
        mexErrMsgTxt("R must be square and compatible with b.");
    }

    const double *R = mxGetPr(RMat);
    const mwIndex *irR = mxGetIr(RMat);
    const mwIndex *jcR = mxGetJc(RMat);

    double *b;
    if (mxIsSparse(bMat)) {
        double *btmp = mxGetPr(bMat);
        const mwIndex *irb = mxGetIr(bMat);
        const mwIndex *jcb = mxGetJc(bMat);
        b = mxCalloc(n, sizeof(double));
        for (mwIndex k = jcb[0]; k < jcb[1]; ++k) {
            b[irb[k]] = btmp[k];
        }
    } else {
        b = mxGetPr(bMat);
    }

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double *x = mxGetPr(plhs[0]);

    x[0] = b[0] / R[0];
    for (mwSize j = 1; j < n; ++j) {
        double sum = 0.0;
        mwIndex kstart = jcR[j];
        mwIndex kend = jcR[j + 1] - 1;
        for (mwIndex k = kstart; k < kend; ++k) {
            sum += R[k] * x[irR[k]];
        }
        x[j] = (b[j] - sum) / R[kend];
    }

    if (mxIsSparse(bMat)) {
        mxFree(b);
    }
}
