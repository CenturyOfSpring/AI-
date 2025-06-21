#include "mex.h"
#include <math.h>
#include <matrix.h>

/*
 * Solve R * x = b, where R is upper-triangular and stored in transposed (lower) form.
 * Usage in MATLAB: x = mexbwsolve(Rt, b), where Rt = R'.
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgTxt("mexbwsolve requires exactly 2 input arguments.");
    }
    if (nlhs > 1) {
        mexErrMsgTxt("mexbwsolve returns only one output argument.");
    }

    const mxArray *RtMat = prhs[0];
    const mxArray *bMat = prhs[1];
    if (!mxIsSparse(RtMat)) {
        mexErrMsgTxt("Rt must be a sparse matrix.");
    }

    mwSize n = mxGetM(bMat);
    if (mxGetN(RtMat) != n || mxGetM(RtMat) != n) {
        mexErrMsgTxt("Rt must be a square matrix compatible with b.");
    }

    const double *Rt = mxGetPr(RtMat);
    const mwIndex *irRt = mxGetIr(RtMat);
    const mwIndex *jcRt = mxGetJc(RtMat);

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

    x[n-1] = b[n-1] / Rt[jcRt[n] - 1];

    for (mwSize j = n - 1; j > 0; --j) {
        double sum = 0.0;
        mwIndex kstart = jcRt[j - 1] + 1;
        mwIndex kend = jcRt[j];
        for (mwIndex k = kstart; k < kend; ++k) {
            sum += Rt[k] * x[irRt[k]];
        }
        x[j - 1] = (b[j - 1] - sum) / Rt[kstart - 1];
    }

    if (mxIsSparse(bMat)) {
        mxFree(b);
    }
}
