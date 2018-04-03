import numpy as np

"""
python MVGC rewritten from Matlab MVGC toolbox

refs:

http://erramuzpe.github.io/C-PAC/blog/2015/06/10/multivariate-granger-causality-in-python-for-fmri-timeseries-analysis/
https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mlab.py
https://github.com/benpolletta/MBP-Code/blob/master/mvgc_v1.0/gc/autocov_to_mvgc.m
https://github.com/benpolletta/MBP-Code/blob/master/mvgc_v1.0/core/autocov_to_var.m
https://github.com/benpolletta/MBP-Code/blob/master/mvgc_v1.0/core/tsdata_to_autocov.m
"""


def detrend_mean(x, axis=None):
    # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mlab.py
    x = np.asarray(x)

    if axis is not None and axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    # short-circuit 0-D array.
    if not x.ndim:
        return np.array(0., dtype=x.dtype)

    # short-circuit simple operations
    if axis == 0 or axis is None or x.ndim <= 1:
        return x - x.mean(axis)

    ind = [slice(None)] * x.ndim
    ind[axis] = np.newaxis
    return x - x.mean(axis)[ind]


def tsdata_to_autocov(X, q):
    """
    Calculate sample autocovariance sequence from time series data

    :param X:   multi-trial time series data
    :param q:   number of lags
    :return G:  sample autocovariance sequence
    """
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=2)
        [n, m, N] = np.shape(X)
    else:
        [n, m, N] = np.shape(X)
    X = detrend_mean(X, axis=1)
    G = np.zeros((n, n, (q+1)))

    for k in range(q+1):
        M = N * (m-k)
        Xk = np.asmatrix(X[:, k:m, :].reshape((n, M)))
        Xm = np.asmatrix(X[:, 0:m-k, :].reshape((n, M)).conj().T)
        G[:, :, k] = Xk * Xm / M - 1.0
    # G.shape = (n, n, q+1)
    return G


def autocov_to_var(G):
    """
    Calculate VAR parameters from autocovariance sequence

    :param G:           autocovariance sequence
    :return [A, SIG]:   A - VAR coefficients matrix, SIG - residuals covariance matrix
    """
    [n, _, q1] = G.shape
    q = q1 - 1
    qn = q * n
    G0 = np.asmatrix(G[:, :, 0])                                            # covariance
    GF = np.asmatrix(G[:, :, 1:].reshape((n, qn)).T)                        # forward autoconv seq
    GB = np.asmatrix(G[:, :, 1:].transpose(0, 2, 1).reshape(qn, n))         # backward autoconv seq

    AF = np.asmatrix(np.zeros((n, qn)))                                     # forward  coefficients
    AB = np.asmatrix(np.zeros((n, qn)))                                     # backward coefficients

    # initialise recursion
    k = 1                                                                   # model order
    r = q - k
    kf = np.arange(0, k*n)                                                  # forward indices
    kb = np.arange(r*n, qn)                                                 # backward indices

    G0I = G0.I

    AF[:, kf] = GB[kb, :] * G0I
    AB[:, kb] = GF[kf, :] * G0I

    for k in np.arange(2, q+1):
        GBx = GB[(r-1)*n: r*n, :]
        DF = GBx - AF[:, kf] * GB[kb, :]
        VB = G0 - AB[:, kb] * GB[kb, :]
        AAF = DF * VB.I

        GFx = GF[(k-1)*n: k*n, :]
        DB = GFx - AB[:, kb] * GF[kf, :]
        VF = G0 - AF[:, kf] * GF[kf, :]
        AAB = DB * VF.I

        AFPREV = AF[:, kf]
        ABPREV = AB[:, kb]
        r = q - k

        kf = np.arange(0, k*n)
        kb = np.arange(r*n, qn)

        AF[:, kf] = np.hstack([AFPREV - AAF * ABPREV, AAF])
        AB[:, kb] = np.hstack([AAB, ABPREV - AAB * AFPREV])

    SIG = G0 - AF * GF
    AF = np.asarray(AF).reshape((n, n, q))
    return [AF, SIG]


def autocov_to_mvgc(G, x, y):
    """
    Calculate conditional time-domain MVGC (multivariate Granger causality)

    from the variable |Y| (specified by the vector of indices |y|)
    to the variable |X| (specified by the vector of indices |x|),
    conditional on all other variables |Z| represented in |G|, for
    a stationary VAR process with autocovariance sequence |G|.
    See ref. [1] for details.
    [1] L. Barnett and A. K. Seth, <http://www.sciencedirect.com/science/article/pii/S0165027013003701
    The MVGC Multivariate Granger Causality Toolbox: A New Approach to Granger-causalInference>,
    _J. Neurosci. Methods_ 223, 2014 [ <matlab:open('mvgc_preprint.pdf') preprint> ].

    Ref:
    http://erramuzpe.github.io/C-PAC/blog/2015/06/10/multivariate-granger-causality-in-python-for-fmri-timeseries-analysis/
    https://github.com/benpolletta/MBP-Code/blob/ced60ba9fdcea1efd185dc331023e788b9c48eb6/mvgc_v1.0/gc/autocov_to_mvgc.m


    :param G:   autocovariance sequence
    :param x:   vector of indices of target (causee) multi-variable
    :param y:   vector of indices of source (causal) multi-variable
    :return F:  Granger causality
    """
    n = G.shape[0]
    z = np.arange(n)
    z = np.delete(z, [np.hstack((x, y))])
    # indices of other variables (to condition out)
    xz = np.hstack((x, z))
    xzy = np.hstack((xz, y))
    F = 0

    # full regression
    ixgrid1 = np.ix_(xzy, xzy)
    [AF, SIG] = autocov_to_var(G[ixgrid1])

    # reduced regression
    ixgrid2 = np.ix_(xz, xz)
    [AF, SIGR] = autocov_to_var(G[ixgrid2])

    ixgrid3 = np.ix_(x, x)

    F = np.log(np.linalg.det(SIGR[ixgrid3])) - np.log(np.linalg.det(SIG[ixgrid3]))
    return F


if __name__ == "__main__":
    X = np.random.rand(100, 25)
    G = tsdata_to_autocov(X, 5)
    AF, SIG = autocov_to_var(G)
    print(autocov_to_mvgc(G, np.array([0, 1, 2]), np.array([4, 5, 6])))
