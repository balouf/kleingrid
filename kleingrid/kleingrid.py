import inspect
import time
import numpy as np

from joblib import Parallel, delayed  # type: ignore
from tqdm import tqdm
from functools import cache

from kleingrid.python_implementation.python_edt import python_edt
from kleingrid.julia_implementation.julia_edt import julia_edt


def compute_edt(n=1000, r=2, p=1, q=1, n_runs=10000, julia=True, numba=True, parallel=False):
    if julia:
        return julia_edt(n=n, r=r, p=p, q=q, n_runs=n_runs)
    else:
        return python_edt(n=n, r=r, p=p, q=q, n_runs=n_runs, numba=numba, parallel=parallel)


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def parallelize(values, function=None, default=None, n_jobs=-1):
    if function is None:
        function = compute_edt
    if default is None:
        default = get_default_args(function)

    def with_key(v):
        return function(**{**default, **v})

    return Parallel(n_jobs=n_jobs)(tqdm((
        delayed(with_key)(v) for v in values), total=len(values)))


def simple_function(n=10000, n_runs=10000, **kwargs):
    def f(r):
        return compute_edt(r=r, n=n, n_runs=n_runs, **kwargs).edt
    return cache(f)


def get_target(f, a, b, t):
    fa = f(a)
    fb = f(b)
    c = (a+b)/2
    fc = f(c)
    while fa < fc < fb:
        if fc < t:
            a, fa = c, fc
        else:
            b, fv = c, fc
        c = (a+b)/2
        print(c)
        fc = f(c)
    return c


def gss(f, a, b, cmin, tol=1e-5):
    """Golden-section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]

    Example:
    >>> f = cache(lambda x: (x-2)**2)
    >>> x = gss(f, 1, 5, 50)
    >>> f"{x:.4f}"
    '2.0000'

    """
    gr = (np.sqrt(5) + 1) / 2

    while abs(b - a) > tol:
        c = b - (b - a) / gr
        fc = f(c)
        d = a + (b - a) / gr
        fd = f(d)
        # print(c, fc, d, fd, cmin)
        # if (fd > cmin) and (fc > cmin):
            # print(fc>cmin)
            # print(fd > cmin)
            # print("done")
            # break
        if fc < fd:
            b = d
            cmin = fc
        else:
            a = c
            cmin = fd

    return (b + a) / 2


def get_bounds(n, offset_start=.1, n_runs=10000):
    f = simple_function(n=n, n_runs=n_runs)
    ff = simple_function(n=n, n_runs=100 * n_runs)
    r = 2
    ref = f(r)
    r += offset_start
    while f(r) < 2 * ref:
        r += offset_start
        print(r)
    up2b = r

    up2 = get_target(f, 2, up2b, 2 * ref)

    r = 2 - offset_start
    while (f(r) < ref) and r >= 0:
        r -= offset_start
        print(r)

    loa = r

    m = gss(f, loa, 2, ref)

    if r < 0:
        return r, r, m, up2

    lo = get_target(f, m, loa, ref)

    while (f(r) < 2 * ref) and r >= 0:
        r -= offset_start
        print(r)
    lo2a = r

    if lo2a < 0:
        return lo2a, lo, m, up2

    b = min(lo2a + offset_start, lo)

    lo2 = get_target(f, b, lo2a, 2 * ref)

    return lo2, lo, m, up2


def estimate_alpha(r, p=15, budget=20):
    start = time.time()
    v1 = compute_edt(n=2**p, r=r)
    v2 = compute_edt(n=2**(p+1), r=r)
    while ((time.time() - start) < budget):
        p += 1
        v1 = v2
        v2 = compute_edt(n=2**(p+1), r=r)
        print(np.log2(v2.edt)-np.log2(v1.edt))
    return np.log2(v2.edt)-np.log2(v1.edt), p+1
