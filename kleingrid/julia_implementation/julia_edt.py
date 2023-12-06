import time

from juliacall import Main as jl  # type: ignore
from kleingrid.utils import Result
from kleingrid import __file__ as d
from pathlib import Path

jl.include(str(Path(d).parent / "julia_implementation/kleingrid.jl"))


def julia_edt(n=1000, r=2, p=1, q=1, n_runs=10000):
    """
    Parameters
    ----------
    n: :class:`int`, default=1000
        Grid siDe
    r: :class:`float`, default=2.0
        Shortcut exponent
    p: :class:`int`, default=1
        Local range
    q: :class:`int`, default=1
        Number of shortcuts
    n_runs: :class:`int`, default=10000
        Number of routes to compute

    Returns
    -------
    :class:`~kleingrid.kleingrid.EDT`

    Examples
    --------

    >>> jl.set_seed(42)
    Julia: TaskLocalRNG()
    >>> julia_edt(n=1000, r=.5, n_runs=100)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=79.93, process_time=..., n=1000, r=0.5, p=1, q=1, n_runs=100, julia=True)
    >>> julia_edt(n=1000, r=1, n_runs=100)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=66.39, process_time=..., n=1000, r=1, p=1, q=1, n_runs=100, julia=True)
    >>> julia_edt(n=1000, r=1.5, n_runs=100)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=57.91, process_time=..., n=1000, r=1.5, p=1, q=1, n_runs=100, julia=True)
    >>> julia_edt(n=1000, r=2, n_runs=100)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=69.53, process_time=..., n=1000, r=2, p=1, q=1, n_runs=100, julia=True)
    >>> julia_edt(n=1000, r=2.5, n_runs=100)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=187.74, process_time=..., n=1000, r=2.5, p=1, q=1, n_runs=100, julia=True)
    >>> julia_edt(n=10000000000, r=1.5, n_runs=10000, p=2, q=2)  # doctest: +SKIP
    Result(edt=6846.631, process_time=13.0, n=10000000000, r=1.5, p=2, q=2, n_runs=10000, julia=True)
    """
    start = time.process_time()
    edt = jl.expected_delivery_time(n, r, p, q, n_runs)
    return Result(edt=edt, process_time=time.process_time() - start,
                  n=n, r=r, p=p, q=q, n_runs=n_runs, julia=True)

