"""Main module."""

import numpy as np
from numba import njit


# @inline function radius2shortcut(radius::Int64)
#         angle::Int64 = rand((-2*radius +1):(2*radius))
#         return (radius - abs(angle)),  (sign(angle) * (radius - abs(radius - abs(angle))))


@njit
def radius2shortcut(radius):
    """

    Parameters
    ----------
    radius: :class:`int`
        Radius to draw shortcut from.

    Returns
    -------
    x: :class:`int`
        x relative coordinate
    y: :class:`int`
        y relative coordinate

    Examples
    --------

    >>> np.random.seed(42)
    >>> radius2shortcut(10)
    (-9, 1)
    >>> radius2shortcut(2**40)
    (772882432861, -326629194915)
    """
    angle = np.random.randint(-2*radius+1, 2*radius+1, dtype=np.int64)
    return (radius - np.abs(angle)),  (np.sign(angle) * (radius - np.abs(radius - np.abs(angle))))
