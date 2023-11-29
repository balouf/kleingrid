"""Main module."""
import time

import numpy as np
from numba import njit


@njit
def radius2shortcut(radius, seed=None):
    """

    Parameters
    ----------
    radius: :class:`int`
        Radius to draw shortcut from.
    seed: :class:`int`, optional
        Random seed for reproducibility

    Returns
    -------
    x: :class:`int`
        x relative coordinate
    y: :class:`int`
        y relative coordinate

    Examples
    --------

    >>> radius2shortcut(10, seed=42)
    (-9, 1)
    >>> radius2shortcut(2**40)
    (772882432861, -326629194915)
    """
    if seed is not None:
        np.random.seed(42)
    angle = np.random.randint(-2*radius+1, 2*radius+1)
    return (radius - np.abs(angle)),  (np.sign(angle) * (radius - np.abs(radius - np.abs(angle))))

    # @inline function Draw_r_smaller_than_1(n::Int64, r)
    #     expo = 2-r
    #     pow_max_radius = (2 * (n-1) + 1)^expo - 1
    #     function generator()
    #         radius = floor( (rand() * pow_max_radius + 1)^(1 / expo) )
    #         while rand() * radius * ((1 + 1 / radius)^expo - 1) > expo
    #             radius = floor( (rand() * pow_max_radius + 1)^(1 / expo) )
    #         end
    #         return radius2shortcut(Int64(radius))
    #     end
    #     return generator
    # end


def draw_r_lt_1(n, r):
    """

    Parameters
    ----------
    n: :class:`int`
        grid size
    r: :class:`float`
        Shortcut exponent (<1)

    Returns
    -------
    callable
        A shortcut generator

    Examples
    --------

    >>> gen = draw_r_lt_1(100, .5)
    >>> gen()  # doctest: +SKIP
    (133, 1)
    >>> gen = draw_r_lt_1(1000000000, .8)
    >>> gen()  # doctest: +SKIP
    (46006280, -169028952)
    """
    expo = 2 - r
    pow_max_radius = (2 * (n-1) + 1)**expo - 1

    def generator():
        radius = np.floor( (np.random.rand() * pow_max_radius +1)**(1/expo))
        while np.random.rand() * radius * ((1+1/radius)**expo -1 ) > expo:
            radius = np.floor((np.random.rand() * pow_max_radius + 1) ** (1 / expo))
        return radius2shortcut(np.int64(radius))

    return njit(generator)


@njit
def draw_r_eq_1(n):
    max_radius = 2*(n-1) + 1

    def generator():
        return radius2shortcut(np.random.randint1, max_radius)

    return njit(generator)


@njit
def draw_1_lt_r_lt_2(n, r):
    expo = 2 - r
    pow_max_radius = (2 * (n-1))**expo - 1
    p1 = 1/(1+pow_max_radius/expo)

    def generator():
        while True:
            if np.random.rand() < p1:
                return radius2shortcut(1)
            else:
                radius = np.ceil( (np.random.rand() * pow_max_radius +1)**(1/expo))
                if np.random.rand() * radius * ( 1 - (1 - 1/radius)**expo ) < expo:
                    return radius2shortcut(np.int64(radius))

    return njit(generator)


def draw_r_eq_2(n):
    max_radius = 2*(n-1)
    p1 = 1/(1+np.log(max_radius))

    @njit
    def generator():
        while True:
            if np.random.rand() < p1:
                return radius2shortcut(1)
            else:
                radius = np.ceil( (np.random.rand() * max_radius))
                if np.random.rand() * radius * np.log( 1 + 1/(radius-1)) < 1:
                    return radius2shortcut(np.int64(radius))

    return generator


@njit
def draw_2_lt_r(n, r):
    expo = r-2
    pow_max_radius = 1 / (2 * (n-1))**expo
    p1 = 1/(1 - pow_max_radius / expo)

    def generator(expo=expo, pow_max_radius=pow_max_radius, p1=p1):
        while True:
            if np.random.rand() < p1:
                return radius2shortcut(1)
            else:
                radius = np.ceil( 1 / (np.random.rand() * pow_max_radius + 1 )**(1/expo))
                if np.random.rand() * radius * ( 1 + 1/(radius-1)**expo -1) < expo:
                    return radius2shortcut(np.int64(radius))

    return njit(generator)


@njit
def edt_gen(gen, n, p, q, n_runs):
    steps = 0
    for _ in range(n_runs):
        s_x = np.random.randint(n)
        s_y = np.random.randint(n)
        d_x = np.random.randint(n)
        d_y = np.random.randint(n)
        d = np.abs(s_x-d_x) + np.abs(s_y-d_y)
        while d>0:
            d_s, sh_x, sh_y = 2*n, -1, -1
            for j in range(q):
                c_s, ch_x, ch_y = 2 * n, -1, -1
                while not ( (0 <= ch_x < n-1 ) and (0 <= ch_y < n-1 ) ):
                    r_x, r_y = gen()
                    ch_x = s_x + r_x
                    ch_y = s_y + r_y
                c_s = np.abs(d_x-ch_x) + np.abs(d_y-ch_y)
                if c_s < d_s:
                    d_s, sh_x, sh_y = c_s, ch_x, ch_y
            if d_s < d - p:
                d, s_x, s_y = d_s, sh_x, sh_y
            else:
                d = d-p
                delta_x = min(p, np.abs(d_x-s_x))
                delta_y = p - delta_x
                s_x += delta_x * np.sign(d_x-s_x)
                s_y += delta_y * np.sign(d_y - s_y)
            steps += 1
    return steps / n_runs


def expected_delivery_time(n=1000, r=2, p=1, q=1, n_runs=10000):
    """

    Parameters
    ----------
    n
    r
    p
    q
    n_runs

    Returns
    -------

    Examples
    --------

    >>> expected_delivery_time(n=100)
    """
    start = time.time()
    if r < 1:
        gen = draw_r_lt_1(n, r)
    elif r == 1:
        gen = draw_r_eq_1(n)
    elif r < 2:
        gen = draw_1_lt_r_lt_2(n, r)
    elif r == 2:
        gen = draw_r_eq_2(n)
    else:
        gen = draw_2_lt_r(n, r)
    edt = edt_gen(gen=gen, n=n, p=p, q=q, n_runs=n_runs)
    return {'expected_delivery_time': edt,
            'computation_time': time.time()-start,
            'parameters': {'n': n, 'r': r, 'p': p, 'q': q, 'n_runs': n_runs}}



