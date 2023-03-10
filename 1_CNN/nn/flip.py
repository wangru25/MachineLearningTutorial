from __future__ import division, absolute_import, print_function

import collections
import re
import sys
import warnings
import operator

import numpy as np
import numpy.core.numeric as _nx
from numpy.core import linspace, atleast_1d, atleast_2d, transpose
from numpy.core.numeric import (
    ones, zeros, arange, concatenate, array, asarray, asanyarray, empty,
    empty_like, ndarray, around, floor, ceil, take, dot, where, intp,
    integer, isscalar, absolute, AxisError
    )
from numpy.core.umath import (
    pi, multiply, add, arctan2, frompyfunc, cos, less_equal, sqrt, sin,
    mod, exp, log10, not_equal, subtract
    )
from numpy.core.fromnumeric import (
    ravel, nonzero, sort, partition, mean, any, sum
    )
from numpy.core.numerictypes import typecodes, number
from numpy.lib.twodim_base import diag
from .utils import deprecate
from numpy.core.multiarray import (
    _insert, add_docstring, digitize, bincount, normalize_axis_index,
    interp as compiled_interp, interp_complex as compiled_interp_complex
    )
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc
from numpy.compat import long
from numpy.compat.py3k import basestring

def flip(m, axis=None):
    """
    Reverse the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.
    .. versionadded:: 1.12.0
    Parameters
    ----------
    m : array_like
        Input array.
    axis : None or int or tuple of ints, optional
         Axis or axes along which to flip over. The default,
         axis=None, will flip over all of the axes of the input array.
         If axis is negative it counts from the last to the first axis.
         If axis is a tuple of ints, flipping is performed on all of the axes
         specified in the tuple.
         .. versionchanged:: 1.15.0
            None and tuples of axes are supported
    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.
    See Also
    --------
    flipud : Flip an array vertically (axis=0).
    fliplr : Flip an array horizontally (axis=1).
    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).
    flip(m, 1) is equivalent to fliplr(m).
    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.
    flip(m) corresponds to ``m[::-1,::-1,...,::-1]`` with ``::-1`` at all
    positions.
    flip(m, (0, 1)) corresponds to ``m[::-1,::-1,...]`` with ``::-1`` at
    position 0 and position 1.
    Examples
    --------
    >>> A = np.arange(8).reshape((2,2,2))
    >>> A
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> flip(A, 0)
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    >>> flip(A, 1)
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    >>> np.flip(A)
    array([[[7, 6],
            [5, 4]],
           [[3, 2],
            [1, 0]]])
    >>> np.flip(A, (0, 2))
    array([[[5, 4],
            [7, 6]],
           [[1, 0],
            [3, 2]]])
    >>> A = np.random.randn(3,4,5)
    >>> np.all(flip(A,2) == A[:,:,::-1,...])
    True
    """
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    if axis is None:
        indexer = (np.s_[::-1],) * m.ndim
    else:
        axis = _nx.normalize_axis_tuple(axis, m.ndim)
        indexer = [np.s_[:]] * m.ndim
        for ax in axis:
            indexer[ax] = np.s_[::-1]
        indexer = tuple(indexer)
    return m[indexer]
