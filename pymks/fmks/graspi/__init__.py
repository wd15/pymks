"""Top level functions for the Graspi graph descriptors for materials
science.

"""

import pandas
import numpy as np
import pytest

from ..func import fmap, curry, pipe


def graph_descriptors(data, delta_x=1.0, periodic_boundary=True):
    """Compute graph descriptors for multiple arrays

    Comute graph descriptors using GraSPI

    See XXX for more details

    The graph descriptors are given by

    $$ f[r \\; \\vert \\; l, l'] = \\frac{1}{S} \\sum_s m[s, l] m[s + r, l'] $$

    Note that this currently only works for two phase data. Also, this
    is not working with Dask yet..

    Args:
      data: array of phases (n_samples, n_x, n_y, n_z)
      delta_x: pixel size
      periodic_boundary: whether the boundaries are periodic

    Returns:
      A Pandas array with samples along rows and descriptors along columns

    Example, with 2 x (3, 3) arrays

    >>> data = np.array([[[0, 1, 0],
    ...                   [0, 1, 1],
    ...                   [1, 1, 1]],
    ...                  [[1, 1, 1],
    ...                   [0, 0, 0],
    ...                   [1, 1, 1]]])
    >>> print(graph_descriptors(data))
       b'STAT_n'  b'STAT_e'  ...  b'CT_f_D_tort1'  b'CT_f_A_tort1'
    0        9.0        6.0  ...              0.0         0.833333
    1        9.0        6.0  ...              0.0         1.000000
    <BLANKLINE>
    [2 rows x 20 columns]

    """
    return pipe(
        data,
        fmap(
            graph_descriptors_sample(
                delta_x=delta_x, periodic_boundary=periodic_boundary
            )
        ),
        list,
        pandas.DataFrame,
    )


@curry
def graph_descriptors_sample(data, delta_x=1.0, periodic_boundary=True):
    """Calculate graspi graph descriptors for a single array
    """
    graspi = pytest.importorskip("pymks.fmks.graspi.graspi")
    compute = lambda x: graspi.compute_descriptors(
        x,
        *(data.shape + (3 - len(data.shape)) * (1,)),
        pixelS=delta_x,
        if_per=periodic_boundary
    )
    return pipe(
        data,
        lambda x: x.astype(np.int32).flatten(),
        compute,
        fmap(lambda x: x[::-1]),
        dict,
    )
