"""Provide chunked array class for storing chunked numpy.ndarray in HDF5."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: BSD (3-clause)

from typing import Tuple

import numpy as np


class ChunkedArray:
    """Class for chunked numpy.ndarray.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be stored in chunks.
    shape : tuple of int
        A 3-tuple of int for (x, y, t).
    chunk_size : tuple of int
        The 3-tuple of int as chunk size of chunked storage.
    n_chunk : int
        The chunk count of chunked storage.

    Notes
    -----
    The first two dimensions of the ``chunk_size`` and the ``shape``
    must be the same.

    """
    def __init__(
        self,
        data: np.ndarray,
        shape: Tuple[int, int, int],
        chunk_size: Tuple[int, int, int],
        n_chunk: int,
    ) -> None:
        self.chunkdata = data
        self.shape = shape
        self.chunk_size = chunk_size
        self.n_chunk = n_chunk

        self._chunk_start = self.n_chunk * self.chunk_size[2]
        self._chunk_end = (self.n_chunk + 1) * self.chunk_size[2]
        if self._chunk_end > self.shape[2]:
            self._chunk_end = self.shape[2]

        if self._chunk_start > self.shape[2]:
            raise ValueError(
                f"The chunk number ({n_chunk}) is too large for the given "
                f"shape ({shape})."
            )
        t_chunk_size = self._chunk_end - self._chunk_start
        if t_chunk_size != self.chunkdata.shape[2]:
            raise ValueError(
                f"The chunk size ({chunk_size}) does not match the given "
                f"data shape ({self.chunkdata.shape})."
            )

    @property
    def chunkslice(self) -> Tuple[slice, slice, slice]:
        """Return the slice of the chunk to write to the dataset."""
        out = (
            slice(0, self.shape[0], 1),
            slice(0, self.shape[1], 1),
            slice(self._chunk_start, self._chunk_end, 1),
        )
        return out
