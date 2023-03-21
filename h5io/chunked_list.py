"""Provide chunked list class for storing chunked listsin HDF5."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: BSD (3-clause)

from typing import Tuple, Union, List, Generator


class ChunkedList:
    """Class for chunked lists

    Parameters
    ----------
    data : list
        The portion of the list to store
    size : int
        the size of the complete list
    offset: int
        The offset of the chunk in the original list
    """

    def __init__(
        self,
        data: List,
        size: int,
        offset: int,
    ) -> None:
        self.chunkdata = data
        self.size = size
        self.offset = offset

        if len(self.chunkdata) > self.size:
            raise ValueError(
                f"The length of the chunk ({len(self.chunkdata)}) is too large"
                f" for the given length ({self.size})."
            )

        if self.offset < 0:
            raise ValueError(
                f"The chunk offset ({self.offset}) must be positive."
            )

        if self.offset > self.size:
            raise ValueError(
                f"The chunk offset ({self.offset}) is too large for the given "
                f"length ({self.size})."
            )

        if len(self.chunkdata) + self.offset > self.size:
            raise ValueError(
                f"The chunk offset ({self.offset}) is too large for the given "
                f"length ({self.size}) and chunk data."
            )
        
    @property
    def chunkrange(self) -> range:
        """Return the range of the chunk to write to the dataset."""
        return range(self.offset, self.offset + len(self.chunkdata))

    @property
    def enumerate(self) -> Generator[Tuple[int, int], None, None]:
        """Return the range of the chunk to write to the dataset."""
        for i, x in enumerate(self.chunkdata):
            yield i + self.offset, x