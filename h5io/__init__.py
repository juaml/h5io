"""Python Objects Onto HDF5
"""


from ._h5io import (
    has_hdf5,  # noqa, analysis:ignore
    read_hdf5,  # noqa, analysis:ignore
    write_hdf5,  # noqa, analysis:ignore
    _TempDir,  # noqa, analysis:ignore
    object_diff,  # noqa, analysis:ignore
    list_file_contents,  # noqa, analysis:ignore
)
from ._version import __version__
from .chunked_array import ChunkedArray  # noqa, analysis:ignore
from .chunked_list import ChunkedList  # noqa, analysis:ignore
