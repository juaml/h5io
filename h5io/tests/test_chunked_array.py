import numpy as np
from numpy.testing import assert_array_almost_equal
from pathlib import Path
from h5io._h5io import write_hdf5, read_hdf5
from h5io.chunked_array import ChunkedArray


def test_chunked_array(tmpdir):
    tempdir = Path(tmpdir)
    test_file = tempdir / "test.hdf5"
    n_chunks = 6
    chunk_size = (100, 100, 2)
    shape = (100, 100, chunk_size[2] * n_chunks)
    dtype = np.float32
    all_data = np.random.rand(*shape)

    for t_chunk in range(n_chunks):
        st = t_chunk * chunk_size[2]
        end = (t_chunk + 1) * chunk_size[2]
        t_data = all_data[:, :, st:end]
        # t_data[:, :, :] = t_chunk
        chunk = ChunkedArray(
            t_data, shape=shape, chunk_size=chunk_size, n_chunk=t_chunk
        )
        overwrite="update" if t_chunk > 0 else True
        write_hdf5(
            test_file.as_posix(), chunk, overwrite=overwrite, compression=0
        )

    wrote_data = read_hdf5(test_file.as_posix())
    assert_array_almost_equal(wrote_data, all_data)
