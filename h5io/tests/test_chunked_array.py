from pathlib import Path

import numpy as np
from h5io._h5io import read_hdf5, write_hdf5
from h5io.chunked_array import ChunkedArray
from numpy.testing import assert_array_almost_equal


def test_chunked_array_2d(tmpdir):
    tempdir = Path(tmpdir)
    test_file = tempdir / "test_chunked_array_2d.hdf5"
    n_chunks = 3
    chunk_size = (3, 2)
    shape = (3, chunk_size[1] * n_chunks)
    all_data = np.random.rand(*shape)

    for t_chunk in range(n_chunks):
        st = t_chunk * chunk_size[1]
        end = (t_chunk + 1) * chunk_size[1]
        t_data = all_data[:, st:end]
        chunk = ChunkedArray(
            t_data, shape=shape, chunk_size=chunk_size, n_chunk=t_chunk
        )
        overwrite = "update" if t_chunk > 0 else True
        write_hdf5(
            test_file.as_posix(), chunk, overwrite=overwrite, compression=0
        )

    wrote_data = read_hdf5(test_file.as_posix())
    assert_array_almost_equal(wrote_data, all_data)


def test_chunked_array_2d_inside_dict(tmpdir):
    tempdir = Path(tmpdir)
    test_file = tempdir / "test_chunked_array_2d.hdf5"
    n_chunks = 3
    chunk_size = (3, 2)
    shape = (3, chunk_size[1] * n_chunks)
    all_data = np.random.rand(*shape)

    to_save = {}

    for t_chunk in range(n_chunks):
        st = t_chunk * chunk_size[1]
        end = (t_chunk + 1) * chunk_size[1]
        t_data = all_data[:, st:end]
        chunk = ChunkedArray(
            t_data, shape=shape, chunk_size=chunk_size, n_chunk=t_chunk
        )
        overwrite = "update" if t_chunk > 0 else True
        to_save["data"] = chunk
        write_hdf5(
            test_file.as_posix(), to_save, overwrite=overwrite, compression=0
        )

    wrote_data = read_hdf5(test_file.as_posix())
    assert_array_almost_equal(wrote_data["data"], all_data)


def test_chunked_array_3d(tmpdir):
    tempdir = Path(tmpdir)
    test_file = tempdir / "test_chunked_array_3d.hdf5"
    n_chunks = 6
    chunk_size = (100, 100, 2)
    shape = (100, 100, chunk_size[2] * n_chunks)
    all_data = np.random.rand(*shape)

    for t_chunk in range(n_chunks):
        st = t_chunk * chunk_size[2]
        end = (t_chunk + 1) * chunk_size[2]
        t_data = all_data[:, :, st:end]
        # t_data[:, :, :] = t_chunk
        chunk = ChunkedArray(
            t_data, shape=shape, chunk_size=chunk_size, n_chunk=t_chunk
        )
        overwrite = "update" if t_chunk > 0 else True
        write_hdf5(
            test_file.as_posix(), chunk, overwrite=overwrite, compression=0
        )

    wrote_data = read_hdf5(test_file.as_posix())
    assert_array_almost_equal(wrote_data, all_data)
