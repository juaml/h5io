import math
from pathlib import Path

from h5io._h5io import read_hdf5, write_hdf5
from h5io.chunked_list import ChunkedList


def test_chunked_list(tmpdir):
    tempdir = Path(tmpdir)
    test_file = tempdir / "test_chunked_list.hdf5"

    all_data = list(range(100))
    chunk_size = 7
    n_chunks = math.ceil(len(all_data) / chunk_size)

    for t_chunk in range(n_chunks):
        st = t_chunk * chunk_size
        end = (t_chunk + 1) * chunk_size
        t_data = all_data[st:end]
        chunk = ChunkedList(t_data, len(all_data), t_chunk * chunk_size)
        overwrite = "update" if t_chunk > 0 else True
        write_hdf5(
            test_file.as_posix(), chunk, overwrite=overwrite, compression=0
        )

    wrote_data = read_hdf5(test_file.as_posix())
    assert wrote_data == all_data


def test_chunked_list_random_idx(tmpdir):
    tempdir = Path(tmpdir)
    test_file = tempdir / "test_chunked_list.hdf5"

    all_data = list(range(100))
    chunk_size = 7
    for st in range(0, len(all_data) - chunk_size + 1):
        end = st + chunk_size
        t_data = all_data[st:end]
        chunk = ChunkedList(t_data, len(all_data), st)
        overwrite = "update" if st > 0 else True
        write_hdf5(
            test_file.as_posix(), chunk, overwrite=overwrite, compression=0
        )

    wrote_data = read_hdf5(test_file.as_posix())
    assert wrote_data == all_data