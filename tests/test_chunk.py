import numpy as np

from nero_deploy.control import LatestActionChunk


def test_new_chunk_replaces_unconsumed_actions() -> None:
    chunks = LatestActionChunk()
    chunks.replace(np.array([[1.0], [2.0]], dtype=np.float32))
    assert chunks.next()[0].tolist() == [1.0]
    chunks.replace(np.array([[9.0], [10.0]], dtype=np.float32))
    assert chunks.next()[0].tolist() == [9.0]


def test_remaining_is_latest_only() -> None:
    chunks = LatestActionChunk()
    chunks.replace(np.array([[1.0], [2.0]], dtype=np.float32))
    chunks.next()
    np.testing.assert_allclose(chunks.remaining(), [[2.0]])
