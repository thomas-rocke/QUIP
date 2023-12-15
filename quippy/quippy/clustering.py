from quippy.clustering_module import cur_scores
import numpy as np


def get_cur_scores(vecs, clip_scores=True):
    '''
    Perform a CUR decomposition on the NxM array vecs, returning an N-length array of CUR scores
    '''

    assert len(vecs.shape) in [1, 2]

    if len(vecs.shape) == 1:
        arr = vecs[:, np.newaxis].copy()
    else:
        arr = vecs.copy()
    N = arr.shape[0]

    scores = np.zeros(N, dtype=np.float64)

    cur_scores(arr.T, scores, clip_scores=clip_scores)

    # Normalise scores
    scores /= np.sum(scores)

    return scores