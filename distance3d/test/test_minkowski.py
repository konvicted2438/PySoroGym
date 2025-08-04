<<<<<<< HEAD
import numpy as np
from distance3d.minkowski import minkowski_sum


def test_minkowski_sum():
    vertices1 = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    vertices2 = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    ms = minkowski_sum(vertices1, vertices2)
    assert len(ms) == 4
    assert np.array([0.0, -1.0, 1.0]) in ms
    assert np.array([0.0, 1.0, 1.0]) in ms
    assert np.array([1.0, -1.0, 0.0]) in ms
    assert np.array([1.0, 1.0, 0.0]) in ms
=======
import numpy as np
from distance3d.minkowski import minkowski_sum


def test_minkowski_sum():
    vertices1 = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    vertices2 = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    ms = minkowski_sum(vertices1, vertices2)
    assert len(ms) == 4
    assert np.array([0.0, -1.0, 1.0]) in ms
    assert np.array([0.0, 1.0, 1.0]) in ms
    assert np.array([1.0, -1.0, 0.0]) in ms
    assert np.array([1.0, 1.0, 0.0]) in ms
>>>>>>> a104757f44e5217f3a284e59d7240a5e9f782c7f
