import numpy as np

class Sparc:
    _M = 0;
    _L = 0;
    _n = 0;
    _P = 0;
    _seed = 0;

    def __init__(self, M: int, L: int, n: int, P: float, seed: int):
        return
    
    def code(self, input: np.ndarray) -> np.ndarray:
        assert len(input) == np.round(np.log2(self._M)) * self._L, f"For now only inputs of size L * log M are acceptable"
        return

    def decode(self, input: np.ndarray) -> np.ndarray:
        assert len(input) == self._n, f"size of input must be equal {self._n}"
        return
    