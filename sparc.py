import numpy as np

class Sparc:
    _M: int = 0;
    _L: int = 0;
    _n: int = 0;
    _P: float = 0;
    _seed: int = 0;
    _A: np.ndarray = None;
    _c: np.ndarray = None;

    def __init__(self, M: int, L: int, n: int, P: float, seed: int = 0):
        assert _is_power_of_two(M), "M must be power of 2"

        self._M = M
        self._L = L
        self._n = n
        self._P = P
        self._seed = seed

        np.random.seed(seed)

        self._A = np.random.normal(0, 1 / n**0.5, size=[n, M * L])
        self._c = np.array([(n * P / L)**0.5] * L)
    
    def code(self, input: np.ndarray) -> np.ndarray:
        assert len(input) == int(np.round(np.log2(self._M))) * self._L, f"For now only inputs of size L * log M are acceptable"

        beta = self._construct_beta(input)
        return self._A @ beta

    def decode(self, input: np.ndarray, iterat: int = 50) -> np.ndarray:
        assert len(input) == self._n, f"size of input must be equal {self._n}"
        
        estimated_beta = np.zeros(self._M * self._L) #predication of beta value before the channel
        z = np.zeros(self._n) # residual of decoding
        s = np.zeros(self._M * self._L) # test statistics
        divisor = np.zeros(self._M * self._L) # normalizing probability coefficient by segment
        t_squared = 1 # residual variance
        
        # iterative decoding
        for i in range(iterat):
            z = input - self._A @ estimated_beta + z / t_squared * (self._P - (estimated_beta**2).sum()/self._n)
            s = np.transpose(self._A) @ z + estimated_beta
            estimated_beta = np.exp(s * (self._n * self._P / self._L) ** 0.5 / t_squared)
            for l in range(self._L):
                divisor[self._M * l : self._M * (l + 1)] = (estimated_beta[self._M * l : self._M * (l + 1)]).sum()
            estimated_beta = np.divide(estimated_beta, divisor) * (self._n * self._P / self._L)**0.5
            t_squared = (z**2).sum()/self._n
        return self._decode_beta(estimated_beta)

    def _construct_beta(self, input: np.ndarray) -> np.ndarray:
        beta = np.zeros(self._M * self._L)
        log2m = int(np.round(np.log2(self._M)))

        for i in range(self._L):
            # converting log M number of bits into decimal
            decimal = 0
            for bit in input[i * log2m : (i + 1) * log2m]:
                decimal = (decimal << 1) | bit

            beta[i * self._M + decimal] = self._c[i]

        return beta
    
    def _decode_beta(self, input: np.ndarray) -> np.ndarray:
            log2m = int(np.round(np.log2(self._M)))
            outp = np.zeros(log2m * self._L)
            for i in range(self._L):
                # converting decimal into log M number of bits
                decimal = np.argmax(input[self._M * i: self._M * (i + 1)])
                for j in range(log2m):
                    outp[log2m * (i + 1) - j - 1] = decimal % 2
                    decimal = decimal >> 1
            return outp


def _is_power_of_two(x):
    return (x != 0) and (x & (x-1) == 0)