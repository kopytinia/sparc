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

        state = np.random.get_state()  # saving current random seed
        np.random.seed(seed)  # setting a new one
        self._A = np.random.normal(0, 1 / n**0.5, size=[n, M * L])  # doing random
        np.random.set_state(state)  # returning old seed

        self._c = np.array([(n * P / L)**0.5] * L)

    def code(self, codewords: np.ndarray) -> np.ndarray:
        if codewords.ndim == 1:  # if input is vector
            codewords = codewords.reshape(1, -1)
        elif codewords.ndim > 2:
            assert False, f"input can be either vector or matrix"
        assert codewords.shape[1] == int(np.round(np.log2(self._M))) * self._L, f"Second dim of input must be equal L * log M"  # странная подпись ошибки

        betas = np.zeros((codewords.shape[0], self._L * self._M))
        for i in range(codewords.shape[0]):  # slow
            betas[i] = self._construct_beta(codewords[i])

        return betas @ self._A.T  # [batch_size, n]

    def decode(self, codewords: np.ndarray, iterat: int = 50) -> np.ndarray:
        if codewords.ndim == 1:  # if input is vector
            codewords = codewords.reshape(1, -1)
        elif codewords.ndim > 2:
            assert False, f"input can be either vector or matrix"
        assert codewords.shape[1] == self._n, f"size of input must be equal {self._n}" # Странная подпись ошибки

        batch_size = codewords.shape[0]
        estimated_beta = np.zeros((batch_size, self._M * self._L)) #predication of beta value before the channel
        z = np.zeros((batch_size, self._n)) # residual of decoding
        s = np.zeros((batch_size, self._M * self._L)) # test statistics
        divisor = np.zeros((batch_size, self._M * self._L)) # normalizing probability coefficient by segment
        t_squared = np.ones((batch_size, 1)) # residual variance

        # iterative decoding
        for _ in range(iterat):
            z = codewords - estimated_beta @ self._A.T + z / t_squared * (self._P - (estimated_beta**2).sum(axis=1, keepdims=True) / self._n) 
            s = z @ self._A + estimated_beta
            estimated_beta = np.exp(s * (self._n * self._P / self._L) ** 0.5 / t_squared)
            for l in range(self._L):
                divisor[:, self._M * l : self._M * (l + 1)] = (estimated_beta[:, self._M * l : self._M * (l + 1)]).sum(axis=1, keepdims=True)
            estimated_beta = np.divide(estimated_beta, divisor) * (self._n * self._P / self._L)**0.5
            t_squared = (z**2).sum(axis=1, keepdims=True) / self._n

        decoded_codewords = np.zeros((batch_size, self._L * int(np.round(np.log2(self._M)))))
        for i in range(batch_size):  # slow
            decoded_codewords[i] = self._decode_beta(estimated_beta[i])

        return decoded_codewords  # [batch_size, L * log M]
    
    def _construct_beta(self, codeword: np.ndarray) -> np.ndarray:
        beta = np.zeros(self._M * self._L)
        log2m = int(np.round(np.log2(self._M)))

        for i in range(self._L):
            # converting log M number of bits into decimal
            decimal = 0
            for bit in codeword[i * log2m : (i + 1) * log2m]:
                decimal = (decimal << 1) | bit

            beta[i * self._M + decimal] = self._c[i]

        return beta
    
    def _decode_beta(self, codeword: np.ndarray) -> np.ndarray:
            log2m = int(np.round(np.log2(self._M)))
            outp = np.zeros(log2m * self._L)
            for i in range(self._L):
                # converting decimal into log M number of bits
                decimal = np.argmax(codeword[self._M * i : self._M * (i + 1)])
                for j in range(log2m):
                    outp[log2m * (i + 1) - j - 1] = decimal % 2
                    decimal = decimal >> 1
            return outp


def _is_power_of_two(x):
    return (x != 0) and (x & (x - 1) == 0)


def ber(input_message: np.ndarray, output_message: np.ndarray) -> int:
    assert input_message.shape == output_message.shape, f"Messages must have equal lengths"

    return (input_message != output_message).mean(axis=-1)


def fer(input_message: np.ndarray, output_message: np.ndarray) -> int:
    assert input_message.shape == output_message.shape, f"Messages must have equal lengths"

    return (input_message != output_message).any(axis=-1).astype(int)
