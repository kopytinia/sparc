import numpy as np
from datetime import datetime
import os
import pyldpc

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

        self.input_length = int(np.round(np.log2(self._M))) * L   # getter?
        self.output_length = n   # getter?

    def __str__(self):
        return 'M={}_L={}_n={}_P={}_seed={}'.format(self._M, self._L, self._n, self._P, self._seed)

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

    def decode(self, codewords: np.ndarray, iterat: int = 50, return_llr=False) -> np.ndarray:
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

        decoded_codewords = np.zeros((batch_size, self._L * int(np.round(np.log2(self._M)))), dtype=float)
        for i in range(batch_size):  # slow
            if return_llr:
                decoded_codewords[i] = self._decode_beta_llr(estimated_beta[i])
            else:
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
            outp = np.zeros(log2m * self._L, dtype=int)
            for i in range(self._L):
                # converting decimal into log M number of bits
                decimal = np.argmax(codeword[self._M * i : self._M * (i + 1)])
                for j in range(log2m):
                    outp[log2m * (i + 1) - j - 1] = decimal % 2
                    decimal = decimal >> 1
            return outp

    def _decode_beta_llr(self, beta):  # создание битовой матрицы для каждой беты очень тяжелая история
        beta = beta / beta[:self._M].sum()

        log2m = int(np.round(np.log2(self._M)))
        out_probs = np.zeros(log2m * self._L, dtype=float)
        bit_representation_of_decimals_matrix = np.zeros((self._M, log2m), dtype=float)

        # constructig bit representation of every decimal from 0 to M - 1
        for target_decimal in range(self._M):
            decimal = target_decimal
            for j in range(log2m):
                bit_representation_of_decimals_matrix[target_decimal, log2m - j - 1] = decimal % 2
                decimal = decimal >> 1

        for i in range(self._L):
            out_probs[log2m * i : log2m * (i + 1)] = beta[i * self._M : self._M * (1 + i)] @ bit_representation_of_decimals_matrix

        llr = np.clip(np.log(out_probs / (1 - out_probs)), -50, 50)
        return llr

    def make_hard_decision_for_llr(self, llr):
        bits = (llr >= 0).astype(int)
        return bits


class LDPC:
    def __init__(self, n_code, d_v, d_c, systematic=False, sparse=True, seed=0):
        H, G = pyldpc.make_ldpc(n_code=n_code, d_v=d_v, d_c=d_c, systematic=systematic, sparse=sparse, seed=seed)
        self._seed = seed
        self._H = H
        self._G = G
        self.input_length = G.shape[1]  # getter?
        self.output_length = n_code  # getter?
        print(f'Created LDPC with RATE={G.shape[1]/n_code}')

    def code(self, codewords: np.ndarray) -> np.ndarray:
        if codewords.ndim == 1:  # if input is vector
            codewords = codewords.reshape(1, -1)
        elif codewords.ndim > 2:
            assert False, f"input can be either vector or matrix"

        encoded_words = np.zeros((codewords.shape[0], self.output_length), dtype=float)
        for i, cw in enumerate(codewords):
            encoded_words[i] = pyldpc.utils.binaryproduct(self._G, cw)

        return encoded_words

    def decode(self, codewords: np.ndarray, iterat: int = 20, return_llr=False, input_is_llr=False, snr=None) -> np.ndarray:
        denoised_messages =  pyldpc.decode(H=self._H, y=codewords.T, snr=snr, maxiter=iterat, return_llr=return_llr, input_is_llr=input_is_llr).T
        output_codewords = np.zeros((codewords.shape[0], self.input_length))
        if len(denoised_messages.shape) > 1:
            for i, denoised_message in enumerate(denoised_messages):
                output_codewords[i] = pyldpc.get_message(self._G, denoised_message)
        else:
            output_codewords[0] = pyldpc.get_message(self._G, denoised_messages)
        return output_codewords


class LDPC_SPARC:
    def __init__(self, outer_coder: LDPC, inner_coder: Sparc, inner_iterat=50, outer_iterat=20, **kwargs):
        self.inner_coder = inner_coder
        self.outer_coder = outer_coder
        self.output_length = inner_coder.output_length  # getter?
        self.input_length = outer_coder.input_length  # getter?
        self.inner_iterat = inner_iterat
        self.outer_iterat = outer_iterat
        self._P = inner_coder._P

    def code(self, codewords: np.ndarray) -> np.ndarray:
        outer_codewords = self.outer_coder.code(codewords).astype(int)
        inner_codewords = self.inner_coder.code(outer_codewords)
        return inner_codewords

    def decode(self, codewords: np.ndarray, **kwargs):  # добавить итеративный декодинг
        inner_decoded_codewords = self.inner_coder.decode(codewords, iterat=self.inner_iterat, return_llr=True)
        outer_decoded_codewords = self.outer_coder.decode(-inner_decoded_codewords, iterat=self.outer_iterat, return_llr=False, input_is_llr=True)
        return outer_decoded_codewords


class Simulation:
    _coder = None

    def __init__(self, coder: Sparc, save_path = './results/'):
        self._coder = coder
        self.save_path = save_path
        self.statistics = {'ber': {}, 'fer': {}}

    def run(self, snr_array: np.ndarray, n_samples: int = 100, save_to_file=False, seed: int = 0):
        state = np.random.get_state()
        np.random.seed(seed)

        save_to_file = False  # tmp turned-off saving to file

        if save_to_file:  # оберни в функию все вызовы с использованием файла
            file_name = f"{self._coder}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv"
            with open(self.save_path + file_name, mode='w') as f:
                f.write(f'SNR(dB),input_codeword,output_codeword\n')
                print(f'Created file {os.path.realpath(self.save_path + file_name)}')
                print('Simulation starts')

        for snr in snr_array:
            var = self._coder._P * 10 ** (-snr / 10)
            input_codewords = np.random.binomial(1, 0.5, (n_samples, self._coder.input_length))
            codeword_before_channel = self._coder.code(input_codewords)
            codeword_after_channel = codeword_before_channel + np.random.normal(0, var**0.5, (n_samples, self._coder.output_length))
            output_codeword = self._coder.decode(codeword_after_channel)

            # calc statitic (подумай про то, чтобы обновлять существующее значение)
            self.statistics['ber'].update({snr: ber(input_codewords, output_codeword).mean()})
            self.statistics['fer'].update({snr: fer(input_codewords, output_codeword).mean()})

            if save_to_file:
                self._save_to_file(10 * np.log10(self._coder._P / var), input_codewords, output_codeword, path = self.save_path + file_name)

        np.random.set_state(state)
        print('Simulation finished')
        if save_to_file:
            return os.path.realpath(self.save_path + file_name)

    def _save_to_file(self, snr, input_codewords, decoded_codewords, path):
        with open(path, mode='a') as f:
            for inp_cw, out_cd in zip(input_codewords, decoded_codewords):
                f.write(f'{snr:.5f},{bitsarray2string(inp_cw)},{bitsarray2string(out_cd)}\n')
        print(f'saved for SNR {snr} dB')

    @staticmethod
    def load_results(path):
        converters = {  # convert string value to 
            0: lambda x: float(x),
            1: lambda x: [int(ch) - 48 for ch in x],  # in the following function int('0') == 48 and
            2: lambda x: [int(ch) - 48 for ch in x],  #   int('1') == 49, so we need to substract 48
        }
        snr = np.genfromtxt(path, delimiter=',', usecols=[0], skip_header=True, converters=converters)
        inp = np.genfromtxt(path, delimiter=',', usecols=[1], skip_header=True, converters=converters)
        out = np.genfromtxt(path, delimiter=',', usecols=[2], skip_header=True, converters=converters)
        return snr, inp, out


def _is_power_of_two(x):
    return (x != 0) and (x & (x - 1) == 0)


def bitsarray2string(bitsarray):
    return ''.join(map(str, bitsarray))


def ber(input_message: np.ndarray, output_message: np.ndarray) -> int:
    assert input_message.shape == output_message.shape, f"Messages must have equal lengths"

    return (input_message != output_message).mean(axis=-1)


def fer(input_message: np.ndarray, output_message: np.ndarray) -> int:
    assert input_message.shape == output_message.shape, f"Messages must have equal lengths"

    return (input_message != output_message).any(axis=-1).astype(int)


def groupby(groupby_column, groupby_function, *other_columns):
    unique_vals = np.unique(groupby_column)
    return_val = [np.zeros(len(unique_vals)) for _ in range(len(other_columns))]
    for j, val in enumerate(unique_vals):
        for i in range(len(other_columns)):
            return_val[i][j] = groupby_function(other_columns[i][groupby_column == val])
    return unique_vals, *return_val
