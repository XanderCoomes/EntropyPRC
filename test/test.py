from prcwatermark.prc import PRC
import numpy as np


def sparsity_fn(codeword_len):
    return int(np.log2(codeword_len))

def secret_len_fn(codeword_len):
    return pow(sparsity_fn(codeword_len), 2)

def num_parity_checks_fn(codeword_len):
    return int(0.99 * codeword_len)

def test_key_gen(): 
    codeword_len = 300
    prc_medium = PRC(codeword_len, sparsity_fn, secret_len_fn, num_parity_checks_fn)

def test_encode(): 
    pass


def test_decode(): 
    pass


if __name__ == "__main__":
    test_key_gen()








