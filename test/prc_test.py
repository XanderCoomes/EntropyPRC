from prcwatermark.prc import PRC
import numpy as np

def sparsity_fn(codeword_len):
        return int(np.log2(codeword_len))

class PRCTester:
    def __init__(self, prc): 
        self.prc = prc
    def test_key_gen(self): 
        assert(self.prc.parity_check_matrix @ self.prc.generator_matrix == 0).all()
    def test_encode_decode(self, noise_rate, false_positive_rate):
        codeword = prc.encode(noise_rate) 
        is_codeword = prc.decode(codeword, false_positive_rate)
        if(noise_rate < 0.5):
            assert is_codeword == True
        else:
            assert is_codeword == False

if __name__ == "__main__":
    codeword_len = 300
    prc = PRC(codeword_len, sparsity_fn)
    tester = PRCTester(prc)
    tester.test_key_gen()

    false_positive_rate = 1e-6

    noise_rate = 0.0
    tester.test_encode_decode(noise_rate, false_positive_rate)

    noise_rate = 0.5
    tester.test_encode_decode(noise_rate, false_positive_rate)

    prc.save("./keys")

    








