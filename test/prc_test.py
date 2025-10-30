from prcwatermark.prc import PRC
import numpy as np

def sparsity_fn(codeword_len):
        return int(np.log10(codeword_len))

class PRCTester:
    def __init__(self, prc): 
        self.prc = prc
    def test_key_gen(self): 
        assert(self.prc.parity_check_matrix @ self.prc.generator_matrix == 0).all()
    def test_encode_decode(self, noise_rate, false_positive_rate):
        codeword = prc.encode(noise_rate) 
        is_codeword = prc.is_codeword(codeword, false_positive_rate)
        if(noise_rate < 0.5):
            assert is_codeword == True
        else:
            assert is_codeword == False

    def test_probabalistic_decode(self, noise_rate): 
        codeword = prc.encode(noise_rate)
        prob_codeword = prc.prob_codeword(codeword, noise_rate)
        print(f"Probability Codeword: {prob_codeword:.2%}")

if __name__ == "__main__":
    codeword_len = 300
    prc = PRC(codeword_len, sparsity_fn)
    tester = PRCTester(prc)
    tester.test_key_gen()

    false_positive_rate = 1e-6

    print("Test 1:")
    noise_rate = 0.0
    tester.test_encode_decode(noise_rate, false_positive_rate)
    print()

    print("Test 2:")
    noise_rate = 0.5
    tester.test_encode_decode(noise_rate, false_positive_rate)
    print()

    print("Test 3:")
    noise_rate = 0.30
    tester.test_probabalistic_decode(noise_rate)
    print()

 



    








