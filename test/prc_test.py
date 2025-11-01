from prcwatermark.prc import PRC
import numpy as np

def sparsity_fn(codeword_len):
        return int(np.log10(codeword_len))

class PRCTester:
    def __init__(self, prc): 
        self.prc = prc
    def test_key_gen(self): 
        assert(self.prc.parity_check_matrix @ self.prc.generator_matrix == 0).all()

    def test_encode_decode(self, noise_rate): 
        codeword = prc.encode(noise_rate)
        prob_codeword = prc.prob_codeword(codeword, noise_rate)
        print(f"Probability Codeword: {prob_codeword:.2%}")

if __name__ == "__main__":
    codeword_len = 300
    prc = PRC(codeword_len, sparsity_fn)
    tester = PRCTester(prc)
    tester.test_key_gen()

    print("Test 1:")
    noise_rate = 0.0
    tester.test_encode_decode(noise_rate)
    print()

    print("Test 2:")
    noise_rate = 0.15
    tester.test_encode_decode(noise_rate)
    print()

    print("Test 3:")
    noise_rate = 0.41
    tester.test_encode_decode(noise_rate)
    print()


  

 



    








