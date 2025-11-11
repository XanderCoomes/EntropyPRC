from prcwatermark.prc import PRC
import numpy as np

def sparsity_fn(codeword_len):
    log_base = 4
    return int(np.log2(codeword_len) / np.log2(log_base))

class PRCTester:
    def __init__(self, prc): 
        self.prc = prc
    def test_key_gen(self): 
        assert(self.prc.parity_check_matrix @ self.prc.generator_matrix == 0).all()

    def test_encode_decode(self, noise_rate): 
        codeword = prc.encode(noise_rate)
        prob_codeword = prc.prob_codeword(codeword, noise_rate)
        print(f"Probability Codeword: {prob_codeword:.2%}")

    def test_threshold_decode(self, noise_rate, false_positive_rate):
         codeword = prc.encode(noise_rate)
         prob_codeword = prc.threshold_decode(codeword, false_positive_rate)
         print(f"Threshold Decode Result: {prob_codeword}")

if __name__ == "__main__":
    codeword_len = 250
    prc = PRC(codeword_len, sparsity_fn)
    tester = PRCTester(prc)
    tester.test_key_gen()

    noise_rate = 0.15
    tester.test_threshold_decode(noise_rate, false_positive_rate = 0.01)
    tester.test_threshold_decode(noise_rate, false_positive_rate = 0.02)
    tester.test_threshold_decode(noise_rate, false_positive_rate = 0.05)
    tester.test_threshold_decode(noise_rate, false_positive_rate = 0.10)






  

 



    








