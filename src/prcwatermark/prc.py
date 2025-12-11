import numpy as np
import galois
from scipy.stats import binom

GF = galois.GF(2)
np.set_printoptions(threshold=np.inf, linewidth = 400)


class PRC:
    def __init__(self, codeword_len, sparsity_fn): 
        self.codeword_len = codeword_len
        self.sparsity = sparsity_fn(codeword_len)
        self.secret_len = int(np.log2(codeword_len) ** 2)
        self.num_parity_checks = int(0.99 * self.codeword_len)
        self.gen_key()

    def gen_key(self):
        self.sample_parity_check_matrix()
        self.sample_generator_matrix()
        self.one_time_pad = GF.Random(self.codeword_len)
        self.permutation = np.random.permutation(self.codeword_len)


    def sample_parity_check_matrix(self): 
        self.parity_check_matrix = np.zeros((self.num_parity_checks, self.codeword_len), dtype = int)
        for i in range (self.num_parity_checks):
            sparse_row = np.zeros(self.codeword_len, dtype=int)
            ones_indices = np.random.choice(self.codeword_len, size = self.sparsity, replace = False)
            sparse_row[ones_indices] = 1
            self.parity_check_matrix[i] = sparse_row 
        self.parity_check_matrix = GF(self.parity_check_matrix)

    def sample_generator_matrix(self): 
        null_space = self.parity_check_matrix.null_space()
        null_space = null_space.T 
        self.generator_matrix = np.zeros((self.codeword_len, self.secret_len), dtype = int)
        for i in range (self.secret_len): 
            null_space_column = null_space @ GF.Random(null_space.shape[1])
            self.generator_matrix[:, i] = null_space_column
            pass
        self.generator_matrix = GF(self.generator_matrix)

    def encode(self, noise_rate):
        secret = GF.Random(self.secret_len)
        error = GF(np.random.binomial(1, noise_rate, self.codeword_len))
        codeword = (self.generator_matrix @ secret + self.one_time_pad + error)
        permuted_codeword = codeword[self.permutation]
        return permuted_codeword
    
    def calc_failed_parity_checks(self, bit_str): 
        inv_perm = np.empty_like(self.permutation)
        inv_perm[self.permutation] = np.arange(len(self.permutation))   

        permuted_bit_str = GF(bit_str[inv_perm]) + self.one_time_pad
        failed_parity_checks = np.sum ((self.parity_check_matrix @ permuted_bit_str) == 1)
        return failed_parity_checks

    def prob_binom_odd(self, n, p):
        return (1 - (1 - 2 * p) ** n) / 2

    def prob_codeword(self, bit_str, approx_noise_rate):
        failed_checks = self.calc_failed_parity_checks(bit_str)

        print("Estimated Noise Rate: ", approx_noise_rate)
        print("Parity Checks: ", self.num_parity_checks)
        print("Failed Parity Checks: ", failed_checks)

        fail_prob_dry = 0.5
        fail_prob_water = self.prob_binom_odd(self.sparsity, approx_noise_rate)

        prob_failed_checks_given_dry   = binom.pmf(failed_checks, self.num_parity_checks, fail_prob_dry)
        prob_failed_checks_given_water = binom.pmf(failed_checks, self.num_parity_checks, fail_prob_water)

        prob_codeword = prob_failed_checks_given_water / (prob_failed_checks_given_dry + prob_failed_checks_given_water)

        return prob_codeword
    
    def calc_threshold(self, false_positive_rate): 
        fail_prob_dry = 0.5
        threshold = self.num_parity_checks 
        while(True):
            prob = binom.cdf(threshold, self.num_parity_checks, fail_prob_dry)
            if(prob <= false_positive_rate):
                break
            else:
                threshold -= 1
        return threshold 
    
    def threshold_decode(self, bit_str, false_postive_rate): 
        threshold = self.calc_threshold(false_postive_rate)
        print("Decoding Threshold  :", threshold)
        failed_checks = self.calc_failed_parity_checks(bit_str)
        print("Failed Parity Checks:", failed_checks)
        if(failed_checks > threshold):
            return False
        return True




        



        


    
   

