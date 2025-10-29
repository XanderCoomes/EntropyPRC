import numpy as np
import galois


GF = galois.GF(2)

class PRC:
    def __init__(self, codeword_len, sparsity_fn): 
        self.codeword_len = codeword_len
        self.sparsity = sparsity_fn(codeword_len)
        self.secret_len = pow(self.sparsity, 2)
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
    
    def calc_threshold(self, false_positive_rate):
        return (0.5 - (self.num_parity_checks ** (-0.25))) * self.num_parity_checks; 

    def decode(self, bit_str, false_positive_rate):
        threshold = self.calc_threshold(false_positive_rate)
        print("num parity checks: ", self.num_parity_checks)
        print("threshold: ", threshold)
        inv_perm = np.empty_like(self.permutation)
        inv_perm[self.permutation] = np.arange(len(self.permutation))   

        permuted_bit_str = GF(bit_str[inv_perm]) + self.one_time_pad
        failed_parity_checks = np.sum ((self.parity_check_matrix @ permuted_bit_str) == 1)
        print("failed parity checks: ", failed_parity_checks)


        if(failed_parity_checks < threshold):
            return True;
        else:
            return False; 
    
   

