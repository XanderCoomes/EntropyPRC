# Design.md


## PRC
 - GenKeys(codeword_len, sparsity, secret_length, parity_checks) --> parity_check_matrix, generator_matrix, one_time_pad, permutation
 - Encode(generator_matrix, one_time_pad, permutation, noise_rate) --> codeword
 - Decode(parity_check_matrix, bit_string, one_time_pad, permutation, false_positive_rate) --> is_codeword


 ##