# Design.md

## PRC Functions

PRC Class: 
 - gen_keys(codeword_len, sparsity, secret_len, num_parity_checks) --> parity_check_matrix, generator_matrix, one_time_pad, permutation
    - sample_parity_check_matrix(codeword_len, sparsity, num_parity_checks) --> parity_check_matrix
    - sample_generator_matrix(parity_check_matrix, secret_len) --> generator_matrix
    - sample_one_time_pad(codeword_len) --> one_time_pad
    - sample_permutation(codeword_len) --> permutation

 - encode(generator_matrix, one_time_pad, permutation, noise_rate) --> codeword
    - sample_random_secret(secret_len) --> random_secret

 - decode(parity_check_matrix, bit_string, one_time_pad, permutation, false_positive_rate) --> is_codeword
    - calc_parity_check_threshold(false_positive_rate, codeword_len, num_parity_checks) --> threshold

## Sampling Functions
 - logits(model, tokens) --> logits
 - no_repetition_penalty(logits, tokens, repetition_penalty) --> logits
 - no_repeat_n_gram (logits, tokens, n_gram_penalty) --> logits
 - temperature(logits, temperature) --> logits
 - probs(logits) --> probs
 - top_p(probs, p) --> probs
 - entropy(probs) --> entropy

## Watermarking Functions
 - bias_probs(probs, hash_func, bit) --> probs
 - token_hash(tokens, hash_func) --> bit_string
 - gen_output(prompt, num_tokens, water_parameters, sampling_parameters) --> tokens 
    - prompt_to_tokens(prompt) --> tokens
- detect_watermark(output, water_parmeters, sampling_parameters) --> is_watermarked






