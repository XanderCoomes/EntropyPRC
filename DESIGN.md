# Design.md
## PRC Functions
PRC Class: 
Fields
 - codeword_len
 - sparsity
 - secret_len
 - num_parity_checks
 - parity_check_matrix
 - generator_matrix
 - one_time_pad
 - permutation
 
Methods
 - key_gen()
 - sample_parity_check_matrix()
 - sample_generator_matrix()
 - encode(noise_rate)
 - decode(bit_str, false_positive_rate)

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






