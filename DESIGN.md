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
Sampler Class: 
Fields
 - tokenizer
 - model
 - temperature
 - top_p
 - no_repetition_penalty
 
Methods
 - calc_probs(tokens) --> probs
   - calc_logits(tokens) --> logits
   - apply_repetition_penalty(logits, tokens, repetition_penalty) --> logits
   - apply_temperature(logits, temperature) --> logits
   - softmax(logits) --> probs
   - apply_top_p(probs, p) --> probs
 - calc_entropy(probs) --> entropy

## Watermarking Functions
WaterLLM Class: 
Fields
 - sampler
 - prc
 - majority_encoder
 - entropy_threshold

Methods
 - bias_probs(probs, hash_func, bit) --> probs
 - token_hash(tokens, hash_func) --> bit_string
 - entropy_threshold() --> alpha
 - determine_entropy(tokens)
 - gen_response(prompt) --> tokens 
 - detect_water(output, water_parmeters, sampling_parameters) --> prob_watermarked







