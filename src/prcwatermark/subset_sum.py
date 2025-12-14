import numpy as np

def calc_subset_sum(weights, T, decimals):
    increment = 10 ** -decimals 
    num_t_vals = int(T * (1 / increment) + 1)
    include = [[0] * num_t_vals for _ in range(len(weights))]
    exclude = [[0] * num_t_vals for _ in range(len(weights))]
    weights = np.round(weights, decimals)

    for i in range(len(weights)): 
        for j in range(1, num_t_vals): 
            t = increment * j
            if(i == 0): 
                if(weights[i] <= t):
                    include[i][j] = weights[i]
            else: 
                if(weights[i] <= t): 
                    weight_idx = int(j - weights[i]/increment)
                    include[i][j] = max(include[i - 1][weight_idx], exclude[i - 1][weight_idx]) + weights[i]
                else: 
                    include[i][j] = 0
                exclude[i][j] = max(include[i - 1][j], exclude[i - 1][j])

    t = T
    bitmask = [0] * len(weights)
    for i in range(len(weights) - 1, - 1, -1): 
        idx = int(t / increment)
        if(include[i][idx] > exclude[i][idx]):
            bitmask[i] = 1
            t = T - weights[i]
    return bitmask
    
    



