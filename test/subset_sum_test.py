from prcwatermark.subset_sum import calc_subset_sum

if __name__ == "__main__":
   weights = [0.5, 1.5, 2.5]
   T = 3
   decimals = 2
   subset = calc_subset_sum(weights, T, decimals)
   assert(subset == [1, 0, 1])




