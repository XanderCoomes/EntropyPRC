from prcwatermark.subset_sum import calc_subset_sum

if __name__ == "__main__":
   weights = [7, 1.1, 2, 2.2, 3, 10]
   T = 4.0
   increment = 1.0
   total_weight = calc_subset_sum(weights, T, increment)
   assert(total_weight == 3)




