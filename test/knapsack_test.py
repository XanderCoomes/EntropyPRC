from prcwatermark.knapsack import knapsack

if __name__ == "__main__":
   weights = [7, 1, 2, 2, 3, 10]
   T = 4.0
   increment = 1.0
   total_weight = knapsack(weights, T, increment)
   assert(total_weight == 3)