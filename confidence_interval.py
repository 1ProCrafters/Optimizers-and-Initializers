import numpy as np
import scipy.stats

def confidence_interval(array, confidence, goal):
    if goal == 'min':
        data = array
    elif goal == 'max':
        data = [-x for x in array]
    elif goal == 'average':
        data = array
    else:
        raise ValueError("Invalid goal. It should be 'min' or 'max' or 'average'.")
    n = len(data)
    m = np.mean(data)
    stderr = scipy.stats.sem(data)
    interval = stderr * scipy.stats.t._ppf((1 + confidence) / 2., n-1)
    lower = m - interval
    upper = m + interval
    if goal == 'max':
        lower, upper = -upper, -lower
    return lower, upper


def main():
    arr = input("Enter the array: ")
    goal = input("Enter the goal (min, max, average): ")
    level = input("Enter the confidence level: ")
    print(confidence_interval(arr, level, goal))

if __name__ == "__main__":
    main()
