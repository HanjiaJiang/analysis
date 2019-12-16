import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # get input and output names
    inputs = sys.argv[1:-1]
    output = sys.argv[-1]
    print(inputs)
    print(output)

    for item in inputs:
        levels_str = item.split('/')[1]
        levels_list = np.array(levels_str.split('_')).astype(int)
        print(levels_list)

    np.save(output, [])
