import numpy as np 
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("path",help = "Folder Path to stats")
    args = parser.parse_args()

    mean = np.load(args.path)

    print("max:",mean[0])
    print("sampling:",mean[1])