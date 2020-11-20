import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("path",help = "Folder Path to reward file")
args = parser.parse_args()

reward = np.load(args.path)

print("initial  final best")
print("{:.2f} & {:.2f} & {:.2f}".format(reward[0],reward[-1],max(reward)))