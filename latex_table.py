import pandas as pd 
import numpy as np 
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path",help = "Folder Path to environment")
args = parser.parse_args()

# list = ["Reinforce","A2C","TD","konda"]

givePath = args.path

print(givePath)
print("-------------------------\n")

if givePath[-1] != "/":
    givePath+="/"


Data_Table_final = []
Data_Table_optimal = []
list_variant=[]
list_final_max_mean=[]
list_final_max_std=[]
list_final_sampling_mean=[]
list_final_sampling_std=[]
list_optimal_max_mean=[]
list_optimal_max_std=[]
list_optimal_sampling_mean=[]
list_optimal_sampling_std=[]

for types in os.listdir(givePath):
    branch1 = givePath+types
    for data in os.listdir(branch1):
        branch2 = branch1 + "/" + data

        try:
            meanstd = np.load(branch2+"/" +"final/meanstd.npy")
            meanstd = np.around(meanstd, decimals=2)
        except IOError:
            continue
        try:
            meanstd_optimal = np.load(branch2+"/" +"optimal/meanstd.npy")
            meanstd_optimal = np.around(meanstd_optimal, decimals=2)
        except IOError:
            continue
        # print()
        ##Variant name

        list_variant.append(data)

        list_final_max_mean.append(meanstd[0,0])
        list_final_max_std.append(meanstd[0,1])
        list_final_sampling_mean.append(meanstd[1,0])
        list_final_sampling_std.append(meanstd[1,1])

        
        ###
        list_optimal_max_mean.append(meanstd_optimal[0,0])
        list_optimal_max_std.append(meanstd_optimal[0,1])
        list_optimal_sampling_mean.append(meanstd_optimal[1,0])
        list_optimal_sampling_std.append(meanstd_optimal[1,1])

print("Final table")
Data_Table_final = pd.DataFrame({'Variant': list_variant,
                   'max_mean': list_final_max_mean,
                   'max_std': list_final_max_std,
                   'sampling_mean': list_final_sampling_mean,
                   'sampling_std': list_final_sampling_std
                   })
print(Data_Table_final.to_latex(index=False))
print("------------------")
print("Optimal table")
Data_Table_optimal = pd.DataFrame({'Variant': list_variant,
                   'max_mean': list_optimal_max_mean,
                   'max_std': list_optimal_max_std,
                   'sampling_mean': list_optimal_sampling_mean,
                   'sampling_std': list_optimal_sampling_std
                   })
print(Data_Table_optimal.to_latex(index=False))