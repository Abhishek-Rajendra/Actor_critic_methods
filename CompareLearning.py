import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("environment",help="CartPole or Acrobot or LunarLander or MountainCar")
    args = parser.parse_args()

    lists = ["Reinforce","Qbaseline","A2C","KondaTD(1)"]    

    environment = args.environment
    colors=[['darkgreen', 'lime', 'palegreen'],\
         ['navy', 'cornflowerblue', 'lightsteelblue'], \
             ['firebrick', 'tomato', 'lightsalmon'], \
                 ['yellow','gold','khaki']]

    plt.figure()
    plt.title("Training - " + environment)
    for i,variant in enumerate(lists):
        path = environment + "/" + variant

        if os.path.exists(path):
            dir = os.listdir(path)


            if(variant == "Reinforce"):
                reward_path = path + "/" + dir[2] + "/score.npy"
                temp = dir[2].rsplit("_")
                print(dir[2])
                name = variant + "- baseline"
            else:
                reward_path = path + "/" + dir[0] + "/score.npy"
                temp = dir[0].rsplit("_")
                print(dir[0])
                name = temp[0]
            
            
            size = 10000
            rewards = np.load(reward_path)[:size]
            mean_rewards=pd.Series(rewards).rolling(50).mean()
        
            plt.plot(rewards,color=colors[i][2],alpha=0.6)
            plt.plot(mean_rewards,color=colors[i][0],label=name)

    plt.xlabel('Number of Episodes')
    plt.ylabel('Total Reward of an Episode')
    plt.legend()
    # plt.savefig(filename+"plot/RewardComparison0.01_0.png", bbox_inches='tight')
    plt.show()