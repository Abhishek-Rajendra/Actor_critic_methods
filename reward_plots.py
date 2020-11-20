import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import numpy as np

def plotRewards(episode_rewards, label, alpha, gamma, plot_path):

    plt.figure()
    plt.suptitle(variant+" - "+environment)
    plt.title(r"$\alpha $ = "+alpha+", $\gamma$ = "+gamma)
    plt.plot(range(len(episode_rewards)),episode_rewards, '.-',label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Total reward')
    plt.legend()
    plt.savefig(plot_path+"/finalReward.png")

    plt.figure()
    plt.suptitle(variant+" - "+environment)
    z1=pd.Series(episode_rewards).rolling(50).mean()
    plt.title(r"$\alpha $ = "+alpha+", $\gamma$ = "+gamma+ ",  Best average reward: "+ str(np.max(z1)))
    plt.plot(z1,label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Rewards over past 50 episodes')
    plt.legend()

    plt.savefig(plot_path+"/finalcumulative.png")
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path",help = "Folder Path to reward file")
    args = parser.parse_args()

    path = args.path
    
    directory = os.path.dirname(path)
    #Model data
    modeldata = os.path.split(directory)[1]
    parentdir = os.path.dirname(directory)
    #Variant
    variant = os.path.split(parentdir)[1]
    #Environment
    environment = os.path.dirname(parentdir)

    temp = modeldata.rsplit("_")
    if(variant == "Reinforce"):
        label = variant+"-"+temp[0]
        gamma = temp[1]
        alpha = temp[2]
    else:
        label = variant
        gamma = temp[7]
        alpha = temp[1]+", "+temp[3] 

    score_history = np.load(path)
    plotRewards(score_history,label, alpha, gamma, directory)