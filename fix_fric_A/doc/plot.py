import numpy as np
import matplotlib.pyplot as plt
import argparse
#import os


def learning_curve_data(input_filename="progress.csv"):
    with open(input_filename, 'r') as file:
        lines = list(file)
    header = lines[0].strip().split(',')
    data = np.loadtxt(lines[1:], delimiter=',')

    return data[:,header.index("EpisodesSoFar")], data[:,header.index("EpRewMean")]

i=0.1
while i<3.01:
    for s in range(0,6):
        filename="fricfix"+("%02.1f" % i)+"_seed"+("%d" % s)+"/progress.csv"
        x,y=learning_curve_data(input_filename=filename)
        plt.plot(x,y,label=("seed %d" % s))
    plt.ylabel('Reward')
    plt.xlabel('Training Episode')
    plt.legend()
    filename="curve_fricfix"+("%02.1f" % i)+".eps"
    plt.title(filename)
    plt.savefig(filename)
    plt.close()
    i+=0.1
