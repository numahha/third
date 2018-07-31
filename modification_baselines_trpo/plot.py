import numpy as np
import matplotlib.pyplot as plt
import argparse
#import os


def learning_curve(input_filename="progress.csv",
                   output_filename="learning_curve.png"):
    with open(input_filename, 'r') as file:
        lines = list(file)
    header = lines[0].strip().split(',')
    data = np.loadtxt(lines[1:], delimiter=',')

    if 2==data.ndim:
        plt.plot(data[:,header.index("TimestepsSoFar")],data[:,header.index("EpRewMean")])
        plt.xlabel('Timesteps')
        plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="x")
        plt.ylabel('Average Return')
        plt.savefig(output_filename)
        plt.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    learning_curve(input_filename=args.input, output_filename=args.output)
