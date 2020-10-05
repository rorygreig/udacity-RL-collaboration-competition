import numpy as np
import matplotlib.pyplot as plt


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def plot_scores_with_average(scores, average_scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label="Scores")
    plt.plot(np.arange(len(average_scores)), average_scores, label="Average Scores")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
