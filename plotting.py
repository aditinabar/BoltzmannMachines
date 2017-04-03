import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import numpy as np
from sklearn.decomposition import PCA

import analysis
import train_rbm as rbm
import network as Network

import os


class Plotting(object):
    """Documentation for Plotting
    Plotting data for the neural network.
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=no-self-use

    def __init__(self, network):
        self.__make_experiment_dir()
        # self.history = self.network.history

        # Call Analysis to plot PCA
        self.explore = analysis.Analysis()

        self.net = Network.Network()

        # Configuration Variables
        self.opacity = 0.6

        sns.set(rc={"figure.figsize": (6, 6)})
        np.random.seed(sum(map(ord, "palettes")))

    def __make_experiment_dir(self):
        """ Make image and experiment directories.
        """
        # Make the image directory
        self.image_path = './Images/'
        self.make_dir(self.image_path)

        #  Make experiment directories
        self.GBM = self.image_path + 'genericBM/'
        self.make_dir(self.GBM)
        self.RBM = self.image_path + 'restrictedBM/'
        self.make_dir(self.RBM)

    def make_dir(self, filepath):
        """ Checks that there is a directory for the desired
        location and if it does not exist make one.
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    def plot_energies(self, energies):
        for i in range(len(energies)):
            plt.hist(energies[i], bins=45)
            plt.title("Energies for weight matrix W_ij, part" + str(i + 1))
            plt.xlabel("Energy of configuration")
            plt.ylabel("Step number")
            path = self.GBM + 'energies' + str(i + 1) + '.png'
            plt.draw()
            plt.savefig(path)
            plt.clf()

# To be updated

    def plot_pca(self, filepath='./Images/restrictedBM/', x_dimreduced=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #  target_names = [str(label) for label in xrange(10)]
        #  hidden_cloud = self.network.get_hidden()
        colors = sns.color_palette("Set2", 10)
        for color, i in zip(colors, xrange(10)):
            ax.scatter(x_dimreduced[self.net.x_train_lab == i, 0],
                       x_dimreduced[self.net.x_train_lab == i, 1],
                       x_dimreduced[self.net.x_train_lab == i, 2],
                       c=color)
        path = filepath + 'pca.png'
        plt.draw()
        plt.savefig(path)
        plt.clf()















