import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

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
        self.network = network
        self.history = self.network.history

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
            plt.hist(energies[i])
            plt.title("Energies for weight matrix W_ij, part" + str(i + 1))
            plt.xlabel("Step number")
            plt.ylabel("Energy of configuration")
            path = self.GBM + 'energies' + str(i + 1) + '.png'
            plt.draw()
            plt.savefig(path)
            plt.clf()

















