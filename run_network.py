#!/usr/bin/env python
""" Run the artificial neural network.
"""
from keras.datasets import mnist
import Boltzmann
import numpy as np
import plotting
import train_rbm as rbm


def run_Boltzmann():
    """ Runs the generic Boltzmann machine
    """
    GBM = Boltzmann.Boltzmann()
    plot = plotting.Plotting(GBM)

    W = np.random.random_integers(-1e5, 1e5, size=(GBM.N, GBM.N)) \
        / 1e5
    W = (W + W.T)

    np.fill_diagonal(W, 0)

    W_2 = W / 10

    weights = [W, W_2]

    energy = []

    for weight in weights:
        for i in xrange(GBM.runs):
            GBM.sweep(i, weight)
            GBM.empiricalMean_bySweep(i)
            GBM.refreshVisitingOrder()
            if i > 30:
                GBM.sweep(i, weight)
                GBM.empiricalMean_bySweep(i)
                GBM.refreshVisitingOrder()

        GBM.differences_mean()
        GBM.Total_energy(weight)
        energy.append(GBM.Energies)
    print "len(energy)", len(energy[0])

    plot.plot_energies(energy)

    print "Boltzmann() finished"


def load_mnist():
    (x_train, x_train_lab), \
        (x_test, x_test_lab) = mnist.load_data()
    return x_train, x_train_lab, x_test, x_test_lab


def run_rbm():
    """ runs the Restricted Boltzmann Machine
    """
    x_train, x_train_lab, x_test, x_test_lab = load_mnist()

    restricted = rbm.rbm_matlab(x_train)
    restricted.images_to_vectors()
    restricted.train_rbm()




if __name__ == '__main__':

    print 'Started Program.'
    # run_Boltzmann()
    run_rbm()
    print "End of program"
