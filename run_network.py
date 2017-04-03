#!/usr/bin/env python
""" Run the artificial neural network.
"""
from keras.datasets import mnist
import Boltzmann
import numpy as np
import plotting
import train_rbm as rbm
import analysis
import network

from datetime import datetime


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
        print "New weight matrix"
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
    GBM = Boltzmann.Boltzmann()
    plot = plotting.Plotting(GBM)

    x_train, x_train_lab, x_test, x_test_lab = load_mnist()
    hidden_size = [32, 64]
    Weights_trained = []
    bias_trained = []
    Weights_tested = []

    explore = analysis.Analysis()

    for h in hidden_size:
        restricted = rbm.rbm_matlab(h)
        restricted.train_rbm(x_train)
        Weights_trained.append(restricted.W)
        hidden, reconstruction = restricted.compute_hidden()
        print "PCA x_train"
        explore.principal_component(hidden)

        bias_trained.append(restricted.bias_upW)
        restricted.train_rbm(x_test)
        Weights_tested.append(restricted.W)
        hidden, reconstruction = restricted.compute_hidden()
        print "PCA x_test"
        explore.principal_component(hidden)

    # plot.plot_pca(filepath='./Images/restrictedBM/', x_dimreduced=x_dimreduced)

    rmse_train_test = [restricted.rmse(Weights_trained[i], Weights_tested[i])
                       for i in range(len(Weights_tested))]
    print "rmse_train_test", rmse_train_test
    print 'Weights_trained[0].shape:\n', Weights_trained[0].shape
    return Weights_trained, bias_trained  # Weights_tested


def run_network():
    # using the weights corresponding to hidden_size 32
    """W_tested, """
    W_trained, bias_trained = run_rbm()
    biases = np.squeeze(bias_trained[0].T)
    net = network.Network()
    net.images_to_vectors()
    net.build_model([W_trained[0], biases])
    net.train_model()
    net.evaluate_model()


if __name__ == '__main__':
    start = datetime.now()
    print 'Started Program at ', start
    # run_Boltzmann()
    # run_rbm()
    run_network()
    end = datetime.now()
    print "End of program. Runtime: ", end - start
