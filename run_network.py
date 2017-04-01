#!/usr/bin/env python
""" Run the artificial neural network.
"""
import Boltzmann
import numpy as np
# import network as nw
import plotting
# import mlp_network


def run_Boltzmann():
    """ A function for running the network without
    running the full experiments.
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

    print "energy[0]", energy[0]

    plot.plot_energies(energy)

    print "Boltzmann() finished"


if __name__ == '__main__':

    print 'Started Program.'
    run_Boltzmann()
    print "End of program"
