#!/usr/bin/env python
""" Run the artificial neural network.
"""
import Boltzmann
# import network as nw
# import plotting
# import mlp_network


def run_Boltzmann():
    """ A function for running the network without
    running the full experiments.
    """
    GBM = Boltzmann.Boltzmann()
    GBM.printSTUFF()
    GBM.sweep()
    GBM.printSTUFF() 

    # net = nw.Network(32)
    # net.build_model()
    # net.training()
    # net.predictions()
    # net.gradient_terminal_weights()

    print "Network() finished"


if __name__ == '__main__':

    print 'Started Program.'
    run_Boltzmann()
    print "End of program"
