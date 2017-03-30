from __future__ import division
import numpy as np

import math


class Boltzmann:
    """docstring for ClassName"""
    def __init__(self):

        self.N = 5
        self.W = np.random.random_integers(-1e5, 1e5, size=(self.N, self.N)) \
            / 1e5
        self.W = (self.W + self.W.T)

        np.fill_diagonal(self.W, 0)

        self.bias = np.zeros(self.N)

        # Initial configuration
        self.config = np.random.choice([-1, 1], self.N)
        self.visiting = xrange(self.N)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dynamics(self, i):
        self.weightsOnConnections = self.W[i, :] * self.config
        self.total_input = self.bias[i] + np.sum(self.weightsOnConnections)
        probability_turn_on = self.sigmoid(self.total_input)
        return probability_turn_on

    def sweep(self):
        for i in self.visiting:
            p_turn_on = self.dynamics(i)
            p_turn_off = 1 - p_turn_on
            self.config[i] = np.random.choice([-1, 1], p=[p_turn_off, p_turn_on])

    # Visiting order
    def refreshVisitingOrder(self):
        self.self.visiting = np.roll(self.visiting, 2)

    def printSTUFF(self):
        print "printing weights", self.W
        print "Printing self.config", self.config
