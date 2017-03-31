from __future__ import division
import numpy as np

import math


class Boltzmann:
    """docstring for ClassName"""
    def __init__(self):

        self.runs = 60
        self.N = 5
        self.threshold = 0.05

        self.W = np.random.random_integers(-1e5, 1e5, size=(self.N, self.N)) \
            / 1e5
        self.W = (self.W + self.W.T)

        np.fill_diagonal(self.W, 0)

        self.bias = np.zeros(self.N)

        # Initial configuration
        self.config = np.random.choice([-1, 1], self.N)
        self.visiting = xrange(self.N)

        self.all_configs = np.zeros([self.runs + 1, self.N])
        self.all_configs[0] = self.config

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dynamics(self, i):
        self.weightsOnConnections = self.W[i, :] * self.config
        self.total_input = self.bias[i] + np.sum(self.weightsOnConnections)
        probability_turn_on = self.sigmoid(self.total_input)
        return probability_turn_on

    def sweep(self, sweep_num):
        for i in self.visiting:
            p_turn_on = self.dynamics(i)
            p_turn_off = 1 - p_turn_on
            self.config[i] = np.random.choice([-1, 1], p=[p_turn_off, p_turn_on])
            self.all_configs[sweep_num] = self.config

    # Visiting order
    def refreshVisitingOrder(self):
        self.visiting = np.roll(self.visiting, 2)

    def empiricalMean_bySweep(self, sweep_num=False):
        if sweep_num:
            return np.sum(self.all_configs, axis=0) / sweep_num
        else:
            return np.sum(self.all_configs, axis=0) / len(self.all_configs)

    def printSTUFF(self, sweep_num):
        print "empirical mean", self.empiricalMean_bySweep(sweep_num)
