from __future__ import division
import numpy as np

import math


class Boltzmann:
    """docstring for ClassName"""
    def __init__(self):

        self.runs = 300
        self.N = 30
        self.threshold = 0.01

        self.bias = np.random.randn(self.N)

        # Initial configuration
        self.config = np.random.choice([-1, 1], self.N)
        self.visiting = xrange(self.N)

        self.all_configs = np.zeros([self.runs + 1, self.N])
        self.all_configs[0] = self.config

        self.mean_Mat = np.zeros([self.runs, self.N])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dynamics(self, i, weight):
        self.weightsOnConnections = weight[i, :] * self.config
        self.total_input = self.bias[i] + np.sum(self.weightsOnConnections)
        probability_turn_on = self.sigmoid(self.total_input)
        return probability_turn_on

    def sweep(self, sweep_num, weight):
        for i in self.visiting:
            p_turn_on = self.dynamics(i, weight)
            p_turn_off = 1 - p_turn_on
            self.config[i] = np.random.choice([-1, 1], p=[p_turn_off, p_turn_on])
            self.all_configs[sweep_num] = self.config

    # Visiting order
    def refreshVisitingOrder(self):
        self.visiting = np.roll(self.visiting, 2)

    def empiricalMean_bySweep(self, sweep_num=False):
        if sweep_num:
            mean_by_sweep = np.sum(self.all_configs, axis=0) / (sweep_num + 1)
            self.mean_Mat[sweep_num] = mean_by_sweep

        else:
            return np.sum(self.all_configs, axis=0) / len(self.all_configs)

    def differences_mean(self):
        t = 2
        for i in xrange(111, self.runs, 10):
            differences = self.mean_Mat[i - 10:i] - self.mean_Mat[i - 11:i - 1]
            maximums = [max(differences[j]) for j in xrange(len(differences))]
            if max(maximums) < self.threshold:
                t -= 1
                print i
                print "10 consecutive differences < threshold beginning at", i
                if t == 0:
                    break

    def Total_energy(self, weight):
        self.Energies = []

        for config in self.all_configs:
            config_energy = 0
            for i in xrange(len(self.all_configs[0])):
                for j in xrange(i, len(self.all_configs[0])):
                    config_energy += weight[i, j] * config[i] * config[j]
                energy = -np.sum(config[i] * self.bias[i]) - config_energy
            self.Energies.append(energy)

    def printSTUFF(self):
        print 'shape.mean_Mat', self.mean_Mat.shape
