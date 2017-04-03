from __future__ import division

import train_rbm as rbm

import numpy as np
from sklearn.decomposition import PCA


class Analysis():
    """docstring for ClassName"""
    def __init__(self):
        pass

    def principal_component(self, hidden_cloud):
        """ perform pca
            compute eigenvalues needed to achieve 90% energy of configuration
            compute projection of p classes from PCA, onto the first three
            eigenvectors, plot this.
        """
        pca = PCA(n_components=3)
        pca.fit(hidden_cloud)
        pca90 = PCA(n_components=32)
        pca90.fit(hidden_cloud)
        print('number of components that explain 90% of the energy',
              pca90.explained_variance_ratio_.cumsum())
        self.x_dimreduced = pca.fit(hidden_cloud).transform(hidden_cloud)
        print "x_dimreduced shape", self.x_dimreduced.shape
        return self.x_dimreduced


