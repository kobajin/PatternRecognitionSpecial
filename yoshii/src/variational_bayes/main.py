import argparse
import random, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import psi
from mpl_toolkits.mplot3d import Axes3D

# GaussianMixtureModel with VariationalBayes
class GMM:
        def __init__(self, k, max_iteration, eps=0.0001):
                self.k = k
                self.max_iteration = max_iteration
                self.eps = eps
        
        # learn parameters and output Posterior probabilities
        def fit_predict(self, X):
                n, d = X.shape

                # init parameters
                self.alpha = np.random.randn(self.k)
                self.beta  = np.random.randn(self.k)
                self.m     = np.random.randn(self.k, d)
                self.W     = np.array([np.eye(d) for _ in range(self.k)])
                self.nu    = np.array([max(d, 2 * d + np.random.randn()) for _ in range(self.k)])

                # iterate E and M step
                for _ in range(self.max_iteration):
                        self.e_step(X)
                        self.m_step(X)
                        
                return self.gamma

        # only learn parameters
        def fit(self, X):
                self.fit_predict(X)

        # the 'E' step of EM algorithm
        def e_step(self, X):
                n, d = X.shape
                
                self.gamma = np.zeros((self.k, n))
                for i in range(self.k):
                        for j in range(n):
                                e_log_pi = psi(self.alpha[i]) - psi(np.sum(self.alpha))
                                e_log_sigma = np.sum([psi((self.nu[i] + 1 - i) / 2) for i in range(1, d+1)]) + d*np.log(2) + np.log(np.linalg.det(self.W[i]))
                                
                                t = e_log_pi + e_log_sigma/2 - d/(2*self.beta[i]) - self.nu[i]/2 * (X[j]-self.m[i]).T @ self.W[i] @ (X[j]-self.m[i])
                                self.gamma[i][j] = np.exp(t) 

                self.gamma = self.gamma.T
                for i in range(n):
                        self.gamma[i] = self.gamma[i] / np.sum(self.gamma[i])

                self.gamma = self.gamma.T
                
        # the 'M' step of EM algorithm
        def m_step(self, X):
                n, d = X.shape
                
                # define some variables
                n_k   = np.sum(self.gamma, axis=1)
                x_bar = np.zeros((self.k, d))
                s_k   = np.zeros((self.k, d, d))

                for i in range(self.k):
                        x_bar[i] = self.gamma[i] @ X / n_k[i]
                        t = 0
                        for j in range(n):
                                t += self.gamma[i][j] * (X[j]-x_bar[i])[:,None] @ (X[j]-x_bar[i])[:,None].T
                        s_k[i] = t / n_k[i]

                # calculate next 'alpha'
                self.alpha += n_k

                # calculate next 'W'
                for i in range(self.k):
                        w_inv     = np.linalg.inv(self.W[i]) + s_k[i]/n_k[i] + (self.beta[i]*n_k[i]) / (self.beta[i]+n_k[i]) * (x_bar[i]-self.m[i])[:,None] @ (x_bar[i]-self.m[i])[:,None].T
                        self.W[i] = np.linalg.inv(w_inv) 

                # calculate next 'm'
                for i in range(self.k):
                        self.m[i] = (self.beta[i]*self.m[i] + n_k[i]*x_bar[i]) / (self.beta[i] + n_k[i])

                # calculate next 'beta'
                self.beta += n_k

        def calc_lowerbound(self, X):
                
                
if __name__ == "__main__":
        # prepare argparser and parse commandline options
        parser = argparse.ArgumentParser()
        parser.add_argument("src", help="src csv file")
        parser.add_argument("dst", help="output csv file")
        parser.add_argument("-f", "--figure", help="output figure", action="store_true")
        args = parser.parse_args()

        # read csv and learn
        dat = pd.read_csv(args.src, header=None).values
        gmm = GMM(k=4, max_iteration=50)
        res = gmm.fit_predict(dat).T

        # output csv
        pd.DataFrame(res).to_csv(
                args.dst,
                header=False,
                index=False,
        )
        
        # if --figure is set, then output the graph
        if args.figure:
                fig = plt.figure()
                ax = Axes3D(fig)

                ind = res.argmax(axis=1)
                cm = plt.get_cmap("tab10")
                for i in range(dat.shape[0]):
                        ax.plot([dat[i][0]], [dat[i][1]], [dat[i][2]], "o", color=cm(ind[i]), ms=1)

                ax.view_init(elev=30, azim=45)
                plt.savefig("".join(random.choices(string.ascii_letters, k=10)) + ".png")
                plt.show()
