import argparse
import random, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, wishart
from mpl_toolkits.mplot3d import Axes3D

# GaussianMixtureModel with Gibbs sampling
class GMM:
        def __init__(self, k, max_iteration, eps=0.0001):
                self.k = k
                self.max_iteration = max_iteration
                self.eps = eps
        
        # learn parameters and output Posterior probabilities
        def fit_predict(self, X):
                n, d = X.shape

                # init parameters
                self.z     = np.zeros((n, self.k))
                self.pi    = np.array([1/self.k for _ in range(self.k)])
                self.mu    = np.random.randn(self.k, d)
                self.sigma = np.array([np.eye(d) for _ in range(self.k)])

                self.alpha = np.random.rand(self.k) * 10
                self.m     = np.random.randn(self.k, d)
                self.beta  = np.random.randn(self.k) * 10
                self.W_inv = np.array([np.eye(d) for _ in range(self.k)])
                self.nu    = d + np.random.rand(self.k) * 10
        
                for _ in range(self.max_iteration):
                        # sampling 'z'
                        z_t = np.zeros((self.k, n))
                        for i in range(self.k):
                                rv = multivariate_normal(self.mu[i], self.sigma[i])
                                z_t[i] = self.pi[i] * rv.pdf(X)
                        self.z = z_t.T
                
                        for i in range(n):
                                self.z[i] = self.z[i] / np.sum(self.z[i])

                        # define helper variables
                        sk    = np.sum(self.z, axis=0)
                        sk_x  = np.zeros((self.k, d))
                        sk_xx = np.zeros((self.k, d, d))
                        for i in range(self.k):
                                for j in range(n):
                                        sk_x[i]  += self.z[j][i] * X[j]
                                        sk_xx[i] += self.z[j][i] * X[j][:, None] @ X[j][:, None].T

                        # update Dirichlet parameters
                        self.alpha += sk

                        # sampling 'pi'
                        self.pi = np.random.dirichlet(self.alpha) 

                        # update Gaussian-Wihshart parameters
                        for i in range(self.k):
                                old_beta     = self.beta[i]
                                self.beta[i] += sk[i]
                                old_m        = self.m[i]
                                self.m[i]    = (self.beta[i] * self.m[i] + sk_x[i]) / (self.beta[i] + sk[i])
                                
                                self.nu[i]    += sk[i]
                                self.W_inv[i] += old_beta * old_m[:,None] @ old_m[:,None].T + sk_xx[i] - self.beta[i] * self.m[i][:,None] @ self.m[i][:,None].T

                                # sampling 'mu' and'sigma'
                                self.sigma[i] = wishart.rvs(df=self.nu[i], scale=np.linalg.inv(self.W_inv[i]))
                                self.mu[i]    = multivariate_normal.rvs(mean=self.m[i], cov=np.linalg.inv(self.beta[i] * self.sigma[i]))

                return self.z

        # only learn parameters
        def fit(self, X):
                self.fit_predict(X)
                
if __name__ == "__main__":
        # prepare argparser and parse commandline options
        parser = argparse.ArgumentParser()
        parser.add_argument("src", help="src csv file")
        parser.add_argument("dst", help="output csv file")
        parser.add_argument("-f", "--figure", help="output figure", action="store_true")
        args = parser.parse_args()

        # read csv and learn
        dat = pd.read_csv(args.src, header=None).values
        gmm = GMM(k=4, max_iteration=100)
        res = gmm.fit_predict(dat)

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
