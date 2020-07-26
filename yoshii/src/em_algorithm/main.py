import argparse
import random, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# GaussianMixtureModel with EM algorithm
class GMM:
        def __init__(self, k, max_iteration, eps=0.0001):
                self.k = k
                self.max_iteration = max_iteration
                self.eps = eps
        
        # learn parameters and output Posterior probabilities
        def fit_predict(self, X):
                n, d = X.shape

                # init parameters
                self.mu = np.random.randn(self.k, d)
                self.sigma = np.array([np.eye(d) for _ in range(self.k)])
                self.pi = np.array([1 / self.k for _ in range(self.k)]) 

                # iterate 'E' and 'M' step
                old_likelihood = self.likelihood(X)
                for _ in range(self.max_iteration):
                        self.e_step(X)
                        self.m_step(X)

                        next_likelihood = self.likelihood(X)
                        if np.abs(next_likelihood - old_likelihood) < self.eps:
                                break
                        else:
                                print("likelihood: " + str(next_likelihood))
                                old_likelihood = next_likelihood
                
                params = {
                        'mu'   : self.mu,
                        'sigma': self.sigma,
                        'pi'   : self.pi,
                }

                return self.gamma, params

        # only learn parameters
        def fit(self, X):
                self.fit_predict(X)

        # the 'E' step of EM algorithm
        def e_step(self, X):
                n, d = X.shape

                self.gamma = np.zeros((self.k, n))
                for i in range(self.k):
                        rv = multivariate_normal(self.mu[i], self.sigma[i])
                        self.gamma[i] = self.pi[i] * rv.pdf(X)

                self.gamma = self.gamma.T
                for i in range(n):
                        self.gamma[i] = self.gamma[i] / np.sum(self.gamma[i])
                                        
        # the 'M' step of EM algorithm
        def m_step(self, X):
                n, d = X.shape
                n_k = np.sum(self.gamma, axis=0)

                # calculate next 'mu'
                for i in range(self.k):
                        t = np.zeros(d)
                        for j in range(n):
                                t += self.gamma[j][i] * X[j];
                        self.mu[i] = t / n_k[i]
        
                # calculate next 'sigma'
                for i in range(self.k):
                        t = np.zeros((d, d))
                        for j in range(n):
                                t += self.gamma[j][i] * (X[j] - self.mu[i])[:,None] @ (X[j] - self.mu[i])[:,None].T
                        self.sigma[i] = t / n_k[i]

                # calculate next 'pi'
                self.pi = n_k / n

        # calculate the log likelihood
        def likelihood(self, X):
                n, d = X.shape
                
                l = 0
                for j in range(n):
                        t = 0
                        for i in range(self.k):
                                rv = multivariate_normal(self.mu[i], self.sigma[i])
                                t += self.pi[i] * rv.pdf(X[j])
                        
                        l += np.log(t)

                return l

        def n_parameters(self):
                _, d = self.mu.shape
                cov_params = self.k * d * (d + 1) / 2
                mu_params  = self.k * d

                return cov_params + mu_params

        def aic(self, X):
                return -2 * (self.likelihood(X) + self.n_parameters())
                
if __name__ == "__main__":
        # prepare argparser and parse commandline options
        parser = argparse.ArgumentParser()
        parser.add_argument("src", help="src csv file name")
        parser.add_argument("dst", help="dst csv file name")
        parser.add_argument("params", help="params file name")
        parser.add_argument("k", type=int)
        parser.add_argument("-f", "--figure", help="output figure", action="store_true")
        args = parser.parse_args()

        # read csv and learn
        dat = pd.read_csv(args.src, header=None).values
        gmm = GMM(k=args.k, max_iteration=100)
        res, params = gmm.fit_predict(dat)
        print("AIC: " + str(gmm.aic(dat)))

        # output result csv and params dat
        pd.DataFrame(res).to_csv(
                args.dst,
                header=False,
                index=False,
        )

        open(args.params, mode='w').close()
        with open(args.params, mode='a') as f:
                np.savetxt(f, params['mu'], header='mu')
                for i in range(len(params['sigma'])):
                        np.savetxt(f, params['sigma'][i], header='sigma' + str(i))
                np.savetxt(f, params['pi'], header='pi')
        
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
