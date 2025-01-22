# Author: Dr. Arrykrishna Mootoovaloo
# Date: 17th January 2022
# Email: arrykrish@gmail.com, a.mootoovaloo17@imperial.ac.uk, arrykrishna.mootoovaloo@physics
# Description: Code to transform the inputs to the GP, that is, the input training points

import torch


class PreWhiten(object):
    def __init__(self, xinputs: torch.tensor):
        # compute the covariance of the inputs (ndim x ndim)
        self.cov_train = torch.cov(xinputs.t())
        self.ndim = xinputs.shape[1]

        # compute the Cholesky decomposition of the matrix
        self.chol_train = torch.linalg.cholesky(self.cov_train)

        # compute the mean of the sample
        self.mean_train = torch.mean(xinputs, axis=0).view(1, self.ndim)

    def x_transformation(self, point: torch.tensor) -> torch.tensor:
        """Pre-whiten the input parameters.

        Args:
            point (torch.tensor): the input parameters.

        Returns:
            torch.tensor: the pre-whitened parameters.
        """

        # ensure the point has the right dimensions
        point = point.view(-1, self.ndim)

        # calculate the transformed training points
        transformed = torch.linalg.inv(self.chol_train) @ (point - self.mean_train).t()

        return transformed.t()
