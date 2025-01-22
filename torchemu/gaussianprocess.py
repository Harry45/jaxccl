"""
Author: Dr. Arrykrishna Mootoovaloo
Date: 17th January 2022
Email: arrykrish@gmail.com, a.mootoovaloo17@imperial.ac.uk, arrykrishna.mootoovaloo@physics
Description: The zero mean Gaussian Process (noise-free implementation, otherwise an extension
of this is to also supply a noise covariance matrix)
"""

from typing import Union, Tuple
import torch
import torch.autograd
import numpy as np
from ml_collections import ConfigDict

import torchemu.kernel as kn
import torchemu.transformation as tr


class GaussianProcess(tr.PreWhiten):
    """Zero mean Gaussian Process

    Args:
        inputs (torch.tensor): the inputs.
        outputs (torch.tensor): the outputs.
        jitter (float): the jitter term.
        xtrans (bool): whether to transform the inputs.
        ytrans (bool): whether to transform the outputs.
    """

    def __init__(
        self,
        config: ConfigDict,
        inputs: torch.tensor,
        outputs: torch.tensor,
        prewhiten: bool = True,
        ylog: bool = False,
    ):
        # store the relevant informations
        self.ytrain = outputs.view(-1, 1)
        self.jitter = config.emu.jitter
        self.xtrans = prewhiten
        self.ytrans = ylog

        # get the dimensions of the inputs
        self.ndata, self.ndim = inputs.shape

        assert (
            self.ndata > self.ndim
        ), "N < d, please reshape the inputs such that N > d."

        if self.xtrans and self.ndim >= 2:
            tr.PreWhiten.__init__(self, inputs)

            # transform the inputs
            self.xtrain = tr.PreWhiten.x_transformation(self, inputs)

        else:
            self.xtrain = inputs

        if self.ytrans:
            ytrain = torch.log(self.ytrain)

        else:
            print("Not using log transformation")
            ytrain = self.ytrain

        self.ymean = torch.mean(ytrain)
        self.ystd = torch.std(ytrain)
        self.ytrain = (ytrain - self.ymean) / self.ystd

    def cost(self, parameters: torch.tensor) -> torch.tensor:
        """Calculates the negative log-likelihood of the GP, for fitting the kernel hyperparameters.

        Args:
            parameters (torch.tensor): the set of input parameters.

        Returns:
            torch.tensor: the value of the negative log-likelihood.
        """

        # compute the kernel matrix
        kernel = kn.compute(self.xtrain, self.xtrain, parameters)

        # add the jitter term to the kernel matrix
        kernel = kernel + torch.eye(self.xtrain.shape[0]) * self.jitter

        # compute the chi2 and log-determinant of the kernel matrix
        log_marginal = -0.5 * self.ytrain.t() @ kn.solve(
            kernel, self.ytrain
        ) - 0.5 * kn.logdeterminant(kernel)

        return -log_marginal

    def optimisation(
        self,
        parameters: torch.tensor,
        niter: int = 10,
        lrate: float = 0.01,
        nrestart: int = 5,
    ) -> dict:
        """Optimise for the kernel hyperparameters using Adam in PyTorch.

        Args:
            parameters (torch.tensor): a tensor of the kernel hyperparameters.
            niter (int): the number of iterations we want to use
            lr (float): the learning rate
            nrestart (int): the number of times we want to restart the optimisation

        Returns:
            dict: dictionary consisting of the optimised values of the hyperparameters and the loss.
        """

        dictionary = {}

        for i in range(nrestart):
            # make a copy of the original parameters and perturb it
            # parameters.clone() + torch.randn(parameters.shape) * 0.1
            params = torch.randn(parameters.shape)

            # make sure we are differentiating with respect to the parameters
            params.requires_grad = True

            # initialise the optimiser
            optimiser = torch.optim.Adam([params], lr=lrate)

            loss = self.cost(params)

            # an empty list to store the loss
            record_loss = [loss.item()]

            # run the optimisation
            for _ in range(niter):
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # evaluate the loss
                loss = self.cost(params)

                # record the loss at every step
                record_loss.append(loss.item())

            dictionary[i] = {"parameters": params, "loss": record_loss}

        # get the dictionary for which the loss is the lowest
        self.d_opt = dictionary[
            np.argmin([dictionary[i]["loss"][-1] for i in range(nrestart)])
        ]

        # store the optimised parameters as well
        self.opt_parameters = self.d_opt["parameters"]

        # compute the kernel and store it
        self.kernel_matrix = kn.compute(
            self.xtrain, self.xtrain, self.opt_parameters.data
        )

        # also compute K^-1 y and store it
        self.alpha = kn.solve(self.kernel_matrix, self.ytrain)

        # return the optimised values of the hyperparameters and the loss
        return dictionary

    def mean_prediction(self, testpoint: torch.tensor) -> torch.tensor:
        """Calculates the mean prediction of the GP.

        Args:
            testpoint(torch.tensor): the test point.

        Returns:
            torch.tensor: the mean prediction from the GP
        """

        testpoint = testpoint.view(-1, 1)
        if self.xtrans and self.ndim >= 2:
            testpoint = tr.PreWhiten.x_transformation(self, testpoint)

        k_star = kn.compute(self.xtrain, testpoint, self.opt_parameters.data)
        mean = k_star.t() @ self.alpha
        return mean

    def derivatives(
        self, testpoint: torch.tensor, order: int = 1
    ) -> Union[Tuple[torch.tensor, torch.tensor], torch.tensor]:
        """Calculates the derivatives of the GP.

        Args:
            testpoint(torch.tensor): the test point.
            order(int, optional): the order of the differentiation. Defaults to 1.

        Returns:
            Union[Tuple[torch.tensor, torch.tensor], torch.tensor]: the derivatives of the GP.
        """

        testpoint.requires_grad = True
        mean = self.mean_prediction(testpoint)

        if self.ytrans:
            mean = torch.exp(mean * self.ystd + self.ymean)

        gradient = torch.autograd.grad(mean, testpoint)

        if order == 1:
            return gradient[0] * self.ystd

        hessian = torch.autograd.functional.hessian(self.mean_prediction, testpoint)
        return gradient[0] * self.ystd, hessian * self.ystd

    def prediction(
        self, testpoint: torch.tensor, variance: bool = False
    ) -> Union[Tuple[torch.tensor, torch.tensor], torch.tensor]:
        """Computes the prediction at a given test point.

        Args:
            testpoint(torch.tensor): a tensor of the test point
            variance(bool, optional): if we want to compute the variance as well. Defaults to False.

        Returns:
            Union[Tuple[torch.tensor, torch.tensor], torch.tensor]: The mean and variance or mean only
        """

        testpoint = testpoint.view(-1, 1)

        if self.xtrans and self.ndim >= 2:
            testpoint = tr.PreWhiten.x_transformation(self, testpoint)

        k_star = kn.compute(self.xtrain, testpoint, self.opt_parameters.data)
        mean = k_star.t() @ self.alpha

        # shift the mean back
        mean = mean * self.ystd + self.ymean
        if self.ytrans:
            mean = torch.exp(mean)

        if variance:
            k_star_star = kn.compute(testpoint, testpoint, self.opt_parameters.data)
            var = k_star_star - k_star.t() @ kn.solve(self.kernel_matrix, k_star)
            if self.ytrans:
                var = (self.ystd * mean) ** 2 * var
            else:
                var = self.ystd**2 * var
            return mean, var
        return mean
