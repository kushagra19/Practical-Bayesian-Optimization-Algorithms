import torch
import numpy as np

class ObjectiveFunction(torch.nn.Module):
    '''
        This class is an abstract class for an objective function that
        is to be maximised using Bayesian Optimisation.

        Source:
        https://botorch.org/api/_modules/botorch/test_functions/base.html
    '''

    def __init__(self, dims, low, high, noise_mean=0, noise_variance=None,
        random_state=None, negate=False):
        '''
            Arguments:
            ---------
                - noise_mean: Mean of the normal noise to be added to
                    the function and gradient values
                - noise_variance: Variance of the normal noise to be
                    added to the function and gradient values
                - random_state: Equivalent PyTorch manual_seed
                - negate: Multiplies the value of the function obtained
                    with -1. Use this to minimize the Objective Function.
                - dims: The number of dimensions in the objective
                    function. For example, in Branin function, dims=2
                - low: (A PyTorch tensor) of shape (d,) which each
                    dimension represents the lower limit of x along that
                    dimension
                - high: (A PyTorch tensor) of shape (d,) which each
                    dimension represents the upper limit of x along that
                    dimension
        '''

        super().__init__()
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        self.noise_std = np.sqrt(noise_variance) if noise_variance else None
        self.negate = negate
        self.dims = dims
        self.low = low
        self.high = high

    def evaluate_true(self, X):
        '''
            This function calculates the value of the objective function
            at X without noise. X is a tensor of shape batch_size x d,
            where d is the dimension of the input taken by the Objective
            Function.

            Returns a PyTorch tensor of shape batch_size x 1 if the
            objective function is a single output function.
        '''
        raise NotImplementedError


    def forward(self, X, noise=True):
        '''
            This function calculates the value of the Objective
            Function at X with some Gaussian Noise added. The shape of X
            is batch_size x d, where d is the dimension of the input
            taken by the Objective Function.

            Arguments:
            ---------
                - X: A PyTorch tensor of shape (batch_size, d) or (d, ).
                - noise (boolean): Indicates whether noise should be
                    added.

            Returns:
            -------
                - A PyTorch Tensor of shape (batch_size, 1) or (1, )
                    depending on whether the input X is (batch_size, d)
                    or (d, ).
        '''

        self.X = X.detach().clone()

        batch = self.X.ndimension() > 1
        self.X = self.X if batch else self.X.unsqueeze(0)

        self.X.requires_grad = True
        if self.X.grad:
            self.X.grad.data.zero_()

        self.f_x = self.evaluate_true(X=self.X)
        self.f_x_true = self.f_x.detach.clone()

        with torch.no_grad():
            if noise and self.noise_variance is not None:
                self.f_x += (self.noise_std * torch.randn_like(self.f_x) +
                    self.noise_mean)

        if self.negate:
            self.f_x = -self.f_x

        f_x = (self.f_x.detach().clone() if batch else
            self.f_x.detach().clone().squeeze(0))

        return f_x

    def forward_true(self, X):
        '''
            This function calculates the value of the Objective
            Function at X with some Gaussian Noise added. The shape of X
            is batch_size x d, where d is the dimension of the input
            taken by the Objective Function.

            Arguments:
            ---------
                - X: A PyTorch tensor of shape (batch_size, d) or (d, ).
                - noise (boolean): Indicates whether noise should be
                    added.

            Returns:
            -------
                - A PyTorch Tensor of shape (batch_size, 1) or (1, )
                    depending on whether the input X is (batch_size, d)
                    or (d, ).
        '''

        self.X = X.detach().clone()

        batch = self.X.ndimension() > 1
        self.X = self.X if batch else self.X.unsqueeze(0)

        self.X.requires_grad = True
        if self.X.grad:
            self.X.grad.data.zero_()

        self.f_x = self.evaluate_true(self.X)
        self.f_x_true = self.f_x.detach().clone()

        # with torch.no_grad():
        #     if noise and self.noise_variance is not None:
        #         self.f_x += (self.noise_std * torch.randn_like(self.f_x) +
        #             self.noise_mean)

        # if self.negate:
        #     self.f_x = -self.f_x

        f_x = (self.f_x_true.detach().clone() if batch else
            self.f_x_true.detach().clone().squeeze(0))

        return f_x

    def backward(self, noise=True):
        '''
            This function calculates the gradients at X i.e. the points
            where the Objective Function value was calculated during the
            forward pass.

            Arguments:
            ---------
                - noise (boolean): Indicates whether noise should be
                    added.

            Returns:
            -------
                - A PyTorch Tensor of shape (batch_size, d).
        '''

        external_grad = torch.tensor([1.]*self.X.shape[0])
        self.f_x.backward(gradient=external_grad)

        self.grads = self.X.grad

        if noise and self.noise_variance is not None:
            self.grads += (self.noise_variance * torch.randn_like(self.grads)
                    + self.noise_mean)

        return self.grads.detach().clone()
