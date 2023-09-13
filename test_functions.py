import math
import numpy
import torch
import _pickle as pickle
import gym
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from ObjectiveFunction import ObjectiveFunction

class LeBranke(ObjectiveFunction):
    '''
        This class implements a simple 1D function from Le and Branke,
        "Bayesian Optimization Searching for Robust Solutions." In
        Proceedings of the 2020 Winter Simulation Conference. The
        function is defined as follows:
            -0.5(x+1)sin(pi*x**2)

        Note that this function has to be maximised in the domain
        [0.1, 2.1]. The function attains maximum value of 1.43604
        at x=1.87334.
    '''

    def __init__(self, noise_mean=None, noise_variance=None,
        negate=False):

        dims = 1
        low = torch.tensor([0.1])
        high = torch.tensor([2.1])

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )

        self.true_opt_value = 1.43604

    def evaluate_true(self, X):
        '''
            This function calculates the value of the Branin function
            without any noise.

            For more information, refer to the ObjectiveFunction class
            docs.
        '''
        torch.pi = torch.acos(torch.zeros(1)).item() * 2

        self.f_x = -0.5*(X+1)*torch.sin(torch.pi*X**2)

        return self.f_x.squeeze(-1)


class Branin(ObjectiveFunction):
    '''
        This class implements the Branin function, a simple benchmark
        function in two dimensions.

        Branin is usually evaluated on [-5, 10] x [0, 15]. The minima
        of Branin is 0.397887 which is obtained at (-pi, 12.275),
        (pi, 2.275) and (9.42478, 2.475).

        For information on Branin function visit:
            https://www.sfu.ca/~ssurjano/branin.html
    '''
    def __init__(self, noise_mean=None, noise_variance=None,
        negate=False):

        dims = 2
        low = torch.tensor([-5, 0])
        high = torch.tensor([10, 15])

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )
        self.true_opt_value = 0.397887

    def evaluate_true(self, X):
        '''
            This function calculates the value of the Branin function
            without any noise.

            For more information, refer to the ObjectiveFunction class
            docs.
        '''
        torch.pi = torch.acos(torch.zeros(1)).item() * 2

        self.f_x = ((X[:, 1] - 5.1/(4*torch.pi**2)*X[:, 0]**2 +
                 5*X[:, 0]/torch.pi - 6)**2 +
                 10*(1- 1/(8*torch.pi))*torch.cos(X[:, 0]) + 10)

        return self.f_x

class Levy(ObjectiveFunction):
    '''
        This class implements the Levy function.

        The function is usually evaluated on the hypercube xi belongs
        to [-10, 10] for i=1,...,d. The global minima is 0.0 which is
        obtained at (1,...,1).

        For information on Branin function visit:
            https://www.sfu.ca/~ssurjano/levy.html
    '''
    def __init__(self, dims, noise_mean=None, noise_variance=None,
        negate=False):

        low = torch.tensor([-2]*dims)
        high = torch.tensor([2]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )
        self.true_opt_value = 0.0

    def evaluate_true(self, X):
        '''
            This function calculates the value of the Levy function
            without any noise.

            For more information, refer to the ObjectiveFunction class
            docs.

            Source:
            ------
            https://botorch.org/api/_modules/botorch/test_functions/synthetic.html#Levy
        '''
        w = 1.0 + (X - 1.0) / 4.0
        part1 = torch.sin(math.pi * w[..., 0]) ** 2
        part2 = torch.sum(
            (w[..., :-1] - 1.0) ** 2
            * (1.0 + 10.0 * torch.sin(math.pi * w[..., :-1] + 1.0) ** 2),
            dim=-1,
        )
        part3 = (w[..., -1] - 1.0) ** 2 * (
            1.0 + torch.sin(2.0 * math.pi * w[..., -1]) ** 2
        )

        self.f_x = part1 + part2 + part3

        return self.f_x

class Rosenbrock(ObjectiveFunction):
    '''
    '''
    def __init__(self, dims, noise_mean=None, noise_variance=None,
        negate=False):

        low = torch.tensor([-5]*dims)
        high = torch.tensor([10]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )
        self.true_opt_value = 0.0

    def evaluate_true(self, X):
        return torch.sum(
            100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (X[..., :-1] - 1) ** 2,
            dim=-1,
        )

class Ackley(ObjectiveFunction):
    '''
    '''
    def __init__(self, dims, noise_mean=None, noise_variance=None,
        negate=False):

        low = torch.tensor([-32.768]*dims)
        high = torch.tensor([32.768]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )
        self.true_opt_value = 0.0

        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi

    def evaluate_true(self, X):
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dims) * torch.norm(X, dim=-1))
        part2 = -(torch.exp(torch.mean(torch.cos(c * X), dim=-1)))
        return part1 + part2 + a + math.e

class Hartmann(ObjectiveFunction):
    '''
        TO-DO: Implement Hartmann for multiple dimensions.
    '''
    def __init__(self, dims, noise_mean=None, noise_variance=None,
        negate=False):

        if dims not in (3, 4, 6):
            raise ValueError(f"Hartmann with dim {dims} not defined")

        low = torch.tensor([0.0]*dims)
        high = torch.tensor([1.0]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )
        self.true_opt_value = -3.32237  # For 6 dim Hartmann

        self.ALPHA = torch.tensor([1.0, 1.2, 3.0, 3.2])

        if dims == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        elif dims == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dims == 6:
            self.A = torch.tensor([
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ])
            self.P = torch.tensor([
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ])

    def evaluate_true(self, X):
        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = -(torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1))
        if self.dims == 4:
            H = (1.1 + H) / 0.839
        return H

class Powell(ObjectiveFunction):

    def __init__(self, dims, noise_mean=None, noise_variance=None,
        random_state=None, negate=False):

        low = torch.tensor([-4]*dims)
        high = torch.tensor([5]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = 0.0

    def evaluate_true(self, X):
        result = torch.zeros_like(X[..., 0])
        for i in range(self.dims // 4):
            i_ = i + 1
            part1 = (X[..., 4 * i_ - 4] + 10.0 * X[..., 4 * i_ - 3]) ** 2
            part2 = 5.0 * (X[..., 4 * i_ - 2] - X[..., 4 * i_ - 1]) ** 2
            part3 = (X[..., 4 * i_ - 3] - 2.0 * X[..., 4 * i_ - 2]) ** 4
            part4 = 10.0 * (X[..., 4 * i_ - 4] - X[..., 4 * i_ - 1]) ** 4
            result += part1 + part2 + part3 + part4
        return result

class Holder_Table(ObjectiveFunction):
    '''
        This class implements the Levy function.

        The function is usually evaluated on the hypercube xi belongs
        to [-10, 10] for i=1,...,d. The global minima is 0.0 which is
        obtained at (1,...,1).

        For information on Branin function visit:
            https://www.sfu.ca/~ssurjano/levy.html
    '''
    def __init__(self, dims, noise_mean=None, noise_variance=None,
        random_state=None, negate=False):

        low = torch.tensor([-10]*dims)
        high = torch.tensor([10]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = -19.2085

    def evaluate_true(self, X):
        term = torch.abs(1 - torch.norm(X, dim=-1) / math.pi)
        return -(
            torch.abs(torch.sin(X[..., 0]) * torch.cos(X[..., 1]) * torch.exp(term))
        )

class Cosine8(ObjectiveFunction):

    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):

        low = torch.tensor([-1]*dims)
        high = torch.tensor([1]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = 0.8

    def evaluate_true(self, X):
        return torch.sum(0.1 * torch.cos(5 * math.pi * X) - X ** 2, dim=-1)

class Drop_Wave(ObjectiveFunction):

    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):

        low = torch.tensor([-5.12]*dims)
        high = torch.tensor([5.12]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = -1

    def evaluate_true(self, X):
        norm = torch.norm(X, dim=-1)
        part1 = 1.0 + torch.cos(12.0 * norm)
        part2 = 0.5 * norm.pow(2) + 2.0
        return -part1 / part2

class StyblinskiTang(ObjectiveFunction):

    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):
        
        low = torch.tensor([-5]*dims)
        high = torch.tensor([5]*dims)
        
        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = -39.166166 * dims

    def evaluate_true(self, X):
        return 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(dim=-1)

class DixonPrice(ObjectiveFunction):

    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):
        
        low = torch.tensor([-10]*dims)
        high = torch.tensor([10]*dims)
        
        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = 0

    def evaluate_true(self, X):
        d = self.dims
        part1 = (X[..., 0] - 1) ** 2
        i = X.new(range(2, d + 1))
        part2 = torch.sum(i * (2.0 * X[..., 1:] ** 2 - X[..., :-1]) ** 2, dim=-1)
        return part1 + part2

class SixHumpCamel(ObjectiveFunction):

    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):

        dims = 2
        low = torch.tensor([-3, -2])
        high = torch.tensor([3, 2])

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = -1.0316

    def evaluate_true(self, X):
        x1, x2 = X[..., 0], X[..., 1]
        return (
            (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2
            + x1 * x2
            + (4 * x2 ** 2 - 4) * x2 ** 2
        )

class ThreeHumpCamel(ObjectiveFunction):
    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):

        dims = 2
        low = torch.tensor([-5, -5])
        high = torch.tensor([5, 5])

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = 0

    def evaluate_true(self, X):
        x1, x2 = X[..., 0], X[..., 1]
        return 2.0 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 6.0 + x1 * x2 + x2 ** 2

class Sphere(ObjectiveFunction):
    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):

        low = torch.tensor([-5.12]*dims)
        high = torch.tensor([5.12]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = 0

    def evaluate_true(self, X):
        return torch.sum(X ** 2,dim=-1)

class Bukin(ObjectiveFunction):
    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):
        dims = 2
        low = torch.tensor([-15,-3])
        high = torch.tensor([-5,3])

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = 0

    def evaluate_true(self, X):
        part1 = 100.0 * torch.sqrt(torch.abs(X[..., 1] - 0.01 * X[..., 0] ** 2))
        part2 = 0.01 * torch.abs(X[..., 0] + 10.0)
        return part1 + part2

class Griewank(ObjectiveFunction):
    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):
        low = torch.tensor([-600]*dims)
        high = torch.tensor([600]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = 0

    def evaluate_true(self, X):
        part1 = torch.sum(X ** 2 / 4000.0, dim=-1)
        d = X.shape[-1]
        part2 = -(torch.prod(torch.cos(X / torch.sqrt(X.new(range(1, d + 1)))), dim=-1))
        return part1 + part2 + 1.0

class Michalewicz(ObjectiveFunction):
    r"""Michalewicz synthetic test function.

    d-dim function (usually evaluated on hypercube [0, pi]^d):

        M(x) = sum_{i=1}^d sin(x_i) (sin(i x_i^2 / pi)^20)
    """
    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):
        low = torch.tensor([0]*dims)
        high = torch.tensor([math.pi]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = -9.66015
        self.i = torch.tensor(tuple(range(1, self.dims + 1)), dtype=torch.float)

    def evaluate_true(self, X):
        self.to(device=X.device, dtype=X.dtype)
        m = 10
        return -(
            torch.sum(
                torch.sin(X) * torch.sin(self.i * X ** 2 / math.pi) ** (2 * m), dim=-1
            )
        )

class Sum_exp(ObjectiveFunction):
    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):

        low = torch.tensor([-1]*dims)
        high = torch.tensor([1]*dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self.true_opt_value = 0

    def evaluate_true(self, X):
        val = 0
        for i in range(1,self.dims+1):
            X[:,i-1] = X[:,i-1]**(i+1)
        return torch.sum(X,dims = -1)

class gridworld(object):
    def __init__(self, m=4, n=4):
        self.grid = numpy.zeros((m, n))
        self.m = m
        self.n = n
        self.stateSpace = [i for i in range(self.m * self.n)]
        self.stateSpace.remove(self.m * self.n - 1)
        self.stateSpaceplus = [i for i in range(self.m * self.n)]
        self.actionSpace = {0: -self.m, 1: self.m, 2: -1, 3: +1}
        self.possibleActions = [0,1,2,3]
        self.agentPosition = 0
        self.penality_state = self.add_penality()
        self.tot_steps = 0
        self.max_episode_steps = 500
        with open('probs', 'rb') as file:
            self.reward = pickle.load(file)

    def add_penality(self):
        return numpy.array(6)

    def isTerminalState(self, state):
        # return (state in self.stateSpaceplus and state not in self.stateSpace)
        return (state in self.stateSpaceplus and state not in self.stateSpace or self.tot_steps == self.max_episode_steps)

    def AgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
        return x,y

    def setState(self,state):
        x,y = self.AgentRowAndColumn()
#         self.grid[int(x)][int(y)] = 0
        self.agentPosition = state
        x,y = self.AgentRowAndColumn()
#         self.grid[x][y] = 1

    def offGridMove(self, newstate, oldstate):
        if newstate not in self.stateSpaceplus:
            return True
        elif oldstate % self.m == 0 and newstate % self.m == self.m - 1:
            return True
        elif oldstate % self.m == self.m - 1 and newstate % self.m == self.m:
            return True
        else:
            return False

    def step(self, action):
        self.tot_steps += 1
        x,y = self.AgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]

#         reward = -1 if resultingState not in isTerminalState() else 0
        reward = self.getReward(resultingState)

        if not self.offGridMove(resultingState,self.agentPosition):
            self.setState(resultingState)
            return resultingState,reward,self.isTerminalState(self.agentPosition), None
        else:
            return self.agentPosition,reward,self.isTerminalState(self.agentPosition), None

    def getReward(self,new_state):
        if self.isTerminalState(new_state):
            return +1
      #  elif new_state in self.penality_state:
         #   return -100
        else:
            return numpy.random.choice([-5,-1,-2,1,2,5,10],p = self.reward[self.agentPosition])
      #      x,y = self.AgentRowAndColumn()
            # print(x,y)
      #      return -1*(abs(x-self.m + 1)+abs(y-self.n + 1))
      #      return numpy.random.choice([-5,-1,-2,1,2,5,10],p = [0.2,0.2,0.2,0.2,0.1,0.05,0.05])


    def reset(self):
        self.agentPosition = 0
        self.tot_steps = 0
        self.grid = numpy.zeros((self.m*self.n))
        return self.agentPosition
    def actionSpaceSample(self):
        return numpy.random.choice(self.possibleActions)

    # def render(self):
    #     print("--------------------")
    #     for row in self.grid:
    #         for column in row:
    #             if col == 0:
    #                 print("-", end = '\t')
    #             elif col == 1:
    #                 print ('X', end = '\t')
    #             elif col == 2:
    #                 print("Ain", end  ='\t')
    #             elif col == 3:
    #                 print('Aout', end = '\t')
    #             elif col == 4:
    #                 print("Bin",end = '\t')
    #             elif col == 5:
    #                 print("Bout", end ='\t')
    #         print("\n")
    #     print("--------------------")


class NN_Model_gw(torch.nn.Module):

    def __init__(self,dic):
        super(NN_Model_gw, self).__init__()
        self.linear1 = torch.nn.Linear(16, 1,bias = False)
        # self.activation = torch.nn.Tanh()
        self.activation = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(1, 4, bias = False)
        # with torch.no_grad():
        self.linear1.weight = torch.nn.Parameter(data = dic['W1'].float())
        self.linear2.weight = torch.nn.Parameter(data = dic['W2'].float())
#         torch.nn.init.xavier_normal_(self.linear1.weaight, gain=1.0)
#         torch.nn.init.xavier_normal_(self.linear2.weight, gain=1.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class gw(ObjectiveFunction):
    def __init__(self, dims, noise_mean=None, noise_variance=None,
                 random_state=None, negate=False):
        dims = 20
        low = torch.tensor([-10] * dims)
        high = torch.tensor([10] * dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )

        self._observations = []

        # hyperparameters
        self.H = 1  # number of hidden layer neurons
        self.output_size = 4
        self.batch_size = 100 # every how many episodes to do a param update?
        self.gamma = 0.99 # discount factor for reward
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
        self.resume = False  # resume from previous checkpoint?
        self.render = False

        # model initialization
        self.D = 16  # input dimensionality: 1x8 vector
        if self.resume:
            self.model = pickle.load(open('gw.p', 'rb'))
        else:
            self.model = {}
            self.model['W1'] = torch.randn(self.H, self.D) / torch.sqrt(torch.tensor(self.D))  # "Xavier" initialization
            self.model['W2'] = torch.randn(self.output_size,self.H) / torch.sqrt(torch.tensor(self.H))

        self.grad_buffer = {k: torch.zeros_like(v) for k, v in self.model.items()}  # update buffers that add up gradients over a batch
        self.rmsprop_cache = {k: torch.zeros_like(v) for k, v in self.model.items()}  # rmsprop memory

        self.env = gridworld()
        # self.env.max_episode_steps = 500
        self.observation = self.env.reset()

        self.logps, self.drs = [], []
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0
        self.nn_model = NN_Model_gw(self.model)

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(dim=0))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def to_one_hot(self,i, n_classes=None):
        a = torch.zeros(n_classes)
        a[i] = 1.
        return a

    def computeCost(self, model):
        l = []
        m = torch.zeros_like(model)
        # print(model)
        count = 0
        rew_sample = []
        # print(model.shape)
        for samples in model:
            # print(samples)
            # Count number of episodes
            self.episode_number = 0
            self.episode_number += 1
            toReturn = []
            toReturnReward = 0.0
            # Reshape the model we get according to self.model
            self.model['W1'] = torch.reshape(samples[0:self.H * self.D], (self.H, self.D))
            self.model['W2'] = torch.reshape(samples[self.H * self.D:], (self.output_size, self.H))
            #Declaring fisher matrix
            # F = torch.zeros((self.H*self.D+self.H*2,self.H*self.D+self.H*2))
            # F_matrix = torch.zeros((self.H*self.D+self.H*2,self.H*self.D+self.H*2))
            # Dictionary to store results temporarily
            tmp_dic = {}
            tmp_dic['W1'] = torch.zeros(self.H, self.D)
            tmp_dic['W2'] = torch.zeros(self.output_size, self.H)
            self.nn_model = NN_Model_gw(self.model)
            # print(self.nn_model.linear1.weight,self.nn_model.linear2.weight)
            num_steps = 0
            sum_rew = 0
            # Start
            while(self.episode_number % self.batch_size != 0):
                num_steps += 1
                if self.render:
                    self.env.render()
                # print(self.observation)
                one_hot = self.to_one_hot(self.observation,self.D)
                # print(one_hot)
                x = torch.tensor(one_hot , requires_grad=True)
                # forward the policy network and sample an action from the returned probability
                apred = self.nn_model(x)
                # Calculate probability of each action using softmax
                aprob = torch.nn.functional.softmax(apred, dim=-1)
                # sample action from Categorical distribution
                dist = torch.distributions.Categorical(aprob)
                action = dist.sample()
                # store the log_probability of our action
                self.logps.append(dist.log_prob(action))

                #grads for fisher matrix
                # self.nn_model.zero_grad(set_to_none = True)
                # log_prob_action = dist.log_prob(action)
                # log_prob_action.backward(retain_graph=True)
                # #Storing log_prob grads
                # grads = torch.cat(((torch.reshape(self.nn_model.linear1.weight.grad,\
                #                                  (self.H*self.D,1))),\
                #                   (torch.reshape(self.nn_model.linear2.weight.grad,\
                #                                  (self.H*2,1)))))
                # #Fisher matrix over steps and incrementally average
                # F += (torch.matmul(grads,torch.transpose(grads,0,1))-F)/num_steps

                # step the environment and get new measurements
                self.observation, reward, done, info = self.env.step(action.item())
                # Accumulate reward over episode
                self.reward_sum = self.reward_sum + reward
                # Store reward recieved at each step
                self.drs.append(torch.tensor(reward,dtype = torch.float32))
                if done:  # an episode finished
                    # print(self.reward_sum)
                    # print(self.episode_number)
                    # print(count)
                    count += 1
                    sum_rew += self.reward_sum
                    if self.episode_number % (self.batch_size-1) == 0:
                        # print(sum_rew/(self.batch_size-1))
                        rew_sample.append(sum_rew/(self.batch_size-1))
                    self.reward_sum = 0
                    #Calculate fisher matrix over episode and incrementally average
                    # F_matrix += (F-F_matrix)/self.episode_number
                    # F = torch.zeros((self.H*self.D+self.H*2,self.H*self.D+self.H*2))
                    # print(self.reward_sum)
                    # print("Done")
        #             num_steps = 0
        #             #update episode number
                    self.episode_number += 1
        #             # Stack log_probabilities of chosen action of the epsiode(i.e all steps of an episode)
        #             eplogp = torch.vstack(self.logps)
        #             # Stack rewards got after running an episode
        #             epr = torch.vstack(self.drs)
        #             # free the lists
        #             self.logps, self.drs = [], []
        #             # compute the discounted reward backwards through time
        #             discounted_epr = self.discount_rewards(epr)
        #             # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        #             discounted_epr -= torch.mean(discounted_epr)
        #             discounted_epr /= torch.std(discounted_epr)
        #             discounted_epr = discounted_epr.detach()
        #             # modulate the gradient with advantage (PG magic happens right here.)
        #             eplogp *= discounted_epr
        #             # Calculate loss
        #             loss = eplogp.sum()
        #             # Zero the grads so autograd works more than once
        #             self.nn_model.zero_grad(set_to_none=True)
        #             # Backward prop using pytorch's autograd
        #             loss.backward()
        #             # Store the gradients of each layer in the temporary dictionary
        #             tmp_dic['W1'] = self.nn_model.linear1.weight.grad
        #             tmp_dic['W2'] = self.nn_model.linear2.weight.grad
        #             for k in self.model:
        #                 self.grad_buffer[k] += tmp_dic[k]  # accumulate grad over batch

        #             # perform rmsprop parameter update every batch_size episodes
        #             if self.episode_number % self.batch_size == 0:
        #                 for k, v in self.model.items():
        #                     g = self.grad_buffer[k]  # gradient
        #                     toReturn.append(g / (self.batch_size - 1))
        #                     self.grad_buffer[k] = torch.zeros_like(v)  # reset batch gradient buffer
        #             # boring book-keeping
        #             toReturnReward = toReturnReward + self.reward_sum
        #             if self.episode_number % 100 == 0:
        #                 pickle.dump(self.model, open('gw.p', 'wb'))
        #             self.reward_sum = 0
        #             self.observation = self.env.reset()  # reset env
        #             # tmp_dic['W1'] = torch.zeros(self.H, self.D)
        #             # tmp_dic['W2'] = torch.zeros(self.output_size, self.H)
        #     a = list(toReturn[0].ravel())
        #     b = list(toReturn[1].ravel())
        #     #convert list into tensor (contains all grads over one dim)
        #     store = torch.tensor([a + b])
        #     #calculate natural gradient
        #     # F_matrix += 0.01*torch.eye(30)
        #     # F_inv_matrix = torch.inverse(F_matrix)
        #     # print(F_inv_matrix,store)
        #     # store = torch.transpose(torch.matmul(F_inv_matrix,torch.transpose(store,0,1).float()),0,1)
        #     m[count, :] = store
        #     count += 1
        #     l.append(toReturnReward / (self.batch_size))

        # l = torch.tensor(l)
        # return l, m
        return torch.tensor(rew_sample)

    def evaluate_true(self, x):
        """ This for ActorCritic Algorithm
        """
        # totalCost, totalCostGradient = self.computeCost(x)
        totalCost= self.computeCost(x)
        # a = list(totalCostGradient[0].ravel())
        # b = list(totalCostGradient[1].ravel())
        # params = [totalCost] + a + b
        # return totalCost, totalCostGradient
        return totalCost

    def evaluate(self, x):
        return self.evaluate_true(x)

class NN_Model_cp(torch.nn.Module):

    def __init__(self,dic):
        super(NN_Model_cp, self).__init__()
        self.linear1 = torch.nn.Linear(4, 10,bias = False)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 2, bias = False)
#         with torch.no_grad():
        self.linear1.weight = torch.nn.Parameter(data = dic['W1'].float())
        self.linear2.weight = torch.nn.Parameter(data = dic['W2'].float())
#         torch.nn.init.xavier_normal_(self.linear1.weaight, gain=1.0)
#         torch.nn.init.xavier_normal_(self.linear2.weight, gain=1.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class cart_pole(ObjectiveFunction):
    def __init__(self, dims, noise_mean=None, noise_variance=None,
                 random_state=None, negate=False):

        dims = 60
        low = torch.tensor([-10] * dims)
        high = torch.tensor([10] * dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self._observations = []

        # hyperparameters
        self.H = 10  # number of hidden layer neurons
        self.output_size = 2
        self.batch_size = 1000  # every how many episodes to do a param update?
        self.gamma = 1.0  # discount factor for reward
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
        self.resume = False  # resume from previous checkpoint?
        self.render = False

        # model initialization
        self.D = 4  # input dimensionality: 1x8 vector
        if self.resume:
            self.model = pickle.load(open('lunar_lander.p', 'rb'))
        else:
            self.model = {}
            self.model['W1'] = torch.randn(self.H, self.D) / torch.sqrt(torch.tensor(self.D))  # "Xavier" initialization
            self.model['W2'] = torch.randn(self.output_size,self.H) / torch.sqrt(torch.tensor(self.H))

        self.grad_buffer = {k: torch.zeros_like(v) for k, v in self.model.items()}  # update buffers that add up gradients over a batch
        self.rmsprop_cache = {k: torch.zeros_like(v) for k, v in self.model.items()}  # rmsprop memory

        self.env = gym.make("CartPole-v0")
        # self.env._max_episode_steps = 500
        self.observation = self.env.reset()

        self.logps, self.drs = [], []
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0
        self.nn_model = NN_Model_cp(self.model)

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(dim=0))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def computeCost(self, model):
        l = []
        m = torch.zeros_like(model)
        count = 0
        for samples in model:
            # Count number of episodes
            self.episode_number += 1
            toReturn = []
            toReturnReward = 0.0
            # Reshape the model we get according to self.model
            self.model['W1'] = torch.reshape(samples[0:self.H * self.D], (self.H, self.D))
            self.model['W2'] = torch.reshape(samples[self.H * self.D:], (self.output_size, self.H))
            tmp_dic = {}
            tmp_dic['W1'] = torch.zeros(self.H, self.D)
            tmp_dic['W2'] = torch.zeros(self.output_size, self.H)
            self.nn_model = NN_Model_cp(self.model)
            # print(self.nn_model.linear1.weight,self.nn_model.linear2.weight)
            num_steps = 0
            # Start
            while(self.episode_number % self.batch_size != 0):
                num_steps += 1
                if self.render:
                    self.env.render()
                x = torch.tensor(self.observation, requires_grad=True)
                # forward the policy network and sample an action from the returned probability
                apred = self.nn_model(x)
                # Calculate probability of each action using softmax
                aprob = torch.nn.functional.softmax(apred, dim=-1)
                # sample action from Categorical distribution
                dist = torch.distributions.Categorical(aprob)
                action = dist.sample()
                # store the log_probability of our action
                self.logps.append(dist.log_prob(action))
                # step the environment and get new measurements
                self.observation, reward, done, info = self.env.step(action.item())
                # Accumulate reward over episode
                self.reward_sum = self.reward_sum + reward
                # Store reward recieved at each step
                self.drs.append(torch.tensor(reward))
                if done:  # an episode finished
                    num_steps = 0
                    #update episode number
                    self.episode_number += 1
                    # Stack log_probabilities of chosen action of the epsiode(i.e all steps of an episode)
                    eplogp = torch.vstack(self.logps)
                    # Stack rewards got after running an episode
                    epr = torch.vstack(self.drs)
                    # free the lists
                    self.logps, self.drs = [], []
                    # compute the discounted reward backwards through time
                    discounted_epr = self.discount_rewards(epr)
                    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                    discounted_epr -= torch.mean(discounted_epr)
                    discounted_epr /= torch.std(discounted_epr)
                    discounted_epr = discounted_epr.detach()
                    # modulate the gradient with advantage (PG magic happens right here.)
                    eplogp *= discounted_epr
                    # Calculate loss
                    loss = eplogp.sum()
                    # Zero the grads so autograd works more than once
                    self.nn_model.zero_grad(set_to_none=True)
                    # Backward prop using pytorch's autograd
                    loss.backward()
                    # Store the gradients of each layer in the temporary dictionary
                    tmp_dic['W1'] = self.nn_model.linear1.weight.grad
                    tmp_dic['W2'] = self.nn_model.linear2.weight.grad
                    for k in self.model:
                        self.grad_buffer[k] += tmp_dic[k]  # accumulate grad over batch

                    # perform rmsprop parameter update every batch_size episodes
                    if self.episode_number % self.batch_size == 0:
                        for k, v in self.model.items():
                            g = self.grad_buffer[k]  # gradient
                            toReturn.append(g / (self.batch_size - 1))
                            self.grad_buffer[k] = torch.zeros_like(v)  # reset batch gradient buffer
                    # boring book-keeping
                    toReturnReward = toReturnReward + self.reward_sum
                    if self.episode_number % 100 == 0:
                        pickle.dump(self.model, open('lunar_lander.p', 'wb'))
                    self.reward_sum = 0
                    self.observation = self.env.reset()  # reset env
                    # tmp_dic['W1'] = torch.zeros(self.H, self.D)
                    # tmp_dic['W2'] = torch.zeros(self.output_size, self.H)
            a = list(toReturn[0].ravel())
            b = list(toReturn[1].ravel())
            #convert list into tensor (contains all grads over one dim)
            store = torch.tensor([a + b])
            m[count, :] = store
            count += 1
            l.append(toReturnReward / (self.batch_size))

        l = torch.tensor(l)
        return l, m

    def evaluate_true(self, X):
        """ This for ActorCritic Algorithm
        """
        totalCost, totalCostGradient = self.computeCost(X)
        # a = list(totalCostGradient[0].ravel())
        # b = list(totalCostGradient[1].ravel())
        # params = [totalCost] + a + b
        return totalCost, totalCostGradient

    def evaluate(self, X):
        return self.evaluate_true(X)

class IllustrationND(ObjectiveFunction):
    '''
    '''
    def __init__(self, dims, noise_mean=None, noise_variance=None):

        low = torch.tensor([0.1]*dims)
        high = torch.tensor([2.0]*dims)
        
        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=True,
        )
        self.true_opt_value = None

    def evaluate_true(self, X, theta):
        sigma = 0.1
        n_trials = 100
        temperature = 0.05
        n_model_candidates = 2

        grads = []
        f_x = []
        for i in range(len(X)):
            grads.append([])
            f_x.append([])
            l = X[i].detach().clone().requires_grad_(True)

            for j in range(n_trials):
                # perturb the model parameters
                theta_list = torch.cat([(theta[i] + sigma * torch.randn_like(theta[i])).unsqueeze(0) 
                    for _ in range(n_model_candidates)], dim=0)

                # calculate loss of the model copies
                loss_list = self.trn_loss(theta_list, l)

                # calculate the model copies weights
                weights = torch.softmax(-loss_list / temperature, 0)

                # merge the model copies
                temp = weights * theta_list.T
                theta_updated = torch.sum(temp.T, dim=0)

                # evaluate the merged model on validation
                loss_val = self.val_loss(theta_updated)

                # calculate the hypergradient
                hyper_grad = torch.autograd.grad(loss_val, l)[0]
                grads[-1].append(hyper_grad.detach())
                f_x[-1].append(loss_val)

            grads[-1] = torch.stack(grads[-1])
            f_x[-1] = torch.stack(f_x[-1])

        grad = torch.stack(grads)
        f_x = torch.stack(f_x)
        f_x = f_x.mean(dim=1)
        grad = grad.mean(dim=1)

        self.f_x = -f_x
        self.grads = -grad

        return self.f_x.squeeze()

    def backward(self):
        return self.grads.detach().clone()

    def trn_loss(self, x, l):
        return torch.sum((x-1)**2, dim=1) + torch.sum(l*(x**2), dim=1)

    def val_loss(self, x):
        return torch.sum((x-0.5)**2)

    def find_optimal_theta(self, X):
        X = X.detach().clone()
        X.requires_grad = False
        theta = torch.randn_like(X, requires_grad=True, dtype=self.dtype)
        optimizer = torch.optim.Adam([theta], lr=1e-2)

        batch = X.ndimension() > 1

        external_grad = torch.tensor([1.]*X.shape[0]) if batch else None

        for i in range(1000):
            optimizer.zero_grad()
            func_value = self.train_loss(theta, X)
            func_value.backward(gradient=external_grad)
            # print(theta.grad)
            optimizer.step()

        return theta.detach().clone()

    def train_loss(self, x, l):
        batch = x.ndimension() > 1
        x = x if batch else x.unsqueeze(0)

        trn_loss = torch.sum((x-1)**2, dim=1) + torch.sum(l*(x**2),     dim=1)

        return trn_loss if batch else trn_loss.squeeze()

class NN_Model(torch.nn.Module):

    def __init__(self,dic):
        super(NN_Model, self).__init__()
        self.linear1 = torch.nn.Linear(2, 5, bias=False)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(5, 3, bias=False)

        self.linear1.weight = torch.nn.Parameter(dic['W1'].float())
        self.linear2.weight = torch.nn.Parameter(dic['W2'].float())

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class pongChangeMountainCar(ObjectiveFunction):
    def __init__(self, dims, noise_mean=None, noise_variance=None,
                 random_state=None, negate=False):

        dims = 25
        low = torch.tensor([-10] * dims)
        high = torch.tensor([10] * dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self._observations = []

        # hyperparameters
        self.H = 5  # number of hidden layer neurons
        self.output_size = 3
        self.batch_size = 1000  # every how many episodes to do a param update?
        self.gamma = 1.0  # discount factor for reward
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
        self.resume = False  # resume from previous checkpoint?
        self.render = False

        # model initialization
        self.D = 2  # input dimensionality: 1x2 vector
        if self.resume:
            self.model = pickle.load(open('MountainCarFinalBayesian.p', 'rb'))
        else:
            self.model = {}
            self.model['W1'] = torch.randn(self.H, self.D) / torch.sqrt(torch.tensor(self.D))  # "Xavier" initialization
            self.model['W2'] = torch.randn(self.output_size,self.H) / torch.sqrt(torch.tensor(self.H))

        self.grad_buffer = {k: torch.zeros_like(v) for k, v in self.model.items()}  # update buffers that add up gradients over a batch
        self.rmsprop_cache = {k: torch.zeros_like(v) for k, v in self.model.items()}  # rmsprop memory

        self.env = gym.make("MountainCar-v0")
        self.env._max_episode_steps = 1000
        self.observation = self.env.reset()

        self.logps, self.drs = [], []
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0
        self.nn_model = NN_Model(self.model)

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(dim=0))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def computeCost(self, model):
        l = []
        m = torch.zeros_like(model)
        count = 0
        for samples in model:
            # Count number of episodes
            self.episode_number += 1
            toReturn = []
            toReturnReward = 0.0
            # Reshape the model we get according to self.model
            self.model['W1'] = torch.reshape(samples[0:self.H * self.D], (self.H, self.D))
            self.model['W2'] = torch.reshape(samples[self.H * self.D:], (self.output_size, self.H))
            # Dictionary to store results temporarily
            tmp_dic = {}
            tmp_dic['W1'] = torch.zeros(self.H, self.D)
            tmp_dic['W2'] = torch.zeros(self.output_size, self.H)
            self.nn_model = NN_Model(self.model)
            # print(self.nn_model.linear1.weight,self.nn_model.linear2.weight)
            # Start
            while(self.episode_number % self.batch_size != 0):
                if self.render:
                    self.env.render()
                x = torch.tensor(self.observation, requires_grad=True)
                # forward the policy network and sample an action from the returned probability
                apred = self.nn_model(x)
                # Calculate probability of each action using softmax
                aprob = torch.nn.functional.softmax(apred, dim=-1)
                # sample action from Categorical distribution
                dist = torch.distributions.Categorical(aprob)
                action = dist.sample()
                # store the log_probability of our action
                self.logps.append(dist.log_prob(action))
                # step the environment and get new measurements
                self.observation, reward, done, info = self.env.step(action.item())
                # Accumulate reward over episode
                self.reward_sum = self.reward_sum - reward
                # Store reward recieved at each step
                self.drs.append(torch.tensor(-reward))
                if done:  # an episode finished
                    self.episode_number += 1
                    # Stack log_probabilities of chosen action of the epsiode(i.e all steps of an episode)
                    eplogp = torch.vstack(self.logps)
                    # Stack rewards got after running an episode
                    epr = torch.vstack(self.drs)
                    # free the lists
                    self.logps, self.drs = [], []
                    # compute the discounted reward backwards through time
                    discounted_epr = self.discount_rewards(epr)
                    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                    discounted_epr -= torch.mean(discounted_epr)
                    discounted_epr /= torch.std(discounted_epr)
                    discounted_epr = discounted_epr.detach()
                    # modulate the gradient with advantage (PG magic happens right here.)
                    eplogp *= discounted_epr
                    # Calculate loss
                    loss = eplogp.sum()
                    # Zero the grads so autograd works more than once
                    self.nn_model.zero_grad(set_to_none=True)
                    # Backward prop using pytorch's autograd
                    loss.backward()
                    # Store the gradients of each layer in the temporary dictionary
                    tmp_dic['W1'] = self.nn_model.linear1.weight.grad
                    tmp_dic['W2'] = self.nn_model.linear2.weight.grad
                    for k in self.model:
                        self.grad_buffer[k] += tmp_dic[k]  # accumulate grad over batch

                    # perform rmsprop parameter update every batch_size episodes
                    if self.episode_number % self.batch_size == 0:
                        for k, v in self.model.items():
                            g = self.grad_buffer[k]  # gradient
                            toReturn.append(g / (self.batch_size - 1))
                            self.grad_buffer[k] = torch.zeros_like(v)  # reset batch gradient buffer
                    # boring book-keeping
                    toReturnReward = toReturnReward + self.reward_sum
                    if self.episode_number % 100 == 0:
                        pickle.dump(self.model, open('MountainCarFinalBayesian.p', 'wb'))
                    self.reward_sum = 0
                    self.observation = self.env.reset()  # reset env
                    # tmp_dic['W1'] = torch.zeros(self.H, self.D)
                    # tmp_dic['W2'] = torch.zeros(self.output_size, self.H)
            a = list(toReturn[0].ravel())
            b = list(toReturn[1].ravel())
            m[count, :] = torch.tensor(a + b)
            count += 1
            l.append(toReturnReward / (self.batch_size))
        l = torch.tensor(l)
        return l, m

    def evaluate_true(self, x):
        """ This for ActorCritic Algorithm
        """
        totalCost, totalCostGradient = self.computeCost(x)
        # a = list(totalCostGradient[0].ravel())
        # b = list(totalCostGradient[1].ravel())
        # params = [totalCost] + a + b
        return totalCost, totalCostGradient

    def evaluate(self, x):
        return self.evaluate_true(x)

class NN_Model_ll(torch.nn.Module):

    def __init__(self,dic):
        super(NN_Model_ll, self).__init__()
        self.linear1 = torch.nn.Linear(8, 16,bias = False)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 4, bias = False)
        self.linear1.weight = torch.nn.Parameter(data = dic['W1'].float())
        self.linear2.weight = torch.nn.Parameter(data = dic['W2'].float())

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class lunar_lander(ObjectiveFunction):
    def __init__(self, dims, noise_mean=None, noise_variance=None,
                 random_state=None, negate=False):

        dims = 192
        low = torch.tensor([-10] * dims)
        high = torch.tensor([10] * dims)

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        self._observations = []

        # hyperparameters
        self.H = 16  # number of hidden layer neurons
        self.output_size = 4
        self.batch_size = 1000  # every how many episodes to do a param update?
        self.gamma = 1.0  # discount factor for reward
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
        self.resume = False  # resume from previous checkpoint?
        self.render = False

        # model initialization
        self.D = 8  # input dimensionality: 1x8 vector
        if self.resume:
            self.model = pickle.load(open('lunar_lander.p', 'rb'))
        else:
            self.model = {}
            self.model['W1'] = torch.randn(self.H, self.D) / torch.sqrt(torch.tensor(self.D))  # "Xavier" initialization
            self.model['W2'] = torch.randn(self.output_size,self.H) / torch.sqrt(torch.tensor(self.H))

        self.grad_buffer = {k: torch.zeros_like(v) for k, v in self.model.items()}  # update buffers that add up gradients over a batch
        self.rmsprop_cache = {k: torch.zeros_like(v) for k, v in self.model.items()}  # rmsprop memory

        self.env = gym.make("LunarLander-v2")
        # self.env._max_episode_steps = 500
        self.observation = self.env.reset()

        self.logps, self.drs = [], []
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0
        self.nn_model = NN_Model_ll(self.model)

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(dim=0))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def computeCost(self, model):
        l = []
        m = torch.zeros_like(model)
        count = 0
        for samples in model:
            # Count number of episodes
            self.episode_number += 1
            toReturn = []
            toReturnReward = 0.0
            # Reshape the model we get according to self.model
            self.model['W1'] = torch.reshape(samples[0:self.H * self.D], (self.H, self.D))
            self.model['W2'] = torch.reshape(samples[self.H * self.D:], (self.output_size, self.H))
            # Dictionary to store results temporarily
            tmp_dic = {}
            tmp_dic['W1'] = torch.zeros(self.H, self.D)
            tmp_dic['W2'] = torch.zeros(self.output_size, self.H)
            self.nn_model = NN_Model_ll(self.model)
            # print(self.nn_model.linear1.weight,self.nn_model.linear2.weight)
            num_steps = 0
            # Start
            while(self.episode_number % self.batch_size != 0):
                num_steps += 1
                if self.render:
                    self.env.render()
                x = torch.tensor(self.observation, requires_grad=True)
                # forward the policy network and sample an action from the returned probability
                apred = self.nn_model(x)
                # Calculate probability of each action using softmax
                aprob = torch.nn.functional.softmax(apred, dim=-1)
                # sample action from Categorical distribution
                dist = torch.distributions.Categorical(aprob)
                action = dist.sample()
                # store the log_probability of our action
                self.logps.append(dist.log_prob(action))
                # step the environment and get new measurements
                self.observation, reward, done, info = self.env.step(action.item())
                # Accumulate reward over episode
                self.reward_sum = self.reward_sum + reward
                # Store reward recieved at each step
                self.drs.append(torch.tensor(reward))
                if done:  # an episode finished
                    num_steps = 0
                    self.episode_number += 1
                    # Stack log_probabilities of chosen action of the epsiode(i.e all steps of an episode)
                    eplogp = torch.vstack(self.logps)
                    # Stack rewards got after running an episode
                    epr = torch.vstack(self.drs)
                    # free the lists
                    self.logps, self.drs = [], []
                    # compute the discounted reward backwards through time
                    discounted_epr = self.discount_rewards(epr)
                    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                    discounted_epr -= torch.mean(discounted_epr)
                    discounted_epr /= torch.std(discounted_epr)
                    discounted_epr = discounted_epr.detach()
                    # modulate the gradient with advantage (PG magic happens right here.)
                    eplogp *= discounted_epr
                    # Calculate loss
                    loss = eplogp.sum()
                    # Zero the grads so autograd works more than once
                    self.nn_model.zero_grad(set_to_none=True)
                    # Backward prop using pytorch's autograd
                    loss.backward()
                    # Store the gradients of each layer in the temporary dictionary
                    tmp_dic['W1'] = self.nn_model.linear1.weight.grad
                    tmp_dic['W2'] = self.nn_model.linear2.weight.grad
                    for k in self.model:
                        self.grad_buffer[k] += tmp_dic[k]  # accumulate grad over batch

                    # perform rmsprop parameter update every batch_size episodes
                    if self.episode_number % self.batch_size == 0:
                        for k, v in self.model.items():
                            g = self.grad_buffer[k]  # gradient
                            toReturn.append(g / (self.batch_size - 1))
                            self.grad_buffer[k] = torch.zeros_like(v)  # reset batch gradient buffer
                    # boring book-keeping
                    toReturnReward = toReturnReward + self.reward_sum
                    if self.episode_number % 100 == 0:
                        pickle.dump(self.model, open('lunar_lander.p', 'wb'))
                    self.reward_sum = 0
                    self.observation = self.env.reset()  # reset env
                    # tmp_dic['W1'] = torch.zeros(self.H, self.D)
                    # tmp_dic['W2'] = torch.zeros(self.output_size, self.H)
            a = list(toReturn[0].ravel())
            b = list(toReturn[1].ravel())
            #convert list into tensor (contains all grads over one dim)
            store = torch.tensor([a + b])
            m[count, :] = store
            count += 1
            l.append(toReturnReward / (self.batch_size))
        l = torch.tensor(l)
        return l, m

    def evaluate_true(self, x):
        """ This for ActorCritic Algorithm
        """
        totalCost, totalCostGradient = self.computeCost(x)
        # a = list(totalCostGradient[0].ravel())
        # b = list(totalCostGradient[1].ravel())
        # params = [totalCost] + a + b
        return totalCost, totalCostGradient

    def evaluate(self, x):
        return self.evaluate_true(x)


class RotationTransformation(ObjectiveFunction):
    '''
    '''
    def __init__(self):
        
        torch.pi = torch.acos(torch.zeros(1)).item() * 2

        dims=1
        low = torch.tensor([0.0])
        high = torch.tensor([2*torch.pi])

        super().__init__(
            dims,
            low,
            high,
            noise_mean=0,
            noise_variance=None,
            negate=True
        )

        self.epochs = 5
        # self.device = get_device()
        # self.dataloaders = get_dataloaders()
        # self.feature_transformer = RotTransformer(device=self.device).to(device=self.device)
        

    def evaluate_true(self, X=None):
        sigma = 0.001
        temperature = 0.05
        n_model_candidates = 2

        criterion = nn.CrossEntropyLoss().to(device=self.device)

        self.f_x = 0

        for i, batch in enumerate(self.dataloaders[2]):
            (input_rot, target_rot) = batch
            input_rot = input_rot.to(device=self.device)
            target_rot = target_rot.to(device=self.device)

            model_parameter = [i.detach() for i in get_func_params(self.lenet)]
            input_transformed = self.feature_transformer(self.input_)

            theta_list = [[j + sigma * torch.sign(torch.randn_like(j)) for j in model_parameter] for i in range(n_model_candidates)]
            pred_list = [self.model_patched(input_transformed, params=theta) for theta in theta_list]
            loss_list = [criterion(pred, self.target) for pred in pred_list]
            baseline_loss = criterion(self.model_patched(input_transformed, params=model_parameter), self.target)

            # calculate weights for the different model copies
            weights = torch.softmax(-torch.stack(loss_list)/temperature, 0)

            # merge the model copies
            theta_updated = [sum(map(mul, theta, weights)) for theta in zip(*theta_list)]
            pred_rot = self.model_patched(input_rot, params=theta_updated)

            self.f_x += -1*criterion(pred_rot, target_rot)

        return (self.f_x/len(self.dataloaders[2])).cpu()

    def backward(self, noise=False):
        self.feature_transformer.zero_grad()
        self.f_x.backward()

        counter = 0
        for i in self.feature_transformer.parameters():
            counter += 1

        if counter > 1:
            print("Length of parameters: ", counter)
            print("Parameters: ", self.feature_transformer.parameters())
            raise Exception("More than one parameter")

        self.grads = next(self.feature_transformer.parameters()).grad

        return (self.grads.detach().clone()/len(self.dataloaders[2])).cpu()

    def find_optimal_theta(self, X):
        self._reset(X)

        optimizer = torch.optim.Adam(self.lenet.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss().to(device=self.device)

        for epoch in range(self.epochs):
            loaders = self.dataloaders[0]

            for i, batch in enumerate(loaders):
                (input_, target) = batch
                input_ = input_.to(device=self.device)
                target = target.to(device=self.device)

                logits = self.lenet(self.feature_transformer(input_))
                loss = criterion(logits, target)
                # with torch.no_grad():
                #     self.losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.input_ = input_
        self.target = target

    def _reset(self, X):
        with torch.no_grad():
            for p in self.feature_transformer.parameters():
                p.copy_(X.squeeze(0))
            for p in self.feature_transformer.parameters():
                print(p)

        self.lenet = LeNet().to(device=self.device)
        self.model_patched = make_functional(self.lenet)

if __name__ == "__main__":
    '''
        For testing purposes.
    '''
    le_branke = LeBranke()
    assert round(le_branke.evaluate_true(torch.tensor([[1.87334]])).item(), 4) == 1.4360
    assert round(le_branke.evaluate_true(torch.tensor([[1.23223]])).item(), 5) == 1.11425
    assert round(le_branke.evaluate_true(torch.tensor([[1.58504]])).item(), 5) == -1.29155
    assert round(le_branke.evaluate_true(torch.tensor([[0.734545]])).item(), 6) == -0.860584
