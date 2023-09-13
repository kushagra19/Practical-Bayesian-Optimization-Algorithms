import torch
from tqdm import tqdm
from utils import generate_initial_data, optimize_acq_func_and_get_candidates, get_next_query_point
from GaussianProcess import get_and_fit_simple_custom_gp
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.utils.transforms import unnormalize, standardize, normalize

class BO:
    '''
        This class implements the Bayesian Optimisation loop.
    '''
    def __init__(self, obj_fn, dtype, acq_func, grad_acq=None, init_examples=5,
        order=0, budget=20, query_point_selection="convex", temp_schedule=True,
        num_restarts=2, raw_samples=32, num_fantasies=128,
        grad_acq_name="SumGradientAcquisitionFunction"):
        '''
            Arguments:
            ---------
                - obj_fn: (Instance of ObjectiveFunction class) The 
                    function which needs to be optimized using Bayesian
                    Optimisation.
                - dtype: A PyTorch data type
                - acq_func: (String) Acquisiton Function to be used for
                    the objective function
                - grad_acq: Acquisiton Function to be used for the 
                    gradient function. If order=0 i.e. Zero order BO is
                    performed then this argument will be ignored.
                - init_examples: (int) Number of points where the 
                    objective function is to be queried.
                - order: (int) If 0 then perform 0 order BO else perform
                    1st order BO
                - budget: (int) Number of times the objective function
                    can be evaluated
                - query_point_selection: (string) If set to "convex" 
                    then convex combination used for selecting the next
                    query point. If set to "maximum" then maximum 
                    significance method used for selecting the next 
                    query point.
                - temp_schedule:
                - num_restarts: Number of restarts for optimizing the 
                    acquisition function. Refer to BoTorch docs for
                    `optimize_acqf` for more information.
                - raw_samples: Number of raw samples for optimizing the
                    acquisition function. Refer to BoTorch docs for
                    `optimize_acqf` for more information.
                - grad_acq_name: Name of the gradient acquisition 
                    function. Can take values: 
                    [SumGradientAcquisitionFunction, 
                    PrabuAcquistionFunction]
                - num_fantasies: To be used in Knowledge Gradient 
                    acquisition function.
        '''
        
        self.obj_fn = obj_fn
        self.dtype = dtype
        self.init_examples = init_examples
        self.acq_func = acq_func
        self.grad_acq = grad_acq
        self.order = order
        self.budget = budget
        self.query_point_selection = query_point_selection
        self.temp_schedule = temp_schedule
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.grad_acq_name = grad_acq_name
        self.num_fantasies = num_fantasies
   
    def optimize(self):
        '''
            This function performs the optimization of the Objective
            Function.
            
            Returns:
            -------
                - X (PyTorch tensor): Containing the X's where the 
                    objective function was queried. Shape: 
                    ((budget + init_examples) x d)
                - y (PyTorch tensor): Containing the value of the 
                    objective function at the corresponding X's.
                    Shape: ((budget + init_examples) x 1)
                - grads (PyTorch tensor): Containing the gradient at the
                    corresponding X's of the objective function. Shape:
                    ((budget + init_examples) x d)
        '''
        
        # Generate initial data for GP fiiting
        self.X, self.y, self.grads = generate_initial_data(
            self.obj_fn,
            n=self.init_examples,
            dtype=self.dtype,
            order=self.order
        )

        if self.grads is not None:
            min_init_grads,_ = torch.min(torch.abs(self.grads),0)
        # print("min_init_grads:",min_init_grads)
        grad_cand = []

        for i in tqdm(range(self.budget)):
            # Fit GP
            gps = get_and_fit_simple_custom_gp(
                self.X.detach().clone(), 
                self.y.detach().clone(), 
                self.grads.detach().clone() if self.order else self.grads
            )
            obj_fn_gp, grad_gps = gps
            
            # Optimize acquisition function and get next query point
            original_bounds = torch.cat([self.obj_fn.low.unsqueeze(0),
                            self.obj_fn.high.unsqueeze(0)]).type(self.dtype)
            # Normalize the bounds
            norm_bounds = torch.stack([self.X.min(dim=0)[0],self.X.max(dim=0)[0]])
            bounds = normalize(original_bounds, bounds=norm_bounds)
            
            if self.acq_func == 'EI':
                best_f = torch.max(standardize(self.y))
                # print(best_f,self.y)
                acq_func = ExpectedImprovement(obj_fn_gp, best_f=best_f)
            elif self.acq_func == 'KG':
                print("Using KG Acquisition Function")
                acq_func = qKnowledgeGradient(obj_fn_gp, 
                    num_fantasies=self.num_fantasies)

            if self.grads is not None:
                eps_values = torch.abs((0.75*min_init_grads)/((i+1)**0.5))
            else:
                eps_values = 0
            # print("eps_values:",eps_values)
            candidates = optimize_acq_func_and_get_candidates(
                acq_func=acq_func,
                grad_acq=self.grad_acq,
                bounds=bounds,
                obj_fn_gp = obj_fn_gp,
                best_f = best_f,
                eps_values = eps_values,
                grad_gps=grad_gps,
                order=self.order,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                bud_i = i+1,
                grad_acq_name=self.grad_acq_name
            )
        
            if self.order:
                # temp = exp_temp_schedule(i) if self.temp_schedule else 1
                best_candidate = get_next_query_point(
                    obj_fn_gp,
                    grad_gps,
                    candidates,
                    grad_cand,
                    self.query_point_selection,
                    temp  
                )
            else:
                best_candidate = candidates[0][0]
            
            # Unnormalize the best candidate
            best_candidate = unnormalize(best_candidate, bounds=norm_bounds)

            # print(best_candidate)
            
            # Function and gradient evaluation at the new point
            y_new = self.obj_fn.forward(best_candidate).unsqueeze(0).detach().clone()
            if self.order:
                grad_new = self.obj_fn.backward().detach().clone()
            best_candidate = best_candidate.unsqueeze(0)
            
            # Update X, y and grads
            self.X = torch.cat([self.X, best_candidate])
            self.y = torch.cat([self.y, y_new])
            if self.order:
                self.grads = torch.cat([self.grads, grad_new])
            
        return self.X, self.y,self.grads
        