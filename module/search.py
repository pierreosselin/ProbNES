from evotorch import Problem
from evotorch.algorithms.distributed.gaussian import GaussianSearchAlgorithm, SNES
from typing import Optional, Dict, Any
from evotorch.tools import RealOrVector
from evotorch.core import SolutionBatch
from botorch.utils.transforms import standardize, normalize, unnormalize
from evotorch.distributions import (
    Distribution,
    ExpGaussian,
    ExpSeparableGaussian,
    SeparableGaussian,
    SymmetricSeparableGaussian,
)
import torch
from torch.distributions import MultivariateNormal
from .BASQ._basq_search import BASQ
# TODO Data currently saved both in BO and in search
# TODO Initial distribution should be mu = 0 for fairer comparison

class NESWSABI(SNES):

    DISTRIBUTION_TYPE = ExpSeparableGaussian
    DISTRIBUTION_PARAMS = None

    def __init__(
        self,
        problem: Problem,
        *,
        stdev_init: Optional[RealOrVector] = None,
        radius_init: Optional[RealOrVector] = None,
        popsize: Optional[int] = None,
        center_learning_rate: Optional[float] = None,
        stdev_learning_rate: Optional[float] = None,
        scale_learning_rate: bool = True,
        num_interactions: Optional[int] = None,
        popsize_max: Optional[int] = None,
        optimizer=None,
        optimizer_config: Optional[dict] = None,
        ranking_method: Optional[str] = "nes",
        center_init: Optional[RealOrVector] = None,
        stdev_min: Optional[RealOrVector] = None,
        stdev_max: Optional[RealOrVector] = None,
        stdev_max_change: Optional[RealOrVector] = None,
        obj_index: Optional[int] = None,
        distributed: bool = False,
        popsize_weighted_grad_avg: Optional[bool] = None,
        quad_kwargs: Optional[Dict[str, Any]] = None
    ):
        
        super().__init__(
                    problem,
                    popsize=popsize,
                    center_learning_rate=center_learning_rate,
                    stdev_learning_rate=stdev_learning_rate,
                    stdev_init=stdev_init,
                    radius_init=radius_init,
                    popsize_max=popsize_max,
                    num_interactions=num_interactions,
                    optimizer=optimizer,
                    optimizer_config=optimizer_config,
                    ranking_method=ranking_method,
                    center_init=center_init,
                    stdev_min=stdev_min,
                    stdev_max=stdev_max,
                    stdev_max_change=stdev_max_change,
                    obj_index=obj_index,
                    distributed=distributed,
                    popsize_weighted_grad_avg=popsize_weighted_grad_avg,
                )
        # We kee track of the population sampled
        self.quad_kwargs = quad_kwargs
        self.train_x = torch.tensor([], device = self.problem.device, dtype = self.problem.dtype)
        self.train_y = torch.tensor([], device = self.problem.device, dtype = self.problem.dtype)
        self.weights = torch.tensor([], device = self.problem.device, dtype = self.problem.dtype)
        self._distribution.mu = torch.zeros(self.problem.solution_length, device = self.problem.device, dtype = self.problem.dtype)

    def _step_non_distributed(self):
        # First, we define an inner function which fills the current population by sampling from the distribution.
        def fill_and_eval_pop():
            # This inner function is responsible for filling the main population with samples
            # and evaluate them.
            ## Default option here
            if self._num_interactions is None:
                # If num_interactions is configured as None, this means that we are not going to adapt
                # the population size according to the number of simulation interactions reported
                # by the problem object.

                # We first make sure that the population (which is to be of constant size, since we are
                # not in the adaptive population size mode) is allocated.
                
                # Now, we do in-place sampling on the population
                self._population = SolutionBatch(self.problem, popsize=self._popsize, device=self._distribution.device, empty=True)
                                
                if not self._first_iter:
                    prior = MultivariateNormal(self._distribution.mu, torch.diag(self._distribution.sigma))
                    true_likelihood, device, dtype = self.problem._objective_func, self.problem.device, self.problem.dtype
                    self.quad = BASQ(
                        self.train_x,  # initial locations
                        standardize(self.train_y),  # initial observations
                        prior,  # Gaussian prior distribution
                        true_likelihood,  # true likelihood to be estimated
                        device,  # cpu or cuda
                        dtype,
                        self.quad_kwargs
                    )
                    # self._population.access_values() to change to quadrature point selection
                    result = self.quad.run(1)
                    self._population.set_values(result[0][0])
                    #self._population.set_evals(result[0][1])
                    
                else:
                    self._distribution.sample(out=self._population.access_values(), generator=self.problem)

                # Finally, here, the solutions are evaluated.
                self.problem.evaluate(self._population)
                
                #Get log prob of values for importance sampling
                m = MultivariateNormal(self._get_mu(), torch.diag(self._get_sigma()))
                # Save data
                self.train_x = torch.cat([self.train_x, self._population.values.detach().clone()])
                self.train_y = torch.cat([self.train_y, self._population.evals.detach().clone().flatten()])
                self.weights = torch.cat([self.weights, m.log_prob(self.population.values.detach().clone())])
                
            else:
                # If num_interactions is not None, then this means that we have a threshold for the number
                # of simulator interactions to reach before declaring the phase of sampling complete.
                # In other words, we have to adapt our population size according to the number of simulator
                # interactions reported by the problem object.

                # The 'total_interaction_count' status reported by the problem object shows the global interaction count.
                # Therefore, to properly count the simulator interactions we made during this generation, we need
                # to get the interaction count before starting our sampling and evaluation operations.
                first_num_interactions = self.problem.status.get("total_interaction_count", 0)

                # We will keep allocating and evaluating new populations until the interaction count threshold is reached.
                # These newly allocated populations will eventually concatenated into one.
                # The not-yet-concatenated populations and the total allocated population size will be stored below:
                populations = []
                total_popsize = 0

                # Below, we repeatedly allocate, sample, and evaluate, until our thresholds are reached.
                while True:
                    # Allocate a new population
                    newpop = SolutionBatch(
                        self.problem,
                        popsize=self._popsize,
                        like=self._population,
                        empty=True,
                    )

                    # Update the total population size
                    total_popsize += len(newpop)

                    # Sample new solutions within the newly allocated population
                    self._distribution.sample(out=newpop.access_values(), generator=self.problem)

                    # Evaluate the new population
                    self.problem.evaluate(newpop)

                    # Add the newly allocated and evaluated population into the populations list
                    populations.append(newpop)

                    # In addition to the num_interactions threshold, we might also have a popsize_max threshold.
                    # We now check this threshold.
                    if (self._popsize_max is not None) and (total_popsize >= self._popsize_max):
                        # If the popsize_max threshold is reached, we leave the loop.
                        break

                    # We now compute the number of interactions we have made during this while loop.
                    interactions_made = self.problem.status["total_interaction_count"] - first_num_interactions

                    if interactions_made > self._num_interactions:
                        # If the number of interactions exceeds our threshold, we leave the loop.
                        break

                # Finally, we concatenate all our populations into one.
                self._population = SolutionBatch.cat(populations)
        
        
        
        if self._first_iter:
            # If we are computing the first generation, we just sample from our distribution and evaluate
            # the solutions.

            fill_and_eval_pop()

            self._first_iter = False
        else:
            # If we are computing next generations, then we need to compute the gradients of the last
            # generation, sample a new population, and evaluate the new population's solutions.
            #samples = self._population.access_values(keep_evals=True)
            samples = self.train_x
            
            #fitnesses = self._population.access_evals()[:, self._obj_index]
            m = MultivariateNormal(self._get_mu(), torch.diag(self._get_sigma()))
            fitnesses = self.train_y * torch.exp(m.log_prob(samples) - self.weights)
            
            obj_sense = self.problem.senses[self._obj_index]
            ranking_method = self._ranking_method
            # Build full samples
            
            gradients = self._distribution.compute_gradients(
                samples, fitnesses, objective_sense=obj_sense, ranking_method=ranking_method
            )
            self._update_distribution(gradients)
            fill_and_eval_pop()
