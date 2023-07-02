import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional, Any, Union, Tuple, Callable, Dict, List
import numpy as np
from scipy.special import erf
from gpytorch.kernels.rbf_kernel import *
from gpytorch.settings import trace_mode
from gpytorch.functions import RBFCovariance



def bounded_bivariate_normal_integral(rho, xl, xu, yl, yu):
  """Computes the bounded bivariate normal integral.
  
  Computes the probability that ``xu >= X >= xl and yu >= Y >= yl`` where X
  and Y are jointly Gaussian random variables, with mean ``[0., 0.]`` and
  covariance matrix ``[[1., rho], [rho, 1.]]``.
  Inputs:
      :rho: Correlation coefficient of the bivariate normal random variable
      :xl, yl: Lower bounds of the integral
      :xu, yu: Upper bounds of the integral
  
  Ported from a Matlab implementation by Alan Genz which, in turn, is based on
  the method described by
      Drezner, Z and G.O. Wesolowsky, (1989),
      On the computation of the bivariate normal inegral,
      Journal of Statist. Comput. Simul. 35, pp. 101-107,
  
  Copyright statement of Alan Genz's version:
  ***************
  Copyright (C) 2013, Alan Genz,  All rights reserved.               
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided the following conditions are met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in 
      the documentation and/or other materials provided with the 
      distribution.
    - The contributor name(s) may not be used to endorse or promote 
      products derived from this software without specific prior 
      written permission.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS 
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""
  
  bvnu = unbounded_bivariate_normal_integral
  p = bvnu(rho, xl, yl) - bvnu(rho, xu, yl) \
      - bvnu(rho, xl, yu) + bvnu(rho, xu, yu)
  return max(0., min(p, 1.))

def unbounded_bivariate_normal_integral(rho, xl, yl):
  """Computes the unbounded bivariate normal integral.
  
  Computes the probability that ``X>=xl and Y>=yl`` where X and Y are jointly
  Gaussian random variables, with mean ``[0., 0.]`` and covariance matrix
  ``[[1., rho], [rho, 1.]]``.
  
  Note: to compute the probability that ``X < xl and Y < yl``, use
  ``unbounded_bivariate_normal_integral(rho, -xl, -yl)``. 
  Inputs:
      :rho: Correlation coefficient of the bivariate normal random variable
      :xl, yl: Lower bounds of the integral
  
  Ported from a Matlab implementation by Alan Genz which, in turn, is based on
  the method described by
      Drezner, Z and G.O. Wesolowsky, (1989),
      On the computation of the bivariate normal inegral,
      Journal of Statist. Comput. Simul. 35, pp. 101-107,
  
  Copyright statement of Alan Genz's version:
  ***************
  Copyright (C) 2013, Alan Genz,  All rights reserved.               
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided the following conditions are met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in 
      the documentation and/or other materials provided with the 
      distribution.
    - The contributor name(s) may not be used to endorse or promote 
      products derived from this software without specific prior 
      written permission.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS 
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""
  
  rho = max(-1., min(1., rho))

  if np.isposinf(xl) or np.isposinf(yl):
    return 0.
  elif np.isneginf(xl):
    return 1. if np.isneginf(yl) else _cdf(-yl)
  elif np.isneginf(yl):
    return _cdf(-xl)
  elif rho == 0:
    return _cdf(-xl)*_cdf(-yl)
  
  tp = 2.*np.pi
  h, k = xl, yl
  hk = h*k
  bvn = 0.
  
  if np.abs(rho) < 0.3:
    # Gauss Legendre points and weights, n =  6
    w = np.array([0.1713244923791705, 0.3607615730481384, 0.4679139345726904])
    x = np.array([0.9324695142031522, 0.6612093864662647, 0.2386191860831970])
  elif np.abs(rho) < 0.75:
    # Gauss Legendre points and weights, n = 12
    w = np.array([0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
                  0.2031674267230659, 0.2334925365383547, 0.2491470458134029])
    x = np.array([0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
                  0.5873179542866171, 0.3678314989981802, 0.1252334085114692])
  else:
    # Gauss Legendre points and weights, n = 20
    w = np.array([.01761400713915212, .04060142980038694, .06267204833410906,
                  .08327674157670475, 0.1019301198172404, 0.1181945319615184,
                  0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
                  0.1527533871307259])
    x = np.array([0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
                  0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
                  0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
                  0.07652652113349733])
  
  w = np.tile(w, 2)
  x = np.concatenate([1.-x, 1.+x])
  
  if np.abs(rho) < 0.925:
    hs = .5 * (h*h + k*k)
    asr = .5*np.arcsin(rho)
    sn = np.sin(asr*x)
    bvn = np.dot(w, np.exp((sn*hk-hs)/(1.-sn**2)))
    bvn = bvn*asr/tp + _cdf(-h)*_cdf(-k) 
  else:
    if rho < 0.:
      k = -k
      hk = -hk
    if np.abs(rho) < 1.:
      ass = 1.-rho**2
      a = np.sqrt(ass)
      bs = (h-k)**2
      asr = -.5*(bs/ass + hk)
      c = (4.-hk)/8.
      d = (12.-hk)/80. 
      if asr > -100.:
        bvn = a*np.exp(asr)*(1.-c*(bs-ass)*(1.-d*bs)/3. + c*d*ass**2)
      if hk  > -100.:
        b = np.sqrt(bs)
        sp = np.sqrt(tp)*_cdf(-b/a)
        bvn = bvn - np.exp(-.5*hk)*sp*b*(1. - c*bs*(1.-d*bs)/3.)
      a = .5*a
      xs = (a*x)**2
      asr = -.5*(bs/xs + hk)
      inds = [i for i, asr_elt in enumerate(asr) if asr_elt>-100.]
      xs = xs[inds]
      sp = 1. + c*xs*(1.+5.*d*xs)
      rs = np.sqrt(1.-xs)
      ep = np.exp(-.5*hk*xs / (1.+rs)**2)/rs
      bvn = (a*np.dot(np.exp(asr[inds])*(sp-ep), w[inds]) - bvn)/tp
    if rho > 0:
      bvn +=  _cdf(-max(h, k)) 
    elif h >= k:
      bvn = -bvn
    else:
      if h < 0.:
        L = _cdf(k)-_cdf(h)
      else:
        L = _cdf(-h)-_cdf(-k)
      bvn =  L - bvn
  
  return max(0., min(1., bvn))

def _cdf(z):
  """Cumulative density function (CDF) of the standard normal distribution."""
  return .5 * (1. + erf(z/np.sqrt(2.)))


class Sphere(SyntheticTestFunction):
    r"""Sphere test function.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = x^2

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
    """

    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        self._bounds = [(-10., 10.) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        part1 = torch.norm(X, dim=-1)**2
        return part1

def standardize_return(Y: Tensor) -> Tensor:
    r"""Standardizes (zero mean, unit variance) a tensor by dim=-2.

    If the tensor is single-dimensional, simply standardizes the tensor.
    If for some batch index all elements are equal (or if there is only a single
    data point), this function will return 0 for that batch index.

    Args:
        Y: A `batch_shape x n x m`-dim tensor.

    Returns:
        The standardized `Y`.

    Example:
        >>> Y = torch.rand(4, 3)
        >>> Y_standardized = standardize(Y)
    """

    stddim = -1 if Y.dim() < 2 else -2
    Y_std = Y.std(dim=stddim, keepdim=True)
    Y_std = Y_std.where(Y_std >= 1e-9, torch.full_like(Y_std, 1.0))
    Y_mean = Y.mean(dim=stddim, keepdim=True)
    return (Y - Y.mean(dim=stddim, keepdim=True)) / Y_std, Y_mean, Y_std

class NormalKernel(RBFKernel):
    r"""
    Computes a covariance matrix based on the RBF (squared exponential) kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{RBF}}(\mathbf{x_1}, \mathbf{x_2}) = \exp \left( -\frac{1}{2}
          (\mathbf{x_1} - \mathbf{x_2})^\top \Theta^{-2} (\mathbf{x_1} - \mathbf{x_2}) \right)
       \end{equation*}

    where :math:`\Theta` is a lengthscale parameter.
    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if x1 is a `n x d` matrix. (Default: `None`.)
    :param batch_shape: Set this if you want a separate lengthscale for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf x1` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: `None`)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: `Positive`.)
    :param eps: The minimum value that the lengthscale can take (prevents
        divide by zero errors). (Default: `1e-6`.)

    :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=5))
        >>> covar = covar_module(x)  # Output: LinearOperator of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])))
        >>> covar = covar_module(x)  # Output: LinearOperator of size (2 x 10 x 10)
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
      scale = 1/(np.sqrt(np.linalg.det(2*np.pi*self.lengthscale)))
      if (
          x1.requires_grad
          or x2.requires_grad
          or (self.ard_num_dims is not None and self.ard_num_dims > 1)
          or diag
          or params.get("last_dim_is_batch", False)
          or trace_mode.on()
      ):
          x1_ = x1.div(self.lengthscale)
          x2_ = x2.div(self.lengthscale)
          return scale*postprocess_rbf(self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params))
      return scale*RBFCovariance.apply(
          x1,
          x2,
          self.lengthscale,
          lambda x1, x2: self.covar_dist(x1, x2, square_dist=True, diag=False, **params),
      )
    
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = torch.svd(B)

    H = (V.T) *(torch.diag(s)*V)

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(torch.norm(A).detach().numpy())
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = torch.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = torch.min(torch.real(torch.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = torch.cholesky(B)
        return True
    except:
        return False