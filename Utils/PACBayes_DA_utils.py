import numpy as np
from math import pi as math_pi
import torch
from torch import sqrt
from torch import exp
from torch import max as maximum
from torch import erf
from Models.stochastic_layers import StochasticLinear
import torch.nn as nn

pi = torch.Tensor([math_pi])
# Some useful constants
CTE_1_SQRT_2    = 1.0 / sqrt(torch.Tensor([2.0]))
CTE_1_SQRT_2PI  = 1.0 / sqrt(2 * pi)
CTE_SQRT_2_PI   = sqrt(2.0 / pi)


# Some useful functions, and their derivatives
def gaussian_loss(x):
    return 0.5 * ( 1.0 - erf(x * CTE_1_SQRT_2) )


def gaussian_loss_derivative(x):
    return -CTE_1_SQRT_2PI * exp(-0.5 * x**2)


def gaussian_convex_loss(x):
    return maximum( 0.5*(1.0-erf(x*CTE_1_SQRT_2)) , -x*CTE_1_SQRT_2PI+0.5 )


def gaussian_convex_loss_derivative(x):
    x = maximum(x, 0.0)
    return -CTE_1_SQRT_2PI * exp(-0.5 * x**2)


def gaussian_disagreement(x):
    return 0.5 * ( 1.0 - (erf(x * CTE_1_SQRT_2))**2 )


def gaussian_disagreement_derivative(x):
    return -CTE_SQRT_2_PI * erf(x * CTE_1_SQRT_2) * exp(-0.5 * x**2)


def gaussian_joint_error(x):
    return 0.25 * ( 1.0 - erf(x * CTE_1_SQRT_2) )**2


def gaussian_joint_error_derivative(x):
    return -CTE_1_SQRT_2PI * exp(-0.5 * x**2)*(1.0 - erf(x * CTE_1_SQRT_2) )


JE_SADDLE_POINT_X = torch.Tensor([-0.5060544689891808])
JE_SADDLE_POINT_Y = gaussian_joint_error(JE_SADDLE_POINT_X)
JE_SADDLE_POINT_DX = gaussian_joint_error_derivative(JE_SADDLE_POINT_X)


def gaussian_joint_error_convex(x):
    return maximum( gaussian_joint_error(x), JE_SADDLE_POINT_DX * (x - JE_SADDLE_POINT_X) + JE_SADDLE_POINT_Y)


def gaussian_joint_error_convex_derivative(x):
    return gaussian_joint_error_derivative( maximum(x, JE_SADDLE_POINT_X) )


def pac_bayes_convex_loss(y_hat_source,y_hat_target, y_source, c, b):
    return gaussian_joint_error_convex(y_source*y_hat_source)/c + gaussian_disagreement(y_hat_target)/b


def pac_bayes_loss(y_hat_source,y_hat_target, y_source, c, b):
    return gaussian_joint_error(y_source*y_hat_source)/c + gaussian_disagreement(y_hat_target)/b


def l2_reg(model, prm):
    l2_reg_res = torch.tensor(0.)
    if prm.model_type == 'Standard':
        layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
        for layer in layers:
            l2_reg_res += torch.norm(layer.weight)
            l2_reg_res += torch.norm(layer.bias)
    else:
        layers = [layer for layer in model.modules() if isinstance(layer, StochasticLinear)]
        for layer in layers:
            l2_reg_res += torch.norm(layer.w_mu)
    return l2_reg_res
