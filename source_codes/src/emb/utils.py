import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Function
import sys


# Copyright (c) 2022-present, Huiru Xiao, HKUST-KnowComp.
# All rights reserved.
"""Complex Hyperbolic operations utils functions."""


MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5, torch.complex128: 1e-5}


# ################# MATH FUNCTIONS ########################
class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


# ################# HYP OPS ########################
def expmap0(u, c):
    """Exponential map taken at the origin of the Poincare ball with curvature c.
    Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
    Returns:
        torch.Tensor with tangent points.
    """
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def logmap0(y, c):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.
    Args:
        y: torch.Tensor of size B x d with tangent points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
    Returns:
        torch.Tensor with hyperbolic points.
    """
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def project(x, c):
    """Project points to unit ball with curvature c.
    Args:
        x: re_x + im_x * 1j, torch.Tensor of size B * d with complex hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
    Returns:
        torch.Tensor with projected complex hyperbolic points.
    """
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = 1e-5
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def real_mobius_add(x, y, c):
    """Mobius addition of points in the Poincare ball with curvature c.
    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        y: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
    Returns:
        Tensor of shape B x d representing the element-wise Mobius addition of x and y.
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def mobius_add(x, y):
    """Mobius addition of points in the unit ball with curvature c.
    Args:
        x: torch.Tensor of size B x d with complex hyperbolic points
        y: torch.Tensor of size B x d with complex hyperbolic points
    Returns:
        Tensor of shape B x d representing the element-wise Mobius addition of x and y.
    """
    num = x + y
    denom = 1 + conjugate(x) * y
    return num / denom


# ################# HYP DISTANCES ########################
# c is set to be 1
def chyp_distance(z, w, c=1):
    """Complex hyperbolic distance on the unit ball with curvature c.
    Args:
        z: torch.Tensor of size B x d with dtype=cfloat
        w: torch.Tensor with complex hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size 1 with absolute hyperbolic curvature
    Returns: torch,Tensor with complex hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    eps = BALL_EPS[z.real.dtype]
    zw = HermitianSig(z, w)   # size: B*1 otherwise
    wz = conjugate(zw)   # size: B*1 otherwise
    znorm = torch.clamp(HermitianNorm(z), min=-1, max=-eps)   # size: B*1
    wnorm = torch.clamp(HermitianNorm(w), min=-1, max=-eps)   # size: B*1
    x = torch.add(2 * (zw * wz).real / znorm.expand_as(zw) / wnorm.expand_as(zw), -1)   # size: B*1
    if torch.sum(torch.isnan(x)):
        print('The distance has NaN.')
        print(f'The max distance is {torch.max(x)}')
        sys.exit()
    return torch.acosh(x)


def hyp_distance_multi_c(x, v, c):
    """Hyperbolic distance on Poincare balls with varying curvatures c.
    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size B x d with absolute hyperbolic curvatures
    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    sqrt_c = c ** 0.5
    vnorm = torch.norm(v, p=2, dim=-1, keepdim=True)
    xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True)
    gamma = tanh(sqrt_c * vnorm) / sqrt_c
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


def HermitianSig(z, w):
    return torch.add(torch.sum(z * conjugate(w), dim=-1, keepdim=True), -1)


def conjugate(z):
    return torch.conj(z)


def HermitianNorm(z):
    return HermitianSig(z, z).real


class Distance(Function):
    @staticmethod
    def grad(z, w, znorm, wnorm, zw, wz, x, eps):
        # z: b x 1 x d
        # w: b x n x d
        # znorm: b x 1 x 1
        # wnorm: b x n x 1
        # zw: b x n x 1
        # wz: b x n x 1
        # x: b x n x 1

        p = torch.sqrt(torch.pow(x, 2) - 1) # b x n x 1
        p = torch.clamp(p * torch.pow(znorm, 2) * wnorm, max=-eps)
        _zww = zw * w # b x n x d
        _zwwz = zw * wz # b x n x 1
        a = znorm * _zww.real - _zwwz.real * z.real # b x n x d
        b = znorm * _zww.imag - _zwwz.real * z.imag # b x n x d
        grad_real = 4 * a / p
        grad_imag = 4 * b / p
        return grad_real, grad_imag

    @staticmethod
    def forward(ctx, lhs_e, rhs_e, c=1):
        """Complex hyperbolic distance on the unit ball with curvature c.
        Args:
            z: torch.Tensor of size B x d with dtype=cfloat
            w: torch.Tensor with complex hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
            c: torch.Tensor of size 1 with absolute hyperbolic curvature
        Returns: torch,Tensor with complex hyperbolic distances, size B x 1 if eval_mode is False
                else B x n_entities matrix with all pairs distances
        """
        eps = BALL_EPS[lhs_e.dtype]
        rank = lhs_e.size(-1) // 2
        z = lhs_e[..., :rank] + lhs_e[..., rank:] * 1j
        w = rhs_e[..., :rank] + rhs_e[..., rank:] * 1j
        zw = HermitianSig(z, w)   # size: B*1
        wz = conjugate(zw)   # size: B*1
        znorm = torch.clamp(HermitianNorm(z), min=-1, max=-eps)   # size: B*1
        wnorm = torch.clamp(HermitianNorm(w), min=-1, max=-eps)   # size: B*1
        x = torch.add(2 * (zw * wz).real / znorm.expand_as(zw) / wnorm.expand_as(zw), -1)   # size: B*1
        x = torch.clamp(x, min=1 + eps)
        ctx.eps = eps
        ctx.save_for_backward(z, w, znorm, wnorm, zw, wz, x)
        return torch.acosh(x)

    @staticmethod
    def backward(ctx, g):
        z, w, znorm, wnorm, zw, wz, x = ctx.saved_tensors
        re_gz, im_gz = Distance.grad(z, w, znorm, wnorm, zw, wz, x, ctx.eps)
        re_gw, im_gw = Distance.grad(w, z, wnorm, znorm, wz, zw, x, ctx.eps)
        re_z_grad = g.expand_as(re_gz) * re_gz
        im_z_grad = g.expand_as(im_gz) * im_gz
        re_w_grad = g.expand_as(re_gw) * re_gw
        im_w_grad = g.expand_as(im_gw) * im_gw
        z_grad = torch.cat((re_z_grad, im_z_grad), -1)
        w_grad = torch.cat((re_w_grad, im_w_grad), -1)
        if z.size(1) == 1:
            z_grad = z_grad.sum(1, keepdim=True)
        if w.size(1) == 1:
            w_grad = w_grad.sum(1, keepdim=True)
        return z_grad, w_grad, None, None


def full_p_exp_map(x, v):
    """
        v: vector in the tangent space
    """
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    y = torch.tanh(normv/(1-sqxnorm)) * v/normv
    return p_sum(x, y)


def p_sum(x, y):
    # Mobius addtion
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    sqynorm = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1-1e-5)
    dotxy = torch.sum(x*y, dim=-1, keepdim=True)
    numerator = (1+2*dotxy+sqynorm)*x + (1-sqxnorm)*y
    denominator = 1 + 2*dotxy + sqxnorm*sqynorm
    return numerator/denominator


def euclidean_update(p, d_p, lr):
    p.data = p.data - lr * d_p
    return p.data


def poincare_grad(p, d_p):
    p_sqnorm = torch.clamp(torch.sum(p.data ** 2, dim=-1, keepdim=True), 0, 1 - 1e-5)
    d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p


def poincare_update(p, d_p, lr):
    v = -lr * d_p
    p.data = full_p_exp_map(p.data, v)
    return p.data

# Inherited from https://github.com/HazyResearch/KGEmb
"""Euclidean operations utils functions."""

def euc_sqdistance(x, y):
    """Compute euclidean squared distance between tensors.
    Args:
        x: torch.Tensor of shape (N1 x d)
        y: torch.Tensor of shape (N2 x d)
        eval_mode: boolean
    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    return x2 + y2 - 2 * xy


def givens_rotations(r, x):
    """Givens rotations.
    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate
    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))   # givens size: (N * d/2 * 2)
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)   # normalize to make cos & sin
    x = x.view((r.shape[0], -1, 2))   # x size: (N * d/2 * 2)
    x_rot = givens[..., 0:1] * x + givens[..., 1:] * torch.cat((-x[..., 1:], x[..., 0:1]), dim=-1)   # x_rot size: (N * d/2 * 2)
    return x_rot.view(r.size())   # size: (N * d)


def givens_reflection(r, x):
    """Givens reflections.
    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect
    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
    x = x.view((r.shape[0], -1, 2))
    x_ref = givens[..., 0:1] * torch.cat((x[..., 0:1], -x[..., :1]), dim=-1) + givens[..., 1:] * torch.cat(
        (x[..., 1:], x[..., 0:1]), dim=-1)
    return x_ref.view(r.size())

# Inherited from https://github.com/HazyResearch/KGEmb
"""Hyperbolic operations utils functions."""

MIN_NORM_HYPERBOLIC = 1e-15
BALL_EPS_HYPERBOLIC = {torch.float32: 4e-3, torch.float64: 1e-5}


# ################# MATH FUNCTIONS ########################

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


# ################# HYP OPS ########################

def expmap0(u, c):
    """Exponential map taken at the origin of the Poincare ball with curvature c.
    Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
    Returns:
        torch.Tensor with tangent points.
    """
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM_HYPERBOLIC)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def logmap0(y, c):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.
    Args:
        y: torch.Tensor of size B x d with tangent points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
    Returns:
        torch.Tensor with hyperbolic points.
    """
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM_HYPERBOLIC)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def project(x, c):
    """Project points to Poincare ball with curvature c.
    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM_HYPERBOLIC)
    eps = BALL_EPS_HYPERBOLIC[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c):
    """Mobius addition of points in the Poincare ball with curvature c.
    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        y: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
    Returns:
        Tensor of shape B x d representing the element-wise Mobius addition of x and y.
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM_HYPERBOLIC)


# ################# HYP DISTANCES ########################

def hyp_distance(x, y, c):
    """Hyperbolic distance on the Poincare ball with curvature c.
    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size 1 with absolute hyperbolic curvature
    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    sqrt_c = c ** 0.5
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * xy + c * y2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)
    denom = 1 - 2 * c * xy + c ** 2 * x2 * y2
    pairwise_norm = num / denom.clamp_min(MIN_NORM_HYPERBOLIC)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


def hyp_distance_multi_c(x, v, c):
    """Hyperbolic distance on Poincare balls with varying curvatures c.
    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size B x d with absolute hyperbolic curvatures
    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    sqrt_c = c ** 0.5
    vnorm = torch.norm(v, p=2, dim=-1, keepdim=True)
    xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True)
    gamma = tanh(sqrt_c * vnorm) / sqrt_c
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM_HYPERBOLIC)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c

def get_conve_nn_state_dict(state_dict):
    conve_nn_state_dict = {}
    for param_name in ['mdl.b', 'mdl.conv1.weight', 'mdl.conv1.bias', 'mdl.bn0.weight', 'mdl.bn0.bias',
                       'mdl.bn0.running_mean', 'mdl.bn0.running_var', 'mdl.bn1.weight', 'mdl.bn1.bias',
                       'mdl.bn1.running_mean', 'mdl.bn1.running_var', 'mdl.bn2.weight', 'mdl.bn2.bias',
                       'mdl.bn2.running_mean', 'mdl.bn2.running_var', 'mdl.fc.weight', 'mdl.fc.bias']:
        conve_nn_state_dict[param_name.split(
            '.', 1)[1]] = state_dict['state_dict'][param_name]
    return conve_nn_state_dict


def get_conve_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight']:
        kg_state_dict[param_name.split(
            '.', 1)[1]] = state_dict['state_dict'][param_name]
    return kg_state_dict


def get_tucker_nn_state_dict(state_dict):
    tucker_nn_state_dict = {}
    for param_name in ['mdl.W', 'mdl.bn0.weight', 'mdl.bn0.bias',
                       'mdl.bn0.running_mean', 'mdl.bn0.running_var', 'mdl.bn1.weight', 'mdl.bn1.bias',
                       'mdl.bn1.running_mean', 'mdl.bn1.running_var']:
        tucker_nn_state_dict[param_name.split(
            '.', 1)[1]] = state_dict['state_dict'][param_name]
    return tucker_nn_state_dict


def get_tucker_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight']:
        kg_state_dict[param_name.split(
            '.', 1)[1]] = state_dict['state_dict'][param_name]
    return kg_state_dict


def get_complex_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight',
                       'kg.entity_img_embeddings.weight', 'kg.relation_img_embeddings.weight']:
        kg_state_dict[param_name.split(
            '.', 1)[1]] = state_dict['state_dict'][param_name]
    return kg_state_dict


def get_distmult_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight']:
        kg_state_dict[param_name.split(
            '.', 1)[1]] = state_dict['state_dict'][param_name]
    return kg_state_dict


def get_ptranse_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight']:
        kg_state_dict[param_name.split(
            '.', 1)[1]] = state_dict['state_dict'][param_name]
    return kg_state_dict


def get_ptranse_kg_state_dict_from_vec(path):
    entity_lines = open(path + 'entity2vec.txt').readlines()
    relation_lines = open(path + 'relation2vec.txt').readlines()
    entity_vec = []
    for line in entity_lines:
        entity_vec.append([float(x) for x in line.strip().split()])
    relation_vec = []
    for line in relation_lines:
        relation_vec.append([float(x) for x in line.strip().split()])
    entity_vec_tensor = torch.tensor(entity_vec)
    relation_vec_tensor = torch.tensor(relation_vec)
    kg_state_dict = dict()
    kg_state_dict['entity_embeddings.weight'] = entity_vec_tensor
    kg_state_dict['relation_embeddings.weight'] = relation_vec_tensor
    return kg_state_dict
