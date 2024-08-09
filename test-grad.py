# test-grad.py
#
# (c) 2024-06-28, Philip D. Loewen

import math
import numpy as np
from PrettyPrinter import ppm
from TensorGradient import grad
import sys


def quadform(Q):
    u = np.array([1, 2, 3, 4]).reshape(4, 1)
    v = np.array([3, 4, 5]).reshape(3, 1)
    uQv = (u.T @ Q @ v)[0, 0]
    return uQv


# In a quadratic form like  u'*Q*v,
# treating the vectors u and v as fixed
# leads to a scalar-valued linear function
# of the matrix Q, whose exact gradient is
# the rank-1 product  u*v'. That's independent
# of the evaluation point, so we should be
# able to get the same result from any 2
# evaluation points.

A0 = np.random.rand(4, 3) * 10
ppm(A0, "evaluation point A0")

dqf0 = grad(quadform, A0)
# print(dqf)
ppm(dqf0, "derivative of quadform at A0")

A1 = np.random.rand(4, 3) * 1e10
ppm(A1, "evaluation point A1")

dqf1 = grad(quadform, A1)
# print(dqf)
ppm(dqf1, "derivative of quadform at A1")

print(40 * "=")

# Next up, some scalar-valued linear functions for
# input tensors that are not 2x2 matrices.


def vecdot(w):
    c = np.array([3, 2, 0, 4, 0, 2])
    v = np.tensordot(c, w, axes=c.ndim)
    return v.item()


a = np.array([1, 1, 1, 1, 1, 1])
ppm(a, "evaluation point a")
print(f"vecdot(a) = {vecdot(a)}.")
dvecdot = grad(vecdot, a)
ppm(dvecdot, "calculated gradient")

print(40 * "=")

C = np.random.rand(2, 3, 2)
ppm(C, "C")


def t33(w):
    #    C = np.ones((2,3))
    v = np.tensordot(C, w, axes=C.ndim)
    return v.item()


M = np.zeros(C.shape)
ppm(M, "evaluation point M")
print(f"t33(M) = {t33(M)}.")
dt33 = grad(t33, M)
ppm(dt33, "calculated gradient")

sys.exit(0)
