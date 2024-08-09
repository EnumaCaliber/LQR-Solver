# 2024-06-03 [PDL]

# Given a function f that maps a numpy ndarray to a scalar result,
# and a specific input ndarray A, use a fourth-order difference
# scheme to approximate the ndarray-valued gradient grad f(A).
#
# The return value is an ndarray with the same shape as A,
# organized to support the following kind of linear approximation:
#    f(A + dA) is very close to f(A) + grad(f,A)*dA.
# Here on the right the operator "*" is the appropriate dot-product.
# In numpy the final term added on the right above looks like this:
#    numpy.tensordot( grad(f,A), dA, axes=A.ndim ).item()
# There are two fine points to notice here:
#    1. Specifying the optional parameter "axes" in tensordot is essential
#    2. The tensordot function returns 0-dimensional array,
#       which is a lot like a scalar, but an explicit conversion is
#       required. That's what the "item()" method achieves.

import numpy as np


def grad(f, M0):
    if not isinstance(M0, np.ndarray):
        print("Error in function grad: ", end="")
        print("input argument must have type numpy.ndarray.")
        return None

    df = np.zeros(M0.shape)  # Container for the return value
    dM = np.zeros(M0.shape)  # Perturbation: zero of correct size

    typ = abs(np.absolute(M0).sum() / M0.size)  # average element size
    # print(f"In grad, typ = {typ}.")
    if typ == 0.0:
        # Assume a well-scaled problem.
        typ = 1.0
    h = typ * (np.finfo(float).eps) ** 0.25  # Looks good in experiments

    it = np.nditer(M0, flags=["multi_index"])
    while not it.finished:
        dM[it.multi_index] = 1.0  # One-hot perturbation object

        if True:
            # Use the standard fourth-order difference formula
            tmp = f(M0 + 2 * h * dM) - f(M0 - 2 * h * dM)
            tmp += -8 * f(M0 + h * dM) + 8 * f(M0 - h * dM)
            df[it.multi_index] = -tmp / (12 * h)
        else:
            # Use this simpler second-order formula
            df[it.multi_index] = (f(M0 + h * dM) - f(M0 - h * dM)) / (2 * h)

        dM[it.multi_index] = 0.0  # Reset perturbation object
        it.iternext()

    return df
