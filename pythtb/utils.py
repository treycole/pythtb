import numpy as np
from math import factorial
from itertools import permutations

__all__ = [
    "levi_civita",
    "finite_diff_coeffs",
    "is_Hermitian",
    "pauli_decompose",
]

def detect_degeneracies(eigenvalues, tol=1e-8):
    """
    Detects degeneracies in a list of eigenvalues.

    Parameters:
        eigenvalues (array): List or array of eigenvalues (assumed sorted).
        tol (float): Tolerance for identifying degeneracy.

    Returns:
        degenerate_groups (list of lists): Indices of degenerate eigenvalues.
    """
    eigenvalues = np.array(eigenvalues)
    degenerate_groups = []
    current_group = [0]

    for i in range(1, len(eigenvalues)):
        if abs(eigenvalues[i] - eigenvalues[i-1]) < tol:
            current_group.append(i)
        else:
            if len(current_group) > 1:
                degenerate_groups.append(current_group)
            current_group = [i]

    if len(current_group) > 1:
        degenerate_groups.append(current_group)

    return degenerate_groups

def levi_civita(n, d):
    """
    Constructs the rank-n Levi-Civita tensor in dimension d.

    Parameters:
    n (int): Rank of the tensor (number of indices).
    d (int): Dimension (number of possible index values).

    Returns:
    np.ndarray: Levi-Civita tensor of shape (d, d, ..., d) with n dimensions.
    """
    shape = (d,) * n
    epsilon = np.zeros(shape, dtype=int)
    # Generate all possible permutations of n indices
    for perm in permutations(range(d), n):
        # Compute the sign of the permutation
        sign = np.linalg.det(np.eye(n)[list(perm)])
        epsilon[perm] = int(np.sign(sign))  # +1 for even, -1 for odd permutations

    return epsilon

def finite_diff_coeffs(order_eps, derivative_order=1, mode='central'):
    """
    Compute finite difference coefficients using the inverse of the Vandermonde matrix.

    Parameters:
        stencil_points (array-like): The relative positions of the stencil points (e.g., [-2, -1, 0, 1, 2]).
        derivative_order (int): Order of the derivative to approximate (default is first derivative).

    Returns:
        coeffs (numpy array): Finite difference coefficients for the given stencil.
    """
    if mode not in ["central", "forward", "backward"]:
        raise ValueError("Mode must be 'central', 'forward', or 'backward'.")

    num_points = derivative_order + order_eps

    if mode == "central":
        if num_points % 2 == 0:
            num_points += 1
        half_span = num_points//2
        stencil = np.arange(-half_span, half_span + 1)

    elif mode == "forward":
        stencil = np.arange(0, num_points)

    elif mode == "backward":
        stencil = np.arange(-num_points+1, 1)

    A = np.vander(stencil, increasing=True).T  # Vandermonde matrix
    b = np.zeros(num_points)
    b[derivative_order] = factorial(derivative_order) # Right-hand side for the desired derivative

    coeffs = np.linalg.solve(A, b)  # Solve system Ax = b
    return coeffs, stencil

def is_Hermitian(M):
    """
    Check if a matrix M is Hermitian.

    Parameters:
        M (array-like): A square matrix (as a numpy array or convertible to one).

    Returns:
        bool: True if M is Hermitian, False otherwise.
    """
    M = np.array(M, dtype=complex)
    if M.ndim == 0:
        return np.allclose(M, np.conj(M))
    # 1D: not Hermitian (by usual definition)
    if M.ndim == 1:
        return False
    # Otherwise: check M == M^\dagger
    return np.allclose(M, M.conj().swapaxes(-1, -2))

def pauli_decompose(M):
    """
    Decompose a 2x2 matrix M in terms of the Pauli matrices.

    That is, find coefficients a0, a1, a2, a3 such that:

        M = a0 * I + a1 * sigma_x + a2 * sigma_y + a3 * sigma_z

    Parameters:
        M (array-like): A 2x2 matrix (as a numpy array or convertible to one).
        precision (int): Number of significant digits for the coefficients.

    Returns:
        str: A string representing the decomposition, e.g.
             "1I + 0.3τₓ - 0.2τ_y + 0τ_z"

    Note: This function is applicable only when nspin = 2.
    """
    M = np.array(M, dtype=complex)
    if M.shape != (2, 2):
        raise ValueError("Matrix must be 2x2 for Pauli decomposition.")

    # Define the 2x2 identity and Pauli matrices.
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Compute coefficients using the Hilbert-Schmidt inner product.
    a0 = 0.5 * np.trace(M)
    a1 = 0.5 * np.trace(np.dot(M, sigma_x))
    a2 = 0.5 * np.trace(np.dot(M, sigma_y))
    a3 = 0.5 * np.trace(np.dot(M, sigma_z))

    return [a0, a1, a2, a3]

def no_2pi(x, clos):
    "Make x as close to clos by adding or removing 2pi"
    while abs(clos - x) > np.pi:
        if clos - x > np.pi:
            x += 2.0 * np.pi
        elif clos - x < -1.0 * np.pi:
            x -= 2.0 * np.pi
    return x

def _cart_to_red(a_vecs, cart):
    "Convert cartesian vectors cart to reduced coordinates of a1,a2,a3 vectors"
    # (a1, a2, a3) = tmp
    # matrix with lattice vectors
    # cnv = np.array([a1, a2, a3])
    # cnv = cnv.T  # transpose
    # # reduced coordinates
    # red = np.zeros_like(cart, dtype=float)
    # for i in range(0, len(cart)):
    #     red[i] = np.dot(cnv, cart[i])
    # return red
    cnv = np.linalg.inv(np.array(a_vecs).T)  # inverse
    return np.dot(cart, cnv.T)

def _red_to_cart(a_vecs, red):
    "Convert reduced to cartesian vectors."
    a1, a2, a3 = a_vecs

    basis = np.array([a1, a2, a3])
    cart = np.array(red) @ basis 

    # # cartesian coordinates
    # cart2 = np.zeros_like(red, dtype=float)
    # for i in range(0, len(cart)):
    #     cart2[i, :] = a1 * red[i][0] + a2 * red[i][1] + a3 * red[i][2]
    # print(np.allclose(cart, cart2))  # should be True

    return cart


def _is_int(a):
    return np.issubdtype(type(a), np.integer)



class PositionOperatorApproximationError(Exception):
    """
    Raised when a calculation involving the position operator is attempted
    using a tight-binding model generated by Wannier90, which neglects off-diagonal
    position operator elements.
    """
    pass

def _offdiag_approximation_warning_and_stop():
    raise PositionOperatorApproximationError(
        """

----------------------------------------------------------------------

  It looks like you are trying to calculate Berry-like object that
  involves position operator.  However, you are using a tight-binding
  model that was generated from Wannier90.  This procedure introduces
  approximation as it ignores off-diagonal elements of the position
  operator in the Wannier basis.  This is discussed here in more
  detail:

    http://www.physics.rutgers.edu/pythtb/usage.html#pythtb.w90

  If you know what you are doing and wish to continue with the
  calculation despite this approximation, please call the following
  function on your TBModel object

    my_model.ignore_position_operator_offdiagonal()

----------------------------------------------------------------------

"""
    )
