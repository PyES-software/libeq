import numpy as np


def covariance(J, W):
    """Compute covariance matrix.

    Parameters:
        J (:class:`numpy.ndarray`): the jacobian
        W (:class:`numpy.ndarray`): the weights matrix
    Returns:
        :class:`numpy.ndarray`: an (*p*, *p*)-sized array representing
            the covariance matrix.
    """
    aux2 = J.T @ np.diag(np.diag(W)) @ J
    return np.linalg.pinv(aux2)


def fitting_errors(covar):
    return np.sqrt(np.diag(covar))


def correlation_matrix(covar):
    D = np.diag(covar)
    nD = len(D)
    return covar/np.sqrt(np.dot(D.reshape((nD, 1)), D.reshape((1, nD))))


def extrapoly(x0, X, Y):
    r"""Polynomial extrapolation.

    Given a list of *x* and *y* points, and an *x0* point, this routine
    calculates the polynomials that goes through the given points and
    evaluates them for *x0*. It evaluates *m* extrapolations at once out of
    *n* polynomials of degree *g* -1

    Parameters:
        x0 (:class:`numpy.ndarray`): An (m, n)-array with the values for the
            polynomials to be evaluated.
        X (:class:`numpy.ndarray`): An (n, g)-array with the *x* values of the
            data.
        Y (:class:`numpy.ndarray`): An (n, g)-array with the *y* values of the
            data. Axis-0 must be of the same length than X.

    Returns:
        :class:`numpy.ndarray`: An (m, n)-array with calculated *y* values
             that produce the evaluation of the polynomials at *x0*
    """
    def polyexp(array, axis, n):
        """Given any array and one axis of it, return the same array with one
        extra dimmension in which the new dimmension is the array raised to
        0..n

        >>> a = np.array([[1, 1], [2, 2]])
        >>> polyexp(a, -1, 2)
        [[[1, 1], [1, 1]],
         [[1, 1], [2, 2]],
         [[1, 1], [4, 4]]]
        """
        a = np.expand_dims(array, axis)
        s = [1] * a.ndim
        s[axis] = n+1
        b = np.reshape(np.arange(n+1), tuple(s))
        return np.power(a, b)

    g, n = Y.shape
    Q = polyexp(X.T, -1, g-1)
    assert Q.shape == (n, g, g)
    A = np.linalg.solve(Q, Y.T)
    assert A.shape == (n, g)
    m = x0.shape[1]
    assert x0.shape[0] == n
    xexp = polyexp(x0.T, -1, g-1)
    assert xexp.shape == (m, n, g)
    r = np.sum(xexp * A[np.newaxis, ...], axis=-1)
    return np.squeeze(r)


def quadratic_extrapolation(a, b, c):
    """Perform 3-point extrapolation.

    Given three ordered points of an evenly spaced curve, return the
    quadratic extrapolation for estimating the following point.
    """
    return a - 3*b + 3*c


def nearest(a, b):
    """Find nearest element in a sorted list of numbers.

    Given a two list of indices, returns a sorted list of the indices of
    **a** which are closer to those of **b**
    """
    return np.argsort(np.abs(a[:, None]-b[None, :]), axis=0)


def sample_size_change(data, new_size):
    """Resample a free concentration array.

    Parameters:
        data (:class:`numpy.ndarray`): The data to be resized. Dimmension 0
            must be the sample number. Lenght of dimmension is assumed to
            be the size of the old data.
        new_size (int): The length of the new data.
    Returns:
        :class:`numpy.ndarray`: The resized array along dimmension 0.
    """
    from scipy.interpolate import interp1d
    old_size = data.shape[0]
    f_interp = interp1d(np.arange(old_size), data, axis=0, assume_sorted=True)
    new_x = np.linspace(0, old_size-1, new_size)
    return f_interp(new_x)


def weighting_slope(x, y, error_x, error_y):
    r"""Calculate weighting scheme acording to Gans et al.

    .. math:: w = s_E^2 + \left(\frac{\partial E}{\partial V}\right)^2  s_V^2

    Parameters:
        x (iterable): volume in mL
        y (iterable): emf in mV
        error_x (iterable): error of volume in mL
        error_y (iterable): error of emf in mV

    Returns:
        :class:`numpy.ndarray`: An array containing the calculated weights
    """
    dydx = np.gradient(x, y)
    yield from (1/(error_y**2 + d**2 * error_x**2) for d in dydx)


def m_matrix(jacobian, weights):
    return np.dot(np.dot(jacobian.T, weights), jacobian)


def error_params(jacobian, weights):
    M = m_matrix(jacobian, weights)
    return np.diag(np.linalg.inv(M))


def fit_sigma(residuals: np.ndarray, weights: np.ndarray, npoints: int, nparams: int) -> float:
    """Calculate the fit's sigma value for a given set of residuals and weights."""
    return np.sqrt(np.sum(weights*residuals**2))/(npoints-nparams)


def is_near_singular_lstsq(matrix, thresh=1e-3):
    """
    Test whether the matrix is near singular.
    """
    result = np.linalg.lstsq(matrix, np.eye(matrix.shape[1]), rcond=thresh)
    rank = result[2]                   # third return value = rank in NumPy ≥ 2.0
    s = result[3]                      # singular values
    rcond_est = s[-1] / s[0] if len(s) > 1 else 0.0
    print(f">>> {rcond_est}  {rank}")
    return rcond_est < thresh or rank < min(matrix.shape)
