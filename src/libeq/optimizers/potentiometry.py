"Test collection for potentiometry data fitting."

import itertools
from typing import Protocol, TypeAlias, Any

import numpy as np
from numpy.typing import NDArray
from libeq.data_structure import SolverData
from libeq.solver import solve_equilibrium_equations
from libeq.consts import Flags, LN10

from . import jacobian
from . import libemf
from . import libfit


FArray: TypeAlias = NDArray[float]


def refine_indices(flags: list[Flags]):
    return [i == Flags.REFINE for i in flags]


class Bridge(Protocol):
    def __init__(self, data: SolverData):
        ...

    def accept_values(self) -> None:
        ...

    def matrices(self) -> tuple[FArray, FArray]:
        ...

    def size(self) -> tuple[int, int]:
        ...

    def take_step(self, increments: FArray) -> None:
        ...

    def tmp_residual(self) -> FArray:
        ...

    def weights(self) -> FArray:
        ...


class PotentiometryBridge:
    def __init__(self, data: SolverData, reporter) -> None:
        self._data = data
        self._reporter = reporter
        self._freeconcentration: FArray | None = None

        self._stoich = self._stoichiometry(extended=False)
        self._stoichx = self._stoichiometry(extended=True)
        self._nspecies, self._ncomponents = self._stoich.shape
        self._ntitrations = len(data.potentiometry_opts.titrations)
        self._experimental_points = [ len(t.v_add) for t in self._titrations() ]
        self._total_points = sum(self._experimental_points)
        self._chargesx = np.sum(self._stoichx*data.charges, axis=1)

        # calculate degrees of freedom
        self._dof_beta = sum(1 for _ in data.potentiometry_opts.beta_flags if _ == Flags.REFINE)
        self._dof_conc = 0
        self._slices = []
        self._slopes = np.zeros(self._total_points)
        self._emf0 = np.zeros(self._total_points)
        counter = 0
        for ntit, titration in enumerate(self._titrations()):
            self._dof_conc += sum(1 for _ in titration.c0_flags if _ == Flags.REFINE)
            self._dof_conc += sum(1 for _ in titration.ct_flags if _ == Flags.REFINE)
            self._slices.append(slice(counter, counter+len(titration.emf)))
            self._slopes[self._slices[-1]] = titration.slope / LN10
            self._emf0[self._slices[-1]] = titration.e0
            counter += len(titration.emf)
        self._dof = self._dof_beta + self._dof_conc

        ##> self._hindices
        self._hindices = []
        for titration in data.potentiometry_opts.titrations:
            self._hindices.extend(len(titration.v_add) * [titration.electro_active_compoment])

        ##> self._experimental_remf
        self._experimental_emf = np.concatenate([t.emf for t in data.potentiometry_opts.titrations])

        self._bmatrixt = np.concatenate([jacobian.bmatrix_t(t.v_add, t.v0, self._ncomponents)
                                         for t in self._titrations()])

        self._bmatrixb = np.concatenate([jacobian.bmatrix_b(t.v_add, t.v0, self._ncomponents)
                                         for t in self._titrations()])

        self._weights = np.concatenate([libemf.emf_weights(t.v_add, t.v0_sigma, t.emf, t.e0_sigma)
            for t in data.potentiometry_opts.titrations])

        # initial variable vector
        self._idx_refinable = []
        idx_refinable_beta = refine_indices(self._data.potentiometry_opts.beta_flags)
        self._idx_refinable.extend(idx_refinable_beta)
        beta_to_refine = np.extract(idx_refinable_beta, self._data.log_beta)
        concs_to_refine = []

        for titration in self._data.potentiometry_opts.titrations:
            if titration.c0_flags:
                assert len(titration.c0_flags) == self._ncomponents
                idx_refinable_c0 = refine_indices(titration.c0_flags)
            else:
                idx_refinable_c0 = self._ncomponents*[False]
            self._idx_refinable.extend(idx_refinable_c0)
            concs_to_refine.append(np.extract(idx_refinable_c0, titration.c0))

            if titration.ct_flags:
                assert len(titration.ct_flags) == self._ncomponents
                idx_refinable_ct = refine_indices(titration.ct_flags)
            else:
                idx_refinable_ct = self._ncomponents*[False]
            self._idx_refinable.extend(idx_refinable_ct)

            concs_to_refine.append(np.extract(idx_refinable_ct, titration.ct))

        self._variables = np.concatenate([beta_to_refine*LN10, *concs_to_refine])
        self._step = np.zeros(self._dof, dtype=float)

    def accept_values(self) -> None:
        "Accepts the step values and consolidates the data."
        self._variables += self._step
        self._step[...] = 0.0

    def final_values(self):
        yield self._beta()
        yield from self._titration_parameters()

    def iteration_history(self, **kwargs):
        ...

    def matrices(self) -> tuple[FArray, FArray]:
        "Return the jacobian and the residual arrays."
        # 1. calculate free concentrations
        freec = self._calc_free_concs(initial=True, update=True)
        assert freec.shape == (self._total_points, self._nspecies + self._ncomponents)

        # 2. calculate A
        amatrix = jacobian.amatrix(freec, self._stoichx)
        assert amatrix.shape == (self._total_points, self._ncomponents, self._ncomponents)

        # 3. calculate jacobian part referring to beta
        dlogc_dlogbeta = jacobian.dlogcdlogbeta(amatrix, freec, self._stoich)
        assert dlogc_dlogbeta.shape == (self._total_points, self._ncomponents, self._nspecies)
        jbeta = dlogc_dlogbeta

        # 4. calculate jacobian part referring to titration parameters
        _jc0 = jacobian.solve_xmatrix(amatrix, self._bmatrixt)
        _jct = jacobian.solve_xmatrix(amatrix, self._bmatrixb)
        jtit = np.zeros((self._total_points, self._ncomponents, 2*self._ntitrations*self._ncomponents), dtype=float)
        _js = [np.concatenate((_jc0[s], _jct[s]), axis=2) for s in self._slices]
        for n, s1 in enumerate(self._slices):
            s2 = slice(n*2*self._ncomponents, (n+1)*2*self._ncomponents)
            jtit[s1, :, s2] = _js[n]

        # 5. compute the total jacobian
        jac = self._slopes[:, None, None] * np.concatenate([jbeta, jtit], axis=2)
        assert jac.shape == (self._total_points, self._ncomponents, self._nspecies + 2*self._ncomponents*self._ntitrations)

        # 6. remove non refined parts
        trimmed_jac1 = jac[..., self._idx_refinable]
        trimmed_jac2 = trimmed_jac1[np.arange(self._total_points), self._hindices].copy()

        # 7. compute residual
        residual = self.__calculate_residual(freec)

        return trimmed_jac2, residual

    def relative_change(self, step):
        return step/self._variables

    def report_raw(self, text):
        print(text)

    def report_step(self, **kwargs):
        kwargs['log_beta'] = self._beta()
        self._reporter(**kwargs)

    def size(self) -> tuple[int, int]:
        "Return number of points, number os variables."
        return self._total_points, self._dof

    def take_step(self, increments: FArray) -> None:
        if increments.shape != self._step.shape:
            raise ValueError(f"Shape mismatch: {increments.shape} != {self._step.shape}")
        self._step[:] = increments[:]

    def tmp_residual(self) -> FArray:
        freec = self._calc_free_concs(initial=True, update=False)
        return self.__calculate_residual(freec)

    def weights(self) -> FArray:
        return np.diag(self._weights)

    @property
    def degrees_of_freedom(self) -> int:
        return self._dof

    @property
    def number_of_titrations(self) -> int:
        return self._ntitrations

    def _analytical_concentration(self) -> FArray:
        aconc = []
        for titration, (c0, ct) in zip(self._titrations(), self._titration_parameters()):
            aux = (c0[None, :] * titration.v0 + ct[None, :] * titration.v_add[:, None]) / \
                (titration.v0 + titration.v_add[:, None])
            aconc.append(aux)
        return np.concatenate(aconc, axis=0)

    def _background_concentration(self) -> FArray:
        # bconc = []
        # for titration, (c0b, ctb) in zip(self._data.potentiometry_opts.titrations,
        #                                  self._titration_parameters()):
        #     aux = (c0b[None, :] * titration.v0 + ctb[None, :] * titration.v_add[:, None]) / \
        #         (titration.v0 + titration.v_add[:, None])
        #     bconc.append(aux)
        bconc = [
            (titration.c0back * titration.v0 + titration.ctback * titration.v_add[:, None]) / \
                (titration.v0 + titration.v_add[:, None])
            for titration in self._titrations()
        ]
        return np.concatenate(bconc, axis=0)

    def _beta(self):
        beta = self._data.log_beta.copy()
        idx = refine_indices(self._data.potentiometry_opts.beta_flags)
        beta[idx] = (self._variables[:self._dof_beta] + self._step[:self._dof_beta]) / LN10
        return beta

    def _stoichiometry(self, extended=False):
        "Get stoichiometry array."
        number_components = self._data.stoichiometry.shape[0]
        if extended:
            return np.vstack((np.eye(number_components, dtype=int),
                              np.array(self._data.stoichiometry.T)))
        return self._data.stoichiometry.T

    def _titrations(self):
        yield from iter(self._data.potentiometry_opts.titrations)

    def _titration_parameters(self):
        itx = iter(self._variables[self._dof_beta:].tolist())
        itd = iter(self._step[self._dof_beta:].tolist())

        def select(c, flags):
            x = c.copy()
            for n, i in enumerate(flags):
                if i == Flags.REFINE:
                    x[i] = next(itx) + next(itd)
            return x

        for titration in self._titrations():
            c0 = select(titration.c0, titration.c0_flags)
            ct = select(titration.ct, titration.ct_flags)
            yield c0, ct
        
    def _calc_free_concs(self, initial=False, update=False) -> FArray:
        _initial_guess = None if initial else self._freeconcentration
        log_beta = self._beta()
        total_concentration = self._analytical_concentration()

        #charges = self._data.charges
        background_ions_concentration = self._background_concentration()
        independent_component_activity = None

        outer_fixed_point_params = {
            "ionic_strength_dependence": self._data.ionic_strength_dependence,
            "reference_ionic_str_species": self._data.reference_ionic_str_species,
            "reference_ionic_str_solids": self._data.reference_ionic_str_solids,
            "dbh_values": self._data.dbh_values.copy(),
            "charges": self._chargesx,
            "independent_component_activity": independent_component_activity,
            "background_ions_concentration": background_ions_concentration,
        }
        c, *_ = solve_equilibrium_equations(
            stoichiometry=self._data.stoichiometry,
            solid_stoichiometry=self._data.solid_stoichiometry,
            original_log_beta=log_beta,
            original_log_ks=self._data.log_ks,
            total_concentration=total_concentration,
            outer_fiexd_point_params=outer_fixed_point_params,
            initial_guess=_initial_guess,
            full=True)
        if update:
            self._freeconcentration = c
        return c

    def __calculate_residual(self, free_concentrations):
        assert free_concentrations.shape == (self._total_points, self._nspecies + self._ncomponents)
        eactive = libemf.hselect(free_concentrations, self._hindices) 
        calculated_emf = self._emf0 + self._slopes * np.log(eactive)
        assert calculated_emf.shape == (self._total_points,)
        residual = self._experimental_emf - calculated_emf
        return residual


def PotentiometryOptimizer(data: SolverData, reporter=None) -> dict[str, Any]:
    """
    Solve a potentiometry problem. Refine constants and possibly, concentrations.

    Parameters:
    -------
    data : SolverData
        The data for the refinement.

    Returns:
        x : 
            the refined data
        concs :
            the final free concentrations
        final_log_beta :
            the final refined constant values
        b_error :
            the fitting error in the concentration
        cor_matrix :
            the correlation matrix
        cov_matrix :
            the covariance matrix
        return_extra :
            additional information
    -------
    """
    bridge: Bridge = PotentiometryBridge(data, reporter)
    fit_result = libfit.levenberg_marquardt(bridge, debug=True)
    values = bridge.final_values()
    final_beta = next(values)
    final_total_concentration = list(itertools.islice(values, bridge.number_of_titrations))
    
    return {
        'final_beta': final_beta,
        'final_total_concentration': final_total_concentration
    }


def covariance_fun(J, W, F):
    """Compute covariance matrix.

    Returns the covariance matrix :math:`CV = inv(J'.W.J)*MSE`
    Where MSE is mean-square error :math:`MSE = (R'*R)/(N-p)`
    where *R* are the residuals, *N* is the number of observations and
    *p* is the number of coefficients estimated

    Parameters:
        J (:class:`numpy.ndarray`): the jacobian
        W (:class:`numpy.ndarray`): the weights matrix
        F (:class:`numpy.ndarray`): the residuals
    Returns:
        :class:`numpy.ndarray`: an (*p*, *p*)-sized array representing
            the covariance matrix.
    """
    num_params = J.shape[1] if J.ndim == 2 else 1
    mse = np.sum(F * np.diag(W) * F) / (len(F) - num_params)
    temp = np.linalg.inv(np.atleast_2d(np.dot(np.dot(J.T, W), J)))
    return temp * mse


def fit_final_calcs(jacobian, resids, weights):
    """Perform final calculations common to some routines.

    Parameters:
        jacobian (:class:`numpy.array`): the jacobian
        resids (:class:`numpy.array`): the residuals
        weights (:class:`numpy.array`): the weights
    Returns:
        * the error in beta
        * the correlation matrix
        * the covariance matrix
    """
    covariance = covariance_fun(jacobian, weights, resids)
    cov_diag = np.diag(covariance)
    error_B = np.sqrt(cov_diag) / np.log(10)
    lenD = len(cov_diag)
    correlation = covariance / np.sqrt(
        np.dot(cov_diag.reshape((lenD, 1)), cov_diag.reshape((1, lenD)))
    )
    return error_B, correlation, covariance


def ravel(x, y, flags):
    """Update values from one iterable with other iterable according to flags.

    This function does the opposite action than :func:`unravel`.

    Parameters:
        x (iterable): the original array values
        y (iterable): the updated values to be plugged into *x*.
        flags (sequence): flags indicating how to update *x* with *y*. Accepted
            values are:

            * 0: value is to be kept constant
            * 1: value is to be refined and the corresponding value from x
              will be substituted by the corresponding value from y.
            * >2: value is restrained. All places with the same number are
              refined together and the ratio between them is maintained.

    Yields:
        float: Raveled values.
    """
    # indices of the reference parameter for constraining
    ref_index = {i: flags.index(i) for i in range(2, 1 + max(flags))}
    ref_val = {}

    ity = iter(y)
    for i, f in enumerate(flags):
        if f == 1:  # refinable: return new value
            yield next(ity)
        elif f == 0:  # constant: return old value
            yield x[i]
        else:  # constrained: return or compute
            if i == ref_index[f]:
                val = next(ity)  # reference value: return new value
                ref_val[f] = val  # and store ref value
                yield val
            else:  # other: compute proportional value
                yield x[i] * ref_val[f] / x[ref_index[f]]

