"Test collection for potentiometry data fitting."

import itertools
from typing import Protocol, TypeAlias, Any

import numpy as np
from numpy.typing import NDArray
from libeq.data_structure import SolverData
from libeq.solver import solve_equilibrium_equations
from libeq.consts import Flags, LN10

from .. import excepts
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

    def final_result(self) -> dict:
        ...

    def matrices(self) -> tuple[FArray, FArray]:
        ...

    def size(self) -> tuple[int, int]:
        ...

    def trial_step(self, increments: FArray) -> None:
        ...

    def tmp_residual(self) -> FArray:
        ...

    def weights(self) -> FArray:
        ...


class PotentiometryBridge:
    bridge.incorporate_stdev(stdev)
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
        if np.any(np.isnan(self._weights)):
            raise ValueError("Some calculated weight values are NaN")

        # initial variable vector
        self._idx_refinable = []
        idx_refinable_beta = refine_indices(self._data.potentiometry_opts.beta_flags)
        self._slice_betas = slice(0, sum(1 for _ in idx_refinable_beta if _))
        self._any_beta_refined = any(idx_refinable_beta)
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
            if any(idx_refinable_c0):
                concs_to_refine.extend(np.extract(idx_refinable_c0, titration.c0).tolist())

            if titration.ct_flags:
                assert len(titration.ct_flags) == self._ncomponents
                idx_refinable_ct = refine_indices(titration.ct_flags)
            else:
                idx_refinable_ct = self._ncomponents*[False]
            self._idx_refinable.extend(idx_refinable_ct)
            if any(idx_refinable_ct):
                concs_to_refine.append(np.extract(idx_refinable_ct, titration.ct).tolist())

        self._any_conc_refined = (len(concs_to_refine) > 0)

        # very important !! the betas are in natural logarithm!
        self._variables = np.concatenate([beta_to_refine*LN10, np.array(concs_to_refine)])
        self._step = np.zeros(self._dof, dtype=float)
        self._previous_values = np.empty_like(self._variables)

        # Calculate free concetrations initially
        self._freeconcentration = self._calc_free_concs(initial=True, update=False)

    def accept_step(self) -> None:
        "Update the variables values and reset increments to 0.0."
        self._previous_values[:] = self._variables
        self._variables += self._step
        self._step[...] = 0.0

    def reject_step(self) -> None:
        """
        The step is rejected and increments are reset to 0.0 without updating the variables.
        """
        self._step[...] = 0.0

    def final_result(self) -> dict:
        variables = self._variables.copy()
        variables[self._slice_betas] /= LN10

        istd = next(self.stdev)

        err_log_beta = [next(istd)/consts.LN10 if f == Flags.REFINE else None
                        for f in self._data.potentiometry_opts.beta_flags]

        err_titr_parms = [
            [
                [next(istd) if c0f == Flags.REFINE else None for c0f in t.c0_flags],
                [next(istd) if ctf == Flags.REFINE else None for c0f in t.c0_flags]
            ] for t in self._titrations() ]

        # fvals = self.final_values()
        eactive = libemf.hselect(self._freeconcentration, self._hindices) 
        
        retval = {
            'final variables': variables,
            'final log beta': self._beta(),
            'error log beta': err_log_beta,
            'final titration parameters': list(self._titration_parameters()),
            'error titration parameters': err_titr_parms,
            'free concentration': self._freeconcentration,
            'slices': self._slices,
            'total concentration': self._analytical_concentration(),
            'read emf': self._experimental_emf,
            'eactive': eactive,
            'background ion concentration': self._background_concentration(),
            'weights': self._weights
        }
        return retval


    # def final_values(self):
    #     yield self._beta()
    #     yield from self._titration_parameters()

    def incorporate_stdev(self, stdev):
        assert len(stdev) == len(self.variables)
        self.stdev = stdev

    def matrices(self) -> tuple[FArray, FArray]:
        """
        Compute the jacobian and the residual arrays for the accepted step.
        """
        # 1. calculate free concentrations
        freec = self._calc_free_concs(initial=False, update=True)
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
        jac =  np.concatenate([jbeta, jtit], axis=2)
        assert jac.shape == (self._total_points, self._ncomponents, self._nspecies + 2*self._ncomponents*self._ntitrations)

        # 6. remove non refined parts
        trimmed_jac1 = jac[..., self._idx_refinable]
        trimmed_jac2 = trimmed_jac1[np.arange(self._total_points), self._hindices]
        trimmed_jac3 = -LN10*self._slopes[:, None] * trimmed_jac2

        # 7. compute residual
        residual = self.__calculate_residual(freec)

        return trimmed_jac3, residual

    def relative_change(self, step):
        return step/self._variables

    def report_raw(self, text):
        print(text)

    def report_step(self, **kwargs):
        """
        Pass refinement parameters on each iteration to report.
        """
        kwargs['log_beta'] = self._beta()
        kwargs['previous log beta'] = self._previous_values[self._slice_betas]/LN10
        kwargs['increment'] /= LN10
        kwargs['stoichiometry'] = self._stoich
        kwargs['any beta refined'] = self._any_beta_refined
        kwargs['any conc refined'] = self._any_conc_refined
        kwargs['titration params'] = list(self._titration_parameters())
        self._reporter(**kwargs)

    def size(self) -> tuple[int, int]:
        "Return number of points, number of variables."
        return self._total_points, self._dof

    def trial_step(self, increments: FArray) -> None:
        if increments.shape != self._step.shape:
            raise ValueError(f"Shape mismatch: {increments.shape} != {self._step.shape}")
        self._step[:] = increments[:]

    def tmp_residual(self) -> FArray:
        """
        Calculate and return residual for the trial step.
        """
        freec = self._calc_free_concs(initial=False, update=False)
        return self.__calculate_residual(freec)

    def weights(self) -> FArray:
        return np.diag(self._weights)

    @property
    def degrees_of_freedom(self) -> int:
        """
        The number of variables to refine.
        """
        return sum(self._experimental_points) - self._dof

    @property
    def number_of_titrations(self) -> int:
        """
        The number of titrations sets available.
        """
        return self._ntitrations

    def _analytical_concentration(self) -> FArray:
        aconc = []
        for titration, (c0, ct) in zip(self._titrations(), self._titration_parameters()):
            aux = (c0[None, :] * titration.v0 + ct[None, :] * titration.v_add[:, None]) / \
                (titration.v0 + titration.v_add[:, None])
            aconc.append(aux)
        return np.concatenate(aconc, axis=0)

    def _background_concentration(self) -> FArray:
        bconc = [
            (titration.c0back * titration.v0 + titration.ctback * titration.v_add[:, None]) / \
                (titration.v0 + titration.v_add[:, None])
            for titration in self._titrations()
        ]
        return np.concatenate(bconc, axis=0)

    def _beta(self):
        """
        Return log10(beta) for the trial step.
        """
        beta = self._data.log_beta.copy()
        idx = refine_indices(self._data.potentiometry_opts.beta_flags)
        beta[idx] = (self._variables[self._slice_betas] + self._step[self._slice_betas]) / LN10
        return beta

    def _stoichiometry(self, extended=False):
        "Get stoichiometry array."
        number_components = self._data.stoichiometry.shape[0]
        if extended:
            return np.vstack((np.eye(number_components, dtype=int),
                              np.array(self._data.stoichiometry.T)))
        return self._data.stoichiometry.T

    def _titrations(self):
        """
        Iterate over the titrations.
        """
        yield from iter(self._data.potentiometry_opts.titrations)

    def _titration_parameters(self):
        itx = iter(self._variables[self._dof_beta:].tolist())
        itd = iter(self._step[self._dof_beta:].tolist())

        def select(c, flags):
            x = c.copy()
            for n, i in enumerate(flags):
                if i == Flags.REFINE:
                    x[n] = next(itx) + next(itd)
            return x

        for titration in self._titrations():
            c0 = select(titration.c0, titration.c0_flags)
            ct = select(titration.ct, titration.ct_flags)
            yield c0, ct
        
    def _calc_free_concs(self, initial=False, update=False) -> FArray:
        if initial or self._freeconcentration is None:
            _initial_guess = None
        else:
            _initial_guess = self._freeconcentration[:,:self._ncomponents]
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
        try:
            c, *_ = solve_equilibrium_equations(
                stoichiometry=self._data.stoichiometry,
                solid_stoichiometry=self._data.solid_stoichiometry,
                original_log_beta=log_beta,
                original_log_ks=self._data.log_ks,
                total_concentration=total_concentration,
                outer_fiexd_point_params=outer_fixed_point_params,
                initial_guess=_initial_guess,
                full=True)
        except excepts.DivergedIonicStrengthWarning as divergwarn:
            raise divergwarn
        else:
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
    fit_status, fit_result = libfit.levenberg_marquardt(bridge, debug=False)

    stdev, cor, cov = fit_final_calcs(fit_result['jacobian'], fit_result['residuals'], bridge.weights()) 
    bridge.incorporate_stdev(stdev)

    retval = bridge.final_result()
    retval.update(fit_result)
    retval['covariance'] = cov
    retval['correlation'] = cor

    return retval


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
    stdev = np.sqrt(cov_diag)
    lenD = len(cov_diag)
    correlation = covariance / np.sqrt(
        np.dot(cov_diag.reshape((lenD, 1)), cov_diag.reshape((1, lenD)))
    )
    return stdev, correlation, covariance


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
        if f == Flags.REFINE:  # refinable: return new value
            yield next(ity)
        elif f == Flags.CONSTANT:  # constant: return old value
            yield x[i]
        else:  # constrained: return or compute
            if i == ref_index[f]:
                val = next(ity)  # reference value: return new value
                ref_val[f] = val  # and store ref value
                yield val
            else:  # other: compute proportional value
                yield x[i] * ref_val[f] / x[ref_index[f]]

