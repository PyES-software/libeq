import json
from functools import cached_property, cache
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field, model_validator
from pydantic_numpy.typing import (
    Np1DArrayFp64,
    Np2DArrayFp64,
    Np2DArrayInt8,
    Np2DArrayInt16,
    Np2DArrayInt32,
    Np2DArrayInt64,
    Np1DArrayInt64,
    Np1DArrayBool,
)

from .utils import NumpyEncoder

from .parsers import parse_BSTAC_file
from .parsers import parse_superquad_file
from .consts import Flags


def _assemble_species_names(components, stoichiometry):
    """Assemble human-readable species names from component labels and stoichiometry.

    For each species (column of *stoichiometry*), a name is constructed by
    concatenating the component names raised to their respective stoichiometric
    coefficients.  Negative coefficients are represented as ``(OH)`` (or
    ``(OH)n`` for |n| > 1), consistent with the convention used in
    potentiometry software.

    Parameters
    ----------
    components : list of str
        Names of the chemical components (rows of *stoichiometry*).
    stoichiometry : numpy.ndarray
        2-D integer array of shape ``(n_components, n_species)`` containing the
        stoichiometric coefficients.

    Returns
    -------
    list of str
        List of ``n_species`` species name strings.
    """
    species_names = ["" for _ in range(stoichiometry.shape[1])]
    for i, row in enumerate(stoichiometry):
        for j, value in enumerate(row):
            if value < 0:
                species_names[j] += f"(OH){np.abs(value) if value != -1 else ''}"
            elif value > 0:
                species_names[j] += f"({components[i]}){value if value != 1 else ''}"

    return species_names


def _validate_or_error(
    field: str, condition: bool, error_message: str, errors: Dict[str, str]
) -> Dict[str, str]:
    """Validate a single condition and record an error message on failure.

    Evaluates *condition*; if it is ``False`` (or raises an exception), the
    *error_message* string is stored under *field* in *errors*.

    Parameters
    ----------
    field : str
        Key to use when storing the error in *errors*.
    condition : bool
        Boolean expression to test.  An exception during evaluation is treated
        as a failed condition.
    error_message : str
        Human-readable description of the validation failure.
    errors : dict of str to str
        Accumulator dictionary that is updated in-place and returned.

    Returns
    -------
    dict of str to str
        The updated *errors* dictionary.
    """
    try:
        if not condition:
            errors[field] = error_message
    except Exception as _:
        errors[field] = error_message

    return errors


class DistributionParameters(BaseModel):
    """Parameters for a species-distribution calculation over a pX range.

    Attributes
    ----------
    c0 : numpy.ndarray or None
        Initial total concentrations of each component (mol/L).  ``None`` if
        not yet set.
    c0_sigma : numpy.ndarray or None
        Standard deviations of the initial concentrations (mol/L).
    initial_log : float or None
        Starting value of the pX axis (negative log of the free concentration
        of the independent component).
    final_log : float or None
        Ending value of the pX axis.  Must be greater than *initial_log*.
    log_increments : float or None
        Step size along the pX axis.  Must be positive.
    independent_component : int or None
        Zero-based index of the component whose concentration is varied.
    cback : float
        Concentration of the background electrolyte (mol/L). Defaults to 0.
    """
    c0: Np1DArrayFp64 | None = None
    c0_sigma: Np1DArrayFp64 | None = None

    initial_log: float | None = None
    final_log: float | None = None
    log_increments: float | None = None

    independent_component: int | None = None

    cback: float = 0


class TitrationParameters(BaseModel):
    """Base parameters shared by titration-type calculations.

    Attributes
    ----------
    c0 : numpy.ndarray or None
        Initial total concentrations of each component in the titration vessel
        (mol/L).
    c0_sigma : numpy.ndarray or None
        Standard deviations of the initial concentrations (mol/L).
    ct : numpy.ndarray or None
        Concentrations of each component in the titrant solution (mol/L).
    ct_sigma : numpy.ndarray or None
        Standard deviations of the titrant concentrations (mol/L).
    c0_flags : list of int
        Refinement flags for each initial concentration.  Each element should
        be a :class:`~libeq.consts.Flags` value.
    ct_flags : list of int
        Refinement flags for each titrant concentration.
    c0back : float
        Background-electrolyte concentration in the initial solution (mol/L).
        Defaults to 0.0.
    ctback : float
        Background-electrolyte concentration in the titrant (mol/L).
        Defaults to 0.0.
    """
    c0: Np1DArrayFp64 | None = None
    c0_sigma: Np1DArrayFp64 | None = None
    ct: Np1DArrayFp64 | None = None
    ct_sigma: Np1DArrayFp64 | None = None
    c0_flags: List[int] = []
    ct_flags: List[int] = []

    c0back: float = 0.0
    ctback: float = 0.0


class SimulationTitrationParameters(TitrationParameters):
    """Parameters for a *simulated* (forward) titration calculation.

    Extends :class:`TitrationParameters` with the geometric parameters needed
    to generate the titre grid automatically.

    Attributes
    ----------
    v0 : float or None
        Initial volume of the solution in the titration vessel (mL).
    v_increment : float or None
        Volume of titrant added at each step (mL).  Must be positive.
    n_add : int or None
        Total number of titrant additions (i.e. number of data points).
    """
    v0: float | None = None
    v_increment: float | None = None
    n_add: int | None = None


class PotentiometryTitrationsParameters(TitrationParameters):
    """Parameters for a single potentiometric titration experiment.

    Extends :class:`TitrationParameters` with electrode, volume, and raw
    experimental data needed to fit equilibrium constants from EMF readings.

    Attributes
    ----------
    electro_active_compoment : int or None
        Zero-based index of the component detected by the electrode.
    e0 : float or None
        Standard electrode potential (mV).
    e0_sigma : float or None
        Standard deviation of the electrode potential (mV).
    slope : float or None
        Nernstian slope of the electrode (mV per log unit).
    v0 : float or None
        Initial volume of the solution (mL).
    v0_sigma : float or None
        Standard deviation of the initial volume (mL).
    v_add : numpy.ndarray or None
        Cumulative volumes of titrant added at each experimental point (mL).
    emf : numpy.ndarray or None
        Measured electrode potentials at each experimental point (mV).
    px_range : list of list of float or None
        List of ``[pX_min, pX_max]`` pairs that define the pX windows used to
        select active data points.  Points whose pX falls outside every window
        are excluded.
    ignored : numpy.ndarray of bool or None
        Boolean mask with ``True`` for points that have been manually excluded.
        Initialised automatically to all-``False`` when not provided.
    get_titre : numpy.ndarray
        Computed property returning the titrant volumes for non-ignored and
        in-range points.
    get_emf : numpy.ndarray
        Computed property returning the EMF values for non-ignored and
        in-range points.
    pX : numpy.ndarray
        Computed property returning ``(e0 - emf) / slope`` for active points.
    """
    electro_active_compoment: int | None = None         # index of the component
    e0: float | None = None                             # in mV
    e0_sigma: float | None = None                       # in mV
    slope: float | None = None                          # in mV
    v0: float | None = None                             # in mL
    v0_sigma: float | None = None                       # in mL
    v_add: Np1DArrayFp64 | None = None                  # in mL
    emf: Np1DArrayFp64 | None = None                    # in mV
    px_range: List[list[float]] | None = None           # dimmensionless
    ignored: Optional[Np1DArrayBool] = None

    @computed_field
    @cached_property
    def get_titre(self) -> Np1DArrayFp64:
        """Return the active titrant volumes.

        Returns
        -------
        numpy.ndarray
            Array of titrant volumes (mL) for the data points that are neither
            manually ignored nor outside the :attr:`px_range` windows.
        """
        return self.__get_property(self.v_add)

    @computed_field
    @cached_property
    def get_emf(self) -> Np1DArrayFp64:
        """Return the active EMF values.

        Returns
        -------
        numpy.ndarray
            Array of EMF values (mV) for the data points that are neither
            manually ignored nor outside the :attr:`px_range` windows.
        """
        return self.__get_property(self.emf)

    @computed_field
    @cached_property
    def pX(self) -> Np1DArrayFp64:
        """Return the pX values for the active data points.

        pX is defined as ``(e0 - emf) / slope`` and represents the negative
        logarithm of the free concentration of the electroactive component.

        Returns
        -------
        numpy.ndarray
            Array of pX values for the active (non-ignored, in-range) points.
        """
        return (self.e0 - self.get_emf) / self.slope

    @model_validator(mode='after')
    def set_ignored(self):
        """Initialise the *ignored* mask when it has not been set explicitly.

        If :attr:`ignored` is ``None``, it is replaced with an all-``False``
        boolean array of the same length as :attr:`v_add`, meaning no point
        is ignored by default.

        Returns
        -------
        PotentiometryTitrationsParameters
            The validated model instance.
        """
        if self.ignored is None:
            self.ignored = np.full_like(self.v_add, False, dtype=bool)
        return self

    def __get_property(self, prop: Np1DArrayFp64) -> Np1DArrayFp64:
        userused = np.logical_not(self.ignored)
        pxused = self.__getpxused()
        used = np.logical_and(userused, pxused)
        return np.extract(used, prop)

    def __getpxused(self):
        if not self.px_range:
            return np.full_like(self.ignored, True, dtype=bool)

        fullpx = (self.e0 - self.emf) / self.slope
        f = [np.logical_and(fullpx>pxmin, fullpx<pxmax) for pxmin, pxmax in self.px_range]
        return np.logical_or.reduce(f)


class PotentiometryOptions(BaseModel):
    """Top-level configuration for a potentiometric fitting run.

    Attributes
    ----------
    titrations : list of PotentiometryTitrationsParameters
        One entry per independent titration dataset.  Defaults to an empty list.
    weights : {"constants", "calculated", "given"}
        Weighting scheme applied to the residuals during optimisation.
        ``"constants"`` assigns equal weight to all points;
        ``"calculated"`` derives weights from the Gans *et al.* propagation
        formula; ``"given"`` uses the sigma values supplied in the data.
        Defaults to ``"constants"``.
    beta_flags : list of int
        Refinement flags (see :class:`~libeq.consts.Flags`) for each
        formation constant.
    conc_flags : list of int
        Refinement flags for analytical concentrations.
    pot_flags : list of int
        Refinement flags for potentiometric parameters (e.g. *E*:sub:`0`).
    """
    titrations: List[PotentiometryTitrationsParameters] = []
    weights: Literal["constants", "calculated", "given"] = "constants"
    beta_flags: List[int] = []
    conc_flags: List[int] = []
    pot_flags: List[int] = []


class SolverData(BaseModel):
    """Main data container for the equilibrium solver and optimizers.

    This Pydantic model aggregates all thermodynamic and experimental
    parameters needed by :func:`~libeq.solver.EqSolver` and
    :func:`~libeq.optimizers.PotentiometryOptimizer`.  Extra fields are
    forbidden to prevent silent mis-spellings.

    Parameters
    ----------
    components : list of str
        Names of the chemical components (master species).
    stoichiometry : numpy.ndarray
        Integer array of shape ``(n_components, n_species)`` with the
        stoichiometric coefficients of each soluble species.
    solid_stoichiometry : numpy.ndarray, optional
        Integer array of shape ``(n_components, n_solids)`` for solid phases.
        Defaults to an empty array (no solids).
    log_beta : numpy.ndarray
        log10 of the formation constants for soluble species.
    log_beta_sigma : numpy.ndarray, optional
        Uncertainties (1 σ) in *log_beta*.  Defaults to an empty array.
    log_beta_ref_dbh : numpy.ndarray, optional
        Reference ionic-strength correction coefficients for *log_beta*,
        shape ``(3, n_species)`` or ``(0, 2)`` when unused.
    log_ks : numpy.ndarray, optional
        log10 of the solubility products.
    log_ks_sigma : numpy.ndarray, optional
        Uncertainties (1 σ) in *log_ks*.
    log_ks_ref_dbh : numpy.ndarray, optional
        Reference ionic-strength correction coefficients for *log_ks*.
    charges : numpy.ndarray, optional
        Integer charges of each component.
    ionic_strength_dependence : bool, optional
        If ``True``, the Davies–Brønsted–Hückel correction is applied to
        adjust constants for the actual ionic strength.  Defaults to ``False``.
    reference_ionic_str_species : float or numpy.ndarray, optional
        Reference ionic strength at which *log_beta* values were determined.
        Defaults to 0.
    reference_ionic_str_solids : float or numpy.ndarray, optional
        Reference ionic strength at which *log_ks* values were determined.
        Defaults to 0.
    dbh_params : numpy.ndarray, optional
        Eight-element array with the Davies–Brønsted–Hückel parameters
        ``[a, b, c0, c1, d0, d1, e0, e1]``.  Defaults to zeros.
    temperature : float, optional
        Absolute temperature in Kelvin.  Defaults to 298.15 K.
    distribution_opts : DistributionParameters, optional
        Parameters for species-distribution calculations.
    titration_opts : SimulationTitrationParameters, optional
        Parameters for simulated titration calculations.
    potentiometry_opts : PotentiometryOptions, optional
        Parameters and experimental data for potentiometric fitting.

    Attributes
    ----------
    model_ready : tuple of (bool, dict)
        ``(True, {})`` when the minimum required fields are present.
    distribution_ready : tuple of (bool, dict)
        ``(True, {})`` when all distribution parameters are valid.
    titration_ready : tuple of (bool, dict)
        ``(True, {})`` when all simulated-titration parameters are valid.
    potentiometry_ready : tuple of (bool, dict)
        ``(True, {})`` when all potentiometric fitting parameters are valid.
    species_charges : numpy.ndarray
        Net charge of each soluble species.
    solid_charges : numpy.ndarray
        Net charge of each solid species.
    z_star_species : numpy.ndarray
        Auxiliary charge term ``Σ p_ij * z_j^2 - z_species^2`` for soluble
        species, used in Brønsted–Hückel corrections.
    p_star_species : numpy.ndarray
        Sum of stoichiometric coefficients minus 1 for each soluble species.
    z_star_solids : numpy.ndarray
        Analogous charge term for solid species.
    p_star_solids : numpy.ndarray
        Sum of stoichiometric coefficients for each solid species.
    dbh_values : dict
        Pre-computed Davies–Brønsted–Hückel correction arrays keyed by
        ``"species"`` and ``"solids"``.
    species_names : list of str
        Auto-generated names for all species (components + complexes).
    solids_names : list of str
        Auto-generated names for all solid species.
    nc : int
        Number of components.
    ns : int
        Number of soluble species.
    nf : int
        Number of solid (precipitating) species.
    """

    distribution_opts: DistributionParameters = DistributionParameters()
    titration_opts: SimulationTitrationParameters = SimulationTitrationParameters()
    potentiometry_opts: PotentiometryOptions = PotentiometryOptions()

    components: List[str]
    stoichiometry: Np2DArrayInt8 | Np2DArrayInt16 | Np2DArrayInt32 | Np2DArrayInt64
    solid_stoichiometry:  Np2DArrayInt8 | Np2DArrayInt16 | Np2DArrayInt32 | Np2DArrayInt64 = np.array([], dtype=int)
    log_beta: Np1DArrayFp64
    log_beta_sigma: Np1DArrayFp64 = np.array([])
    log_beta_ref_dbh: Np2DArrayFp64 = np.empty((0, 2))
    log_ks: Np1DArrayFp64 = np.array([])
    log_ks_sigma: Np1DArrayFp64 = np.array([])
    log_ks_ref_dbh: Np2DArrayFp64 = np.empty((0, 2))

    charges: Np1DArrayInt64 = np.array([])

    ionic_strength_dependence: bool = False
    reference_ionic_str_species: Np1DArrayFp64 | float = 0
    reference_ionic_str_solids: Np1DArrayFp64 | float = 0
    dbh_params: Np1DArrayFp64 = np.zeros(8)

    temperature: float = 298.15                   # Kelvin

    @computed_field
    @cached_property
    def model_ready(self) -> tuple[bool, dict[str, str]]:
        """Check whether the minimal fields for any solver run are present.

        Returns
        -------
        tuple of (bool, dict of str to str)
            ``(True, {})`` if the model is valid; ``(False, errors)`` where
            *errors* maps field names to human-readable error descriptions.
        """
        fields_to_check = [
            {
                "field": "Components",
                "condition": len(self.components) > 0,
                "error_message": "At least one component must be provided",
            },
            {
                "field": "Stoichiometry",
                "condition": self.stoichiometry.size > 0,
                "error_message": "Stoichiometry matrix must be provided",
            },
        ]

        errors = {}
        for check in fields_to_check:
            errors = _validate_or_error(**check, errors=errors)

        if errors:
            return False, errors
        else:
            return True, errors

    @computed_field
    @cached_property
    def distribution_ready(self) -> tuple[bool, dict[str, str]]:
        """Check whether all parameters for a distribution calculation are set.

        Returns
        -------
        tuple of (bool, dict of str to str)
            ``(True, {})`` if all distribution parameters are valid;
            ``(False, errors)`` otherwise.
        """
        fields_to_check = [
            {
                "field": "Initial pX",
                "condition": (self.distribution_opts.initial_log is not None),
                "error_message": "Initial log must be provided",
            },
            {
                "field": "Final pX",
                "condition": (self.distribution_opts.final_log is not None)
                and (self.distribution_opts.final_log > self.distribution_opts.initial_log),
                "error_message": "Final log must be provided and it must be bigger that the initial value",
            },
            {
                "field": "pX increments",
                "condition": (self.distribution_opts.log_increments is not None)
                and (self.distribution_opts.log_increments > 0),
                "error_message": "Log increments must be provided and must be positive",
            },
            {
                "field": "Independent component",
                "condition": self.distribution_opts.independent_component is not None,
                "error_message": "Independent component must be provided",
            },
        ]

        errors = {}
        for check in fields_to_check:
            errors = _validate_or_error(**check, errors=errors)

        if errors:
            return False, errors
        else:
            return True, errors

    @computed_field
    @cached_property
    def titration_ready(self) -> tuple[bool, dict[str, str]]:
        """Check whether all parameters for a simulated titration are set.

        Returns
        -------
        tuple of (bool, dict of str to str)
            ``(True, {})`` if all titration parameters are valid;
            ``(False, errors)`` otherwise.
        """
        fields_to_check = [
            {
                "field": "Initial Volume",
                "condition": (self.titration_opts.v0 is not None)
                and (self.titration_opts.v0 > 0),
                "error_message": "Initial volume must be provided",
            },
            {
                "field": "Volume increments",
                "condition": (self.titration_opts.v_increment is not None)
                and (self.titration_opts.v_increment > 0),
                "error_message": "Volume increments must be provided",
            },
            {
                "field": "Number of points",
                "condition": (self.titration_opts.n_add is not None)
                and (self.titration_opts.n_add > 0),
                "error_message": "Number of titration points must be provided",
            },
            {
                "field": "Titrant concentration",
                "condition": (self.titration_opts.ct is not None)
                and (self.titration_opts.ct != 0).any(),
                "error_message": "Titrant concentrations must be provided",
            },
        ]

        errors = {}
        for check in fields_to_check:
            errors = _validate_or_error(**check, errors=errors)

        if errors:
            return False, errors
        else:
            return True, errors

    @computed_field
    @cached_property
    def potentiometry_ready(self) -> tuple[bool, dict[str, str]]:
        """Check whether all parameters for a potentiometric fitting run are set.

        Returns
        -------
        tuple of (bool, dict of str to str)
            ``(True, {})`` if all potentiometry parameters are valid;
            ``(False, errors)`` otherwise.
        """
        fields_to_check = [
            {
                "field": "Initial Volume",
                "condition": np.fromiter(
                    (t.v0 is not None for t in self.potentiometry_opts.titrations), bool
                ).all()
                and np.fromiter(
                    (t.v0 > 0 for t in self.potentiometry_opts.titrations), bool
                ).all(),
                "error_message": "Initial volume must be provided for all titrations",
            },
            {
                "field": "Volume sigma",
                "condition": np.fromiter(
                    (
                        t.v0_sigma is not None
                        for t in self.potentiometry_opts.titrations
                    ),
                    bool,
                ).all()
                and np.fromiter(
                    (t.v0_sigma > 0 for t in self.potentiometry_opts.titrations), bool
                ).all(),
                "error_message": "Volume standard deviation must be provided for all titrations",
            },
            {
                "field": "E0",
                "condition": np.fromiter(
                    (t.e0 is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                and np.fromiter(
                    (t.e0 > 0 for t in self.potentiometry_opts.titrations), bool
                ).all(),
                "error_message": "Electrode potential must be provided for all titrations",
            },
            {
                "field": "E Sigma",
                "condition": np.fromiter(
                    (
                        t.e0_sigma is not None
                        for t in self.potentiometry_opts.titrations
                    ),
                    bool,
                ).all()
                and np.fromiter(
                    (t.e0_sigma > 0 for t in self.potentiometry_opts.titrations), bool
                ).all(),
                "error_message": "Electrode standard deviation must be provided for all titrations",
            },
            {
                "field": "Slope",
                "condition": np.fromiter(
                    (t.slope is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                and np.fromiter(
                    (t.slope != 0 for t in self.potentiometry_opts.titrations), bool
                ).all(),
                "error_message": "Slope must be provided for all titrations",
            },
            {
                "field": "Titrant concentration",
                "condition": np.fromiter(
                    (t.ct is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                and np.fromiter(
                    ((t.ct != 0).any() for t in self.potentiometry_opts.titrations),
                    bool,
                ).all(),
                "error_message": "Titrant concentrations must be provided",
            },
            {
                "field": "Initial concentration",
                "condition": np.fromiter(
                    (t.c0 is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                and np.fromiter(
                    ((t.c0 != 0).any() for t in self.potentiometry_opts.titrations),
                    bool,
                ).all(),
                "error_message": "Initial concentrations must be provided",
            },
            {
                "field": "Ignored points",
                "condition": np.fromiter(
                    (t.ignored is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                and np.fromiter(
                    (
                        ~(t.ignored == True).all()  # noqa: E712
                        for t in self.potentiometry_opts.titrations
                    ),
                    bool,
                ).all(),
                "error_message": "At least one point must not be ignored",
            },
            {
                "field": "Number of points",
                "condition": np.fromiter(
                    (t.v_add is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                and np.fromiter(
                    (t.v_add.size > 0 for t in self.potentiometry_opts.titrations),
                    bool,
                ).all(),
                "error_message": "Titrations need at least one point",
            },
            {
                "field": "Optimization targets",
                "condition": np.bool_(
                    np.array((self.potentiometry_opts.beta_flags))
                ).any(),
                "error_message": "No optimization targets selected",
            },
        ]

        errors = {}
        for check in fields_to_check:
            errors = _validate_or_error(**check, errors=errors)

        if errors:
            return False, errors
        else:
            return True, errors

    @computed_field
    @cached_property
    def species_charges(self) -> Np1DArrayFp64:
        """Net charge of each soluble species.

        Computed as the inner product of the stoichiometry matrix and the
        component charges: ``Σ_j p_ij * z_j`` for each species *i*.

        Returns
        -------
        numpy.ndarray
            1-D float array of length ``ns`` with the net charge of each
            soluble species.
        """
        return (self.stoichiometry * self.charges[:, np.newaxis]).sum(axis=0)

    @computed_field
    @cached_property
    def solid_charges(self) -> Np1DArrayFp64:
        """Net charge of each solid (precipitating) species.

        Computed analogously to :attr:`species_charges` but using
        :attr:`solid_stoichiometry`.

        Returns
        -------
        numpy.ndarray
            1-D float array of length ``nf`` with the net charge of each solid
            species.
        """
        return (self.solid_stoichiometry * self.charges[:, np.newaxis]).sum(axis=0)

    @computed_field(repr=False)
    @cached_property
    def z_star_species(self) -> Np1DArrayFp64:
        """Auxiliary charge term for soluble species used in ion-strength corrections.

        Defined as ``Σ_j p_ij * z_j^2 - z_species_i^2``, where *z_j* are the
        component charges and *z_species_i* is the net charge of species *i*.

        Returns
        -------
        numpy.ndarray
            1-D float array of length ``ns``.
        """
        return (self.stoichiometry * (self.charges[:, np.newaxis] ** 2)).sum(
            axis=0
        ) - self.species_charges**2

    @computed_field(repr=False)
    @cached_property
    def p_star_species(self) -> Np1DArrayFp64:
        """Sum of stoichiometric coefficients minus one for each soluble species.

        Defined as ``Σ_j p_ij - 1``.  Used in Davies–Brønsted–Hückel
        corrections.

        Returns
        -------
        numpy.ndarray
            1-D float array of length ``ns``.
        """
        return self.stoichiometry.sum(axis=0) - 1

    @computed_field(repr=False)
    @cached_property
    def z_star_solids(self) -> Np1DArrayFp64:
        """Auxiliary charge term for solid species used in ion-strength corrections.

        Defined analogously to :attr:`z_star_species` but computed from
        :attr:`solid_stoichiometry`.

        Returns
        -------
        numpy.ndarray
            1-D float array of length ``nf``.
        """
        return (self.solid_stoichiometry * (self.charges[:, np.newaxis] ** 2)).sum(
            axis=0
        ) - self.solid_charges**2

    @computed_field(repr=False)
    @cached_property
    def p_star_solids(self) -> Np1DArrayFp64:
        """Sum of stoichiometric coefficients for each solid species.

        Defined as ``Σ_j q_ij``.  Used in Davies–Brønsted–Hückel corrections
        for solid phases.

        Returns
        -------
        numpy.ndarray
            1-D float array of length ``nf``.
        """
        return self.solid_stoichiometry.sum(axis=0)

    @computed_field
    @cached_property
    def dbh_values(self) -> Dict[str, Np1DArrayFp64]:
        """Pre-computed Davies–Brønsted–Hückel correction arrays.

        Calculates the ionic-strength-dependent correction coefficients for
        both soluble species and solid phases using :attr:`dbh_params`.  Any
        species that have per-species reference values stored in
        :attr:`log_beta_ref_dbh` or :attr:`log_ks_ref_dbh` override the
        globally computed values.

        Returns
        -------
        dict of str to dict
            Outer keys are ``"species"`` and ``"solids"``.  Each inner dict
            contains the arrays ``"azast"``, ``"adh"``, ``"bdh"``,
            ``"cdh"``, ``"ddh"``, ``"edh"``, and ``"fib"``.
        """
        result = dict()
        for phase, iref, per_species_cde, z, p in zip(
            ("species", "solids"),
            (self.reference_ionic_str_species, self.reference_ionic_str_solids),
            (self.log_beta_ref_dbh, self.log_ks_ref_dbh),
            (self.z_star_species, self.z_star_solids),
            (self.p_star_species, self.p_star_solids),
        ):
            dbh_values = dict()
            dbh_values["azast"] = self.dbh_params[0] * z
            dbh_values["adh"] = self.dbh_params[0]
            dbh_values["bdh"] = self.dbh_params[1]
            dbh_values["cdh"] = self.dbh_params[2] * p + self.dbh_params[3] * z
            dbh_values["ddh"] = self.dbh_params[4] * p + self.dbh_params[5] * z
            dbh_values["edh"] = self.dbh_params[6] * p + self.dbh_params[7] * z
            dbh_values["fib"] = np.sqrt(iref) / (1 + self.dbh_params[1] * np.sqrt(iref))

            not_zero_columns = np.where(np.any(per_species_cde != 0, axis=0))[0]
            for i in not_zero_columns:
                dbh_values["cdh"][i] = per_species_cde[0][i]
                dbh_values["ddh"][i] = per_species_cde[1][i]
                dbh_values["edh"][i] = per_species_cde[2][i]

            result[phase] = dbh_values
        return result

    @computed_field
    @cached_property
    def species_names(self) -> List[str]:
        """Auto-generated names for all species (components and complexes).

        The list starts with the component names followed by the names of the
        soluble complexes assembled by :func:`_assemble_species_names`.

        Returns
        -------
        list of str
            Names of all ``nc + ns`` species.
        """
        return self.components + _assemble_species_names(
            self.components, self.stoichiometry
        )

    @computed_field
    @cached_property
    def solids_names(self) -> List[str]:
        """Auto-generated names for all solid (precipitating) species.

        Returns
        -------
        list of str
            Names of all ``nf`` solid species assembled by
            :func:`_assemble_species_names`.
        """
        return _assemble_species_names(self.components, self.solid_stoichiometry)

    @computed_field
    @cached_property
    def nc(self) -> int:
        """Number of chemical components.

        Returns
        -------
        int
            Number of rows in :attr:`stoichiometry`.
        """
        return self.stoichiometry.shape[0]

    @computed_field
    @cached_property
    def ns(self) -> int:
        """Number of soluble species (complexes).

        Returns
        -------
        int
            Number of columns in :attr:`stoichiometry`.
        """
        return self.stoichiometry.shape[1]

    @computed_field
    @cached_property
    def nf(self) -> int:
        """Number of solid (precipitating) species.

        Returns
        -------
        int
            Number of columns in :attr:`solid_stoichiometry`.
        """
        return self.solid_stoichiometry.shape[1]

    @classmethod
    def load_from_bstac(cls, file_path: str) -> "SolverData":
        """Load a :class:`SolverData` instance from a BSTAC-format file.

        Reads and parses the BSTAC file at *file_path*, translating its
        sections into the fields expected by :class:`SolverData`, including
        components, stoichiometry, log10 stability constants,
        ionic-strength correction parameters, and titration data.

        Parameters
        ----------
        file_path : str
            Path to the BSTAC input file.

        Returns
        -------
        SolverData
            A fully initialised :class:`SolverData` instance.
        """
        data = dict()
        with open(file_path, "r") as file:
            lines = file.readlines()
        parsed_data = parse_BSTAC_file(lines)

        temperature = parsed_data["TEMP"]
        data["temperature"] = temperature
        data["stoichiometry"] = np.array(
            [
                [d[key] for key in d if key.startswith("IX")]
                for d in parsed_data["species"]
            ], dtype=int
        ).T
        data["solid_stoichiometry"] = np.empty(
            (data["stoichiometry"].shape[0], 0), dtype=int
        )
        data["log_beta"] = np.array([d["BLOG"] for d in parsed_data["species"]])

        data["charges"] = np.array(parsed_data.get("charges", []))
        data["components"] = parsed_data["comp_name"]
        data["ionic_strength_dependence"] = parsed_data["ICD"] != 0
        if data["ionic_strength_dependence"]:
            data["reference_ionic_str_species"] = np.array(
                [parsed_data["IREF"] for _ in range(data["stoichiometry"].shape[1])]
            )
            data["reference_ionic_str_solids"] = np.array(
                [
                    parsed_data["IREF"]
                    for _ in range(data["solid_stoichiometry"].shape[1])
                ]
            )
            data["dbh_params"] = [
                parsed_data[i] for i in ["AT", "BT", "c0", "c1", "d0", "d1", "e0", "e1"]
            ]
        titration_options = [
            PotentiometryTitrationsParameters(
                c0=np.array([c["C0"] for c in t["components_concentrations"]]),
                ct=np.array([c["CTT"] for c in t["components_concentrations"]]),
                electro_active_compoment=(
                    t["titration_comp_settings"][1]
                    if t["titration_comp_settings"][1] != 0
                    else len(data["components"]) - 1
                ),
                e0=t["potential_params"][0],
                e0_sigma=t["potential_params"][1],
                slope=(
                    t["potential_params"][4]
                    if t["potential_params"][4] != 0
                    else (temperature + 273.15) / 11.6048 * 2.303
                ),
                v0=t["v_params"][0],
                v0_sigma=t["v_params"][1],
                ignored=np.array(t["ignored"]),
                v_add=np.array(t["volume"]),
                emf=np.array(t["potential"]),
                c0back=t["background_params"][0] if "background_params" in t else 0,
                ctback=t["background_params"][1] if "background_params" in t else 0,
                px_range=[[parsed_data["PHI"], parsed_data["PHF"]]],
            )
            for t in parsed_data["titrations"]
        ]

        weights = parsed_data.get("MODE", 1)
        match weights:
            case 0:
                weights = "calculated"
            case 1:
                weights = "constants"
            case 2:
                weights = "given"
            case _:
                raise ValueError("Invalid MODE value")

        data["potentiometry_opts"] = PotentiometryOptions(
            titrations=titration_options,
            weights=weights,
            beta_flags=[s["KEY"] for s in parsed_data["species"]],
            conc_flags=[],
            pot_flags=[],
        )
        return cls(**data)

    @classmethod
    def load_from_superquad(cls, file_path: str) -> "SolverData":
        """
        Open superquad file and load it into the class.
        """
        flags_translate = {0: Flags.CONSTANT, 1: Flags.REFINE}

        parsed_data = parse_superquad_file(file_path)
        data = {}
        temp = parsed_data.get('temperature', 298.15)
        data['temperature'] = temp
        data['stoichiometry'] = np.array(parsed_data['stoichiometry'], dtype=int).T
        data["solid_stoichiometry"] = np.empty(
            (data["stoichiometry"].shape[0], 0), dtype=int)
        data["log_beta"] = np.array(parsed_data["log_beta"], dtype=float)
        data['components'] = parsed_data['components']
        data['charges'] = np.zeros(len(parsed_data['components']), dtype=int)

        titration_options = [
            PotentiometryTitrationsParameters(
                c0=np.array(t['initial amount'], dtype=float) / t['starting volume'],
                ct=np.array(t['buret concentration'], dtype=float),
                electro_active_compoment=t['electroactive']-1,
                e0=t['standard potential'],
                e0_sigma=t['potential error'],
                slope=temp / 11.6048 * 2.303,
                v0=t['starting volume'],
                v0_sigma=t["volume error"],
                v_add=np.array(t["titre"], dtype=float),
                emf=np.array(t["potential"], dtype=float),
                c0back=t["background_params"][0] if "background_params" in t else 0,
                ctback=t["background_params"][1] if "background_params" in t else 0,
                ignored=len(t['titre']) * [False],
                px_range=[[0, 0]]
            )
            for t in parsed_data["titrations"]
        ]

        data["potentiometry_opts"] = PotentiometryOptions(
            titrations=titration_options,
            weights="calculated",
            beta_flags=[flags_translate[f] for f in parsed_data["beta flags"]],
            conc_flags=[],
            pot_flags=[],
        )
        return cls(**data)

    @classmethod
    def load_from_pyes(cls, pyes_data: str | dict) -> "SolverData":
        """Load a :class:`SolverData` instance from a PyES project file or dict.

        Parses the JSON structure used by the PyES graphical application and
        maps its fields to the corresponding :class:`SolverData` attributes,
        including soluble and solid species models, ionic-strength correction
        parameters, distribution parameters, simulated titration parameters,
        and potentiometric titration data.

        Parameters
        ----------
        pyes_data : str or dict
            Either a path to a PyES JSON project file or an already-parsed
            Python dict with the same structure.

        Returns
        -------
        SolverData
            A fully initialised :class:`SolverData` instance.
        """
        if isinstance(pyes_data, str):
            with open(pyes_data, "r") as file:
                pyes_data = json.load(file)
        data = dict()
        data["components"] = list(pyes_data["compModel"]["Name"].values())

        data["stoichiometry"] = np.vstack(
            [
                list(pyes_data["speciesModel"][col].values())
                for col in data["components"]
            ]
        )
        data["log_beta"] = np.array(list(pyes_data["speciesModel"]["LogB"].values()))
        data["log_beta_sigma"] = np.array(
            list(pyes_data["speciesModel"]["Sigma"].values())
        )
        data["log_beta_ref_dbh"] = np.vstack(
            (
                list(pyes_data["speciesModel"]["CGF"].values()),
                list(pyes_data["speciesModel"]["DGF"].values()),
                list(pyes_data["speciesModel"]["EGF"].values()),
            )
        )
        data["solid_stoichiometry"] = np.array(
            [
                list(pyes_data["solidSpeciesModel"][col].values())
                for col in data["components"]
            ], dtype=int
        )
        data["log_ks"] = np.array(
            list(pyes_data["solidSpeciesModel"]["LogKs"].values())
        )
        data["log_ks_sigma"] = np.array(
            list(pyes_data["solidSpeciesModel"]["Sigma"].values())
        )
        data["log_ks_ref_dbh"] = np.vstack(
            (
                list(pyes_data["solidSpeciesModel"]["CGF"].values()),
                list(pyes_data["solidSpeciesModel"]["DGF"].values()),
                list(pyes_data["solidSpeciesModel"]["EGF"].values()),
            )
        )

        data["charges"] = np.array(list(pyes_data["compModel"]["Charge"].values()))
        data["ionic_strength_dependence"] = pyes_data["imode"] != 0
        data["reference_ionic_str_species"] = np.array(
            list(pyes_data["speciesModel"]["Ref. Ionic Str."].values())
        )
        data["reference_ionic_str_species"] = np.where(
            data["reference_ionic_str_species"] == 0,
            pyes_data["ris"],
            data["reference_ionic_str_species"],
        )

        data["reference_ionic_str_solids"] = np.array(
            list(pyes_data["solidSpeciesModel"]["Ref. Ionic Str."].values())
        )
        data["reference_ionic_str_solids"] = np.where(
            data["reference_ionic_str_solids"] == 0,
            pyes_data["ris"],
            data["reference_ionic_str_solids"],
        )

        data["dbh_params"] = [
            pyes_data[name] for name in ["a", "b", "c0", "c1", "d0", "d1", "e0", "e1"]
        ]

        data["distribution_opts"] = DistributionParameters(
            c0=np.array(list(pyes_data.get("concModel", {}).get("C0", {}).values())),
            c0_sigma=np.array(
                list(pyes_data.get("concModel", {}).get("Sigma C0", {}).values())
            ),
            initial_log=pyes_data.get("initialLog", 0.0),
            final_log=pyes_data.get("finalLog", 0.0),
            log_increments=pyes_data.get("logInc", 0.0),
            independent_component=pyes_data.get("ind_comp", 0),
            cback=pyes_data.get("cback", 0.0),
        )

        data["titration_opts"] = SimulationTitrationParameters(
            c0=np.array(list(pyes_data.get("concModel", {}).get("C0", {}).values())),
            c0_sigma=np.array(
                list(pyes_data.get("concModel", {}).get("Sigma C0", {}).values())
            ),
            ct=np.array(list(pyes_data.get("concModel", {}).get("CT", {}).values())),
            ct_sigma=np.array(
                list(pyes_data.get("concModel", {}).get("Sigma CT", {}).values())
            ),
            v0=pyes_data.get("v0", 0.0),
            v_increment=pyes_data.get("vinc", 0.0),
            n_add=pyes_data.get("nop", 0),
            c0back=pyes_data.get("c0back", 0.0),
            ctback=pyes_data.get("ctback", 0.0),
        )

        if "potentiometry_data" in pyes_data:
            potentiometry_data = pyes_data["potentiometry_data"]
            titrations = []
            if potentiometry_data["weightsMode"] == 0:
                weights = "constants"
            elif potentiometry_data["weightsMode"] == 1:
                weights = "calculated"
            elif potentiometry_data["weightsMode"] == 2:
                weights = "given"

            c0flags=([Flags.REFINE if v else Flags.CONSTANT
                     for v in g]
                     for g in zip(*potentiometry_data["conc_refine_flags"]))

            for t in potentiometry_data["titrations"]:
                titrations.append(
                    PotentiometryTitrationsParameters(
                        c0=np.array(list(t.get("concView", {}).get("C0", {}).values()), dtype=float),
                        c0_flags=next(c0flags),
                        ct=np.array(list(t.get("concView", {}).get("CT", {}).values()), dtype=float),
                        c0_sigma=np.array(
                            list(t.get("concView", {}).get("Sigma C0", {}).values()), dtype=float
                        ),
                        ct_sigma=np.array(
                            list(t.get("concView", {}).get("Sigma CT", {}).values()), dtype=float
                        ),
                        electro_active_compoment=t["electroActiveComponent"],
                        e0=t["e0"],
                        e0_sigma=t["eSigma"],
                        slope=t["slope"],
                        v0=t["initialVolume"],
                        v0_sigma=t["vSigma"],
                        ignored=np.array(
                            list(t.get("titrationView", {}).get("0", {}).values()), dtype=bool
                        ),
                        v_add=np.array(
                            list(t.get("titrationView", {}).get("1", {}).values())
                        ),
                        emf=np.array(
                            list(t.get("titrationView", {}).get("2", {}).values())
                        ),
                        c0back=t.get("c0back", 0.0),
                        ctback=t.get("ctback", 0.0),
                        px_range=[px_range for px_range in t["pxRange"]],
                    )
                )
            data["potentiometry_opts"] = PotentiometryOptions(
                titrations=titrations,
                weights=weights,
                beta_flags=[Flags.REFINE if v else Flags.CONSTANT
                            for v in potentiometry_data["beta_refine_flags"]],
                conc_flags=[],
                pot_flags=[],
            )
            # data["potentiometry_opts"].conc_flags = [
            #     "constant" if v == 0 else "calculated" if v == 1 else "given"
            #     for v in data["potentiometry_opts"].conc_flags
            # ]
        return cls(**data)

    def to_pyes(self, format: Literal["dict", "json"] = "dict") -> dict[str, Any] | str:
        """Serialise the instance to a PyES-compatible representation.

        Converts all model data into the JSON structure expected by the PyES
        graphical application.  The output can be returned as a Python
        :class:`dict` or as a JSON-formatted string.

        Parameters
        ----------
        format : {"dict", "json"}, optional
            Output format.  ``"dict"`` returns a Python dict; ``"json"``
            serialises it to a JSON string using :class:`~libeq.utils.NumpyEncoder`.
            Defaults to ``"dict"``.

        Returns
        -------
        dict or str
            The PyES project data as a dict or JSON string, depending on
            *format*.
        """
        if isinstance(self.reference_ionic_str_species, (float, int)):
            species_ref_ionic_str = {
                i: self.reference_ionic_str_species for i in range(self.ns)
            }
        else:
            species_ref_ionic_str = {
                i: ris for i, ris in enumerate(self.reference_ionic_str_species)
            }
        soluble_model = {
            f"{comp_name}": {i: v for i, v in enumerate(row)}
            for comp_name, row in zip(self.components, self.stoichiometry)
        }

        if isinstance(self.reference_ionic_str_solids, (float, int)):
            solid_ref_ionic_str = {
                i: self.reference_ionic_str_solids for i in range(self.nf)
            }
        else:
            solid_ref_ionic_str = {
                i: ris for i, ris in enumerate(self.reference_ionic_str_solids)
            }
        solid_model = {
            f"{comp_name}": {i: v for i, v in enumerate(row)}
            for comp_name, row in zip(self.components, self.solid_stoichiometry)
        }

        if self.potentiometry_ready[0]:
            if self.potentiometry_opts.weights == "constants":
                weights_mode = 0
            elif self.potentiometry_opts.weights == "calculated":
                weights_mode = 1
            elif self.potentiometry_opts.weights == "given":
                weights_mode = 2

            potentiometry_section = {
                "potentiometry_data": {
                    "weightsMode": weights_mode,
                    "beta_refine_flags": [
                        bool(f) for f in self.potentiometry_opts.beta_flags
                    ],
                    "titrations": [
                        {
                            "concView": {
                                "C0": {i: c for i, c in zip(self.components, t.c0)},
                                "CT": {i: c for i, c in zip(self.components, t.ct)},
                                "Sigma C0": {
                                    i: 0 for i, _ in zip(self.components, t.c0)
                                },
                                "Sigma CT": {
                                    i: 0 for i, _ in zip(self.components, t.ct)
                                },
                            },
                            "electroActiveComponent": t.electro_active_compoment,
                            "e0": t.e0,
                            "eSigma": t.e0_sigma,
                            "slope": t.slope,
                            "initialVolume": t.v0,
                            "vSigma": t.v0_sigma,
                            "pxRange": t.px_range,
                            "titrationView": {
                                "0": {i: v for i, v in enumerate(t.ignored)},
                                "1": {i: v for i, v in enumerate(t.v_add)},
                                "2": {i: v for i, v in enumerate(t.emf)},
                                "3": {i: 0 for i, _ in enumerate(t.emf)},
                            },
                        }
                        for t in self.potentiometry_opts.titrations
                    ],
                },
            }
        else:
            potentiometry_section = {}

        if self.distribution_ready[0]:
            distribution_section = {
                "ind_comp": self.distribution_opts.independent_component,
                "initialLog": self.distribution_opts.initial_log,
                "finalLog": self.distribution_opts.final_log,
                "logInc": self.distribution_opts.log_increments,
                "cback": self.distribution_opts.cback,
            }
        else:
            distribution_section = {}

        if self.titration_ready[0]:
            titration_section = {
                "v0": self.titration_opts.v0,
                "initv": self.titration_opts.v0,
                "vinc": self.titration_opts.v_increment,
                "nop": self.titration_opts.n_add,
                "c0back": self.titration_opts.c0back,
                "ctback": self.titration_opts.ctback,
            }
        else:
            titration_section = {}

        if self.titration_ready[0] or self.distribution_ready[0]:
            conc_section = {
                "concModel": {
                    "C0": {i: c0 for i, c0 in enumerate(self.distribution_opts.c0)},
                    "Sigma C0": {
                        i: sigma
                        for i, sigma in enumerate(self.distribution_opts.c0_sigma)
                    },
                    "CT": {i: ct for i, ct in enumerate(self.titration_opts.ct)},
                    "Sigma CT": {
                        i: sigma for i, sigma in enumerate(self.titration_opts.ct_sigma)
                    },
                },
            }
        else:
            conc_section = {}

        if self.log_beta_sigma.size == 0:
            self.log_beta_sigma = np.zeros(self.ns)

        if self.log_ks_sigma.size == 0:
            self.log_ks_sigma = np.zeros(self.nf)

        data = {
            "check": "PyES project file --- DO NOT MODIFY THIS LINE!",
            "nc": self.nc,
            "ns": self.ns,
            "np": self.nf,
            "emode": False,
            "imode": 1 if self.ionic_strength_dependence else 0,
            "ris": 0.0,
            "a": self.dbh_params[0],
            "b": self.dbh_params[1],
            "c0": self.dbh_params[2],
            "c1": self.dbh_params[3],
            "d0": self.dbh_params[4],
            "d1": self.dbh_params[5],
            "e0": self.dbh_params[6],
            "e1": self.dbh_params[7],
            "dmode": 0,
            "compModel": {
                "Name": {i: name for i, name in enumerate(self.components)},
                "Charge": {i: int(charge) for i, charge in enumerate(self.charges)},
            },
            "speciesModel": {
                "Ignored": {i: False for i in range(self.ns)},
                "Name": {
                    i: name for i, name in enumerate(self.species_names[self.nc :])
                },
                "LogB": {i: log_b for i, log_b in enumerate(self.log_beta)},
                "Sigma": {i: sigma for i, sigma in enumerate(self.log_beta_sigma)},
                "Ref. Ionic Str.": species_ref_ionic_str,
                "CGF": {i: c for i, c in enumerate(self.dbh_values["species"]["cdh"])},
                "DGF": {i: d for i, d in enumerate(self.dbh_values["species"]["ddh"])},
                "EGF": {i: e for i, e in enumerate(self.dbh_values["species"]["edh"])},
                **soluble_model,
                "Ref. Comp.": {i: self.components[0] for i in range(self.ns)},
            },
            "solidSpeciesModel": {
                "Ignored": {i: False for i in range(self.nf)},
                "Name": {i: name for i, name in enumerate(self.solids_names)},
                "LogKs": {i: log_ks for i, log_ks in enumerate(self.log_ks)},
                "Sigma": {i: sigma for i, sigma in enumerate(self.log_ks_sigma)},
                "Ref. Ionic Str.": solid_ref_ionic_str,
                "CGF": {i: c for i, c in enumerate(self.dbh_values["solids"]["cdh"])},
                "DGF": {i: d for i, d in enumerate(self.dbh_values["solids"]["ddh"])},
                "EGF": {i: e for i, e in enumerate(self.dbh_values["solids"]["edh"])},
                **solid_model,
                "Ref. Comp.": {i: self.components[0] for i in range(self.nf)},
            },
            **conc_section,
            **titration_section,
            **distribution_section,
            **potentiometry_section,
        }

        return (
            data if format == "dict" else json.dumps(data, indent=4, cls=NumpyEncoder)
        )
