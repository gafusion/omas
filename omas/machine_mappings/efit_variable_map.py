r"""
IMAS <-> MDSplus variable map derived from _efit.json (see _common.py for the
PYTHON-mapped constraint helpers referenced below).

Each tuple is (imas_path, mdsplus_path):
  - imas_path is the ODS path with all ".:." AoS wildcards removed (imas_composer format)
  - mdsplus_path is the MDSplus path within the EFIT tree, rooted at r'\TOP:...'
    (i.e. with the "{EFIT_tree}::" prefix stripped and "." replaced by ":")

Entries with no underlying MDSplus signal (constants, EVAL'd tree name/index, or
array-length placeholders backed only by a VALUE) use None for the MDSplus path.

COCOS is determined separately (__cocos_rules__ in _efit.json) from the signs of
\TOP:RESULTS:GEQDSK:BCENTR and \TOP:RESULTS:GEQDSK:CPASMA (MDS_gEQDSK_COCOS_identify
in _common.py); it has no single ODS path so it is not included below.
"""

variable_map = [
    ('equilibrium.code.name', None),  # EVAL: literal EFIT_tree name, not an MDS signal
    ('equilibrium.code.version', None),  # EVAL: literal EFIT_tree name, not an MDS signal
    ('equilibrium.ids_properties.homogeneous_time', None),  # VALUE: constant 1

    ('equilibrium.time', r'\TOP:RESULTS:GEQDSK:GTIME'),  # ms -> s, divide by 1000.
    ('equilibrium.time_slice.time', r'\TOP:RESULTS:GEQDSK:GTIME'),  # ms -> s, divide by 1000.
    ('equilibrium.time_slice', r'\TOP:RESULTS:GEQDSK:GTIME'),  # AoS length = size(GTIME)

    ('equilibrium.time_slice.boundary.outline.r', r'\TOP:RESULTS:GEQDSK:RBBBS'),  # NaN-filled where RBBBS == 0 (padding beyond boundary point count)
    ('equilibrium.time_slice.boundary.outline.z', r'\TOP:RESULTS:GEQDSK:ZBBBS'),  # NaN-filled using RBBBS == 0 mask
    ('equilibrium.time_slice.boundary.x_point', None),  # VALUE: constant 2
    ('equilibrium.time_slice.boundary.x_point.0.r', r'\TOP:RESULTS:AEQDSK:RXPT1'),  # NaN-filled where RXPT1 == 0
    ('equilibrium.time_slice.boundary.x_point.1.r', r'\TOP:RESULTS:AEQDSK:RXPT2'),  # NaN-filled where RXPT2 == 0
    ('equilibrium.time_slice.boundary.x_point.0.z', r'\TOP:RESULTS:AEQDSK:ZXPT1'),  # NaN-filled where ZXPT1 == 0
    ('equilibrium.time_slice.boundary.x_point.1.z', r'\TOP:RESULTS:AEQDSK:ZXPT2'),  # NaN-filled where ZXPT2 == 0

    ('equilibrium.time_slice.boundary_separatrix.outline.r', r'\TOP:RESULTS:GEQDSK:RBBBS'),  # NaN-filled where RBBBS == 0
    ('equilibrium.time_slice.boundary_separatrix.outline.z', r'\TOP:RESULTS:GEQDSK:ZBBBS'),  # NaN-filled using RBBBS == 0 mask
    ('equilibrium.time_slice.boundary_separatrix.x_point', None),  # VALUE: constant 2
    ('equilibrium.time_slice.boundary_separatrix.x_point.0.r', r'\TOP:RESULTS:AEQDSK:RXPT1'),  # NaN-filled where RXPT1 == 0
    ('equilibrium.time_slice.boundary_separatrix.x_point.1.r', r'\TOP:RESULTS:AEQDSK:RXPT2'),  # NaN-filled where RXPT2 == 0
    ('equilibrium.time_slice.boundary_separatrix.x_point.0.z', r'\TOP:RESULTS:AEQDSK:ZXPT1'),  # NaN-filled where ZXPT1 == 0
    ('equilibrium.time_slice.boundary_separatrix.x_point.1.z', r'\TOP:RESULTS:AEQDSK:ZXPT2'),  # NaN-filled where ZXPT2 == 0
    ('equilibrium.time_slice.boundary_separatrix.geometric_axis.r', r'\TOP:RESULTS:AEQDSK:RSURF'),
    ('equilibrium.time_slice.boundary_separatrix.geometric_axis.z', r'\TOP:RESULTS:AEQDSK:ZSURF'),
    ('equilibrium.time_slice.boundary_separatrix.closest_wall_point.distance', r'\TOP:RESULTS:AEQDSK:SEPLIM'),
    ('equilibrium.time_slice.boundary_separatrix.gap', None),  # VALUE: constant 4
    ('equilibrium.time_slice.boundary_separatrix.gap.0.name', None),  # VALUE: constant 'inboard'
    ('equilibrium.time_slice.boundary_separatrix.gap.0.value', r'\TOP:RESULTS:AEQDSK:GAPIN'),
    ('equilibrium.time_slice.boundary_separatrix.gap.1.name', None),  # VALUE: constant 'outboard'
    ('equilibrium.time_slice.boundary_separatrix.gap.1.value', r'\TOP:RESULTS:AEQDSK:GAPOUT'),
    ('equilibrium.time_slice.boundary_separatrix.gap.2.name', None),  # VALUE: constant 'top'
    ('equilibrium.time_slice.boundary_separatrix.gap.2.value', r'\TOP:RESULTS:AEQDSK:GAPTOP'),
    ('equilibrium.time_slice.boundary_separatrix.gap.3.name', None),  # VALUE: constant 'bottom'
    ('equilibrium.time_slice.boundary_separatrix.gap.3.value', r'\TOP:RESULTS:AEQDSK:GAPBOT'),
    ('equilibrium.time_slice.boundary_separatrix.strike_point', None),  # VALUE: constant 4
    ('equilibrium.time_slice.boundary_separatrix.strike_point.0.r', r'\TOP:RESULTS:AEQDSK:RVSID'),  # NaN-filled where value == -0.89
    ('equilibrium.time_slice.boundary_separatrix.strike_point.0.z', r'\TOP:RESULTS:AEQDSK:ZVSID'),  # NaN-filled where value == -0.89
    ('equilibrium.time_slice.boundary_separatrix.strike_point.1.r', r'\TOP:RESULTS:AEQDSK:RVSOD'),  # NaN-filled where value == -0.89
    ('equilibrium.time_slice.boundary_separatrix.strike_point.1.z', r'\TOP:RESULTS:AEQDSK:ZVSOD'),  # NaN-filled where value == -0.89
    ('equilibrium.time_slice.boundary_separatrix.strike_point.2.r', r'\TOP:RESULTS:AEQDSK:RVSIU'),  # NaN-filled where value == -0.89
    ('equilibrium.time_slice.boundary_separatrix.strike_point.2.z', r'\TOP:RESULTS:AEQDSK:ZVSIU'),  # NaN-filled where value == -0.89
    ('equilibrium.time_slice.boundary_separatrix.strike_point.3.r', r'\TOP:RESULTS:AEQDSK:RVSOU'),  # NaN-filled where value == -0.89
    ('equilibrium.time_slice.boundary_separatrix.strike_point.3.z', r'\TOP:RESULTS:AEQDSK:ZVSOU'),  # NaN-filled where value == -0.89
    ('equilibrium.time_slice.boundary_separatrix.triangularity_upper', r'\TOP:RESULTS:AEQDSK:TRITOP'),
    ('equilibrium.time_slice.boundary_separatrix.triangularity_lower', r'\TOP:RESULTS:AEQDSK:TRIBOT'),
    ('equilibrium.time_slice.boundary_separatrix.elongation', r'\TOP:RESULTS:AEQDSK:KAPPA'),
    ('equilibrium.time_slice.boundary_separatrix.minor_radius', r'\TOP:RESULTS:AEQDSK:AMINOR'),
    ('equilibrium.time_slice.boundary_separatrix.psi', r'\TOP:RESULTS:GEQDSK:SSIBRY'),

    # constraints.* below go through _common.py's scalar_constraint_data / vector_constraint_data /
    # concat_constraint_data, which read from \TOP:MEASUREMENTS:... and re-index onto the
    # GEQDSK.GTIME timebase (nearest MEASUREMENTS.MTIME sample); cocosio is forced to 7 for these.
    ('equilibrium.time_slice.constraints.ip.measured', r'\TOP:MEASUREMENTS:PLASMA'),
    ('equilibrium.time_slice.constraints.ip.measured_error_upper', r'\TOP:MEASUREMENTS:SIGPASMA'),
    ('equilibrium.time_slice.constraints.ip.weight', r'\TOP:MEASUREMENTS:FWTPASMA'),
    ('equilibrium.time_slice.constraints.ip.reconstructed', r'\TOP:MEASUREMENTS:CPASMA'),
    ('equilibrium.time_slice.constraints.ip.chi_squared', r'\TOP:MEASUREMENTS:CHIPASMA'),

    ('equilibrium.time_slice.constraints.bpol_probe', r'\TOP:MEASUREMENTS:EXPMPI'),  # AoS length = size(EXPMPI, 0)
    ('equilibrium.time_slice.constraints.bpol_probe.measured', r'\TOP:MEASUREMENTS:EXPMPI'),
    ('equilibrium.time_slice.constraints.bpol_probe.measured_error_upper', r'\TOP:MEASUREMENTS:SIGMPI'),
    ('equilibrium.time_slice.constraints.bpol_probe.weight', r'\TOP:MEASUREMENTS:FWTMP2'),
    ('equilibrium.time_slice.constraints.bpol_probe.reconstructed', r'\TOP:MEASUREMENTS:CMPR2'),
    ('equilibrium.time_slice.constraints.bpol_probe.chi_squared', r'\TOP:MEASUREMENTS:SAIMPI'),

    ('equilibrium.time_slice.constraints.diamagnetic_flux.measured', r'\TOP:MEASUREMENTS:DIAMAG'),
    ('equilibrium.time_slice.constraints.diamagnetic_flux.measured_error_upper', r'\TOP:MEASUREMENTS:SIGDIA'),
    ('equilibrium.time_slice.constraints.diamagnetic_flux.weight', r'\TOP:MEASUREMENTS:FWTDIA'),
    ('equilibrium.time_slice.constraints.diamagnetic_flux.reconstructed', r'\TOP:MEASUREMENTS:CDFLUX'),
    ('equilibrium.time_slice.constraints.diamagnetic_flux.chi_squared', r'\TOP:MEASUREMENTS:CHIDFLUX'),

    ('equilibrium.time_slice.constraints.flux_loop', r'\TOP:MEASUREMENTS:SILOPT'),  # AoS length = size(SILOPT, 0)
    ('equilibrium.time_slice.constraints.flux_loop.measured', r'\TOP:MEASUREMENTS:SILOPT'),
    ('equilibrium.time_slice.constraints.flux_loop.measured_error_upper', r'\TOP:MEASUREMENTS:SIGSIL'),
    ('equilibrium.time_slice.constraints.flux_loop.weight', r'\TOP:MEASUREMENTS:FWTSI'),
    ('equilibrium.time_slice.constraints.flux_loop.reconstructed', r'\TOP:MEASUREMENTS:CSILOP'),
    ('equilibrium.time_slice.constraints.flux_loop.chi_squared', r'\TOP:MEASUREMENTS:SAISIL'),

    ('equilibrium.time_slice.constraints.mse_polarisation_angle', r'\TOP:MEASUREMENTS:TANGAM'),  # AoS length = size(TANGAM, 0)
    ('equilibrium.time_slice.constraints.mse_polarisation_angle.measured', r'\TOP:MEASUREMENTS:TANGAM'),  # ATAN(...) applied
    ('equilibrium.time_slice.constraints.mse_polarisation_angle.measured_error_upper', r'\TOP:MEASUREMENTS:SIGGAM'),  # ATAN(...) applied
    ('equilibrium.time_slice.constraints.mse_polarisation_angle.weight', r'\TOP:MEASUREMENTS:FWTGAM'),
    ('equilibrium.time_slice.constraints.mse_polarisation_angle.reconstructed', r'\TOP:MEASUREMENTS:CMGAM'),
    ('equilibrium.time_slice.constraints.mse_polarisation_angle.chi_squared', r'\TOP:MEASUREMENTS:CHIGAM'),

    # pf_current concatenates the EC-coil and F-coil signals along axis=1 (concat_constraint_data)
    ('equilibrium.time_slice.constraints.pf_current', r'\TOP:MEASUREMENTS:ECCURT'),  # AoS length = size(ECCURT,0) + size(FCCURT,0); concatenated with \TOP:MEASUREMENTS:FCCURT
    ('equilibrium.time_slice.constraints.pf_current.measured', r'\TOP:MEASUREMENTS:ECCURT'),  # concatenated with \TOP:MEASUREMENTS:FCCURT
    ('equilibrium.time_slice.constraints.pf_current.measured_error_upper', r'\TOP:MEASUREMENTS:SIGECC'),  # concatenated with \TOP:MEASUREMENTS:SIGFCC
    ('equilibrium.time_slice.constraints.pf_current.weight', r'\TOP:MEASUREMENTS:FWTEC'),  # concatenated with \TOP:MEASUREMENTS:FWTFC
    ('equilibrium.time_slice.constraints.pf_current.reconstructed', r'\TOP:MEASUREMENTS:CECURR'),  # concatenated with \TOP:MEASUREMENTS:CCBRSP
    ('equilibrium.time_slice.constraints.pf_current.chi_squared', r'\TOP:MEASUREMENTS:CHIECC'),  # concatenated with \TOP:MEASUREMENTS:CHIFCC

    ('equilibrium.time_slice.constraints.pressure', r'\TOP:MEASUREMENTS:PRESSR'),  # AoS length = size(PRESSR, 0)
    ('equilibrium.time_slice.constraints.pressure.measured', r'\TOP:MEASUREMENTS:PRESSR'),
    ('equilibrium.time_slice.constraints.pressure.measured_error_upper', r'\TOP:MEASUREMENTS:SIGPRE'),
    ('equilibrium.time_slice.constraints.pressure.weight', r'\TOP:MEASUREMENTS:FWTPRE'),
    ('equilibrium.time_slice.constraints.pressure.reconstructed', r'\TOP:MEASUREMENTS:CPRESS'),
    ('equilibrium.time_slice.constraints.pressure.chi_squared', r'\TOP:MEASUREMENTS:SAIPRE'),
    ('equilibrium.time_slice.constraints.pressure.position.psi', r'\TOP:MEASUREMENTS:RPRESS'),  # normalized psi -> real psi: |RPRESS| * (SSIBRY - SSIMAG) + SSIMAG

    ('equilibrium.time_slice.constraints.j_tor', r'\TOP:MEASUREMENTS:VZEROJ'),  # AoS length = size(VZEROJ, 0)
    ('equilibrium.time_slice.constraints.j_tor.measured', r'\TOP:MEASUREMENTS:VZEROJ'),  # sign restored from sign(CPASMA); VZEROJ is stored as an unsigned magnitude
    ('equilibrium.time_slice.constraints.j_tor.position.psi', r'\TOP:MEASUREMENTS:SIZEROJ'),  # normalized psi -> real psi: |SIZEROJ| * (SSIBRY - SSIMAG) + SSIMAG

    ('equilibrium.time_slice.global_quantities.ip', r'\TOP:RESULTS:GEQDSK:CPASMA'),
    ('equilibrium.time_slice.global_quantities.li_3', r'\TOP:RESULTS:AEQDSK:LI3'),
    ('equilibrium.time_slice.global_quantities.magnetic_axis.r', r'\TOP:RESULTS:GEQDSK:RMAXIS'),
    ('equilibrium.time_slice.global_quantities.magnetic_axis.z', r'\TOP:RESULTS:GEQDSK:ZMAXIS'),
    ('equilibrium.time_slice.global_quantities.magnetic_axis.b_field_tor', r'\TOP:RESULTS:AEQDSK:BT0'),
    ('equilibrium.time_slice.global_quantities.psi_axis', r'\TOP:RESULTS:GEQDSK:SSIMAG'),
    ('equilibrium.time_slice.global_quantities.psi_boundary', r'\TOP:RESULTS:GEQDSK:SSIBRY'),
    ('equilibrium.time_slice.global_quantities.area', r'\TOP:RESULTS:AEQDSK:AREA'),
    ('equilibrium.time_slice.global_quantities.surface', r'\TOP:RESULTS:AEQDSK:PSURFA'),
    ('equilibrium.time_slice.global_quantities.volume', r'\TOP:RESULTS:AEQDSK:VOLUME'),
    ('equilibrium.time_slice.global_quantities.beta_pol', r'\TOP:RESULTS:AEQDSK:BETAP'),
    ('equilibrium.time_slice.global_quantities.beta_tor', r'\TOP:RESULTS:AEQDSK:BETAT'),
    ('equilibrium.time_slice.global_quantities.beta_normal', r'\TOP:RESULTS:AEQDSK:BETAN'),
    ('equilibrium.time_slice.global_quantities.q_95', r'\TOP:RESULTS:AEQDSK:Q95'),
    ('equilibrium.time_slice.global_quantities.q_axis', r'\TOP:RESULTS:AEQDSK:Q0'),
    ('equilibrium.time_slice.global_quantities.q_min.value', r'\TOP:RESULTS:AEQDSK:QMIN'),

    ('equilibrium.time_slice.profiles_1d.dpressure_dpsi', r'\TOP:RESULTS:GEQDSK:PPRIME'),
    ('equilibrium.time_slice.profiles_1d.f', r'\TOP:RESULTS:GEQDSK:FPOL'),
    ('equilibrium.time_slice.profiles_1d.f_df_dpsi', r'\TOP:RESULTS:GEQDSK:FFPRIM'),
    ('equilibrium.time_slice.profiles_1d.pressure', r'\TOP:RESULTS:GEQDSK:PRES'),
    ('equilibrium.time_slice.profiles_1d.psi', r'\TOP:RESULTS:GEQDSK:PSIN'),  # geqdsk_psi(): denormalized as SSIMAG + PSIN * (SSIBRY - SSIMAG)
    ('equilibrium.time_slice.profiles_1d.q', r'\TOP:RESULTS:GEQDSK:QPSI'),
    ('equilibrium.time_slice.profiles_1d.rho_tor_norm', r'\TOP:RESULTS:GEQDSK:RHOVN'),
    ('equilibrium.time_slice.profiles_1d.j_tor', r'\TOP:RESULTS:FLUXFUN:JEFF'),  # interpolate_psi_1d(): interpolated off the FLUXFUN.PSI grid onto GEQDSK.PSIN (denormalized via SSIMAG/SSIBRY)
    ('equilibrium.time_slice.profiles_1d.j_parallel', r'\TOP:RESULTS:FLUXFUN:JLL'),  # interpolate_psi_1d(): interpolated off the FLUXFUN.PSI grid onto GEQDSK.PSIN (denormalized via SSIMAG/SSIBRY)
    ('equilibrium.time_slice.profiles_1d.volume', r'\TOP:RESULTS:FLUXFUN:VOL'),  # interpolate_psi_1d(): interpolated off the FLUXFUN.PSI grid onto GEQDSK.PSIN (denormalized via SSIMAG/SSIBRY)

    ('equilibrium.time_slice.profiles_2d', None),  # VALUE: constant 1 (single 2D grid per time slice)
    ('equilibrium.time_slice.profiles_2d.grid.dim1', r'\TOP:RESULTS:GEQDSK:R'),  # tile(R, n_times), then transposed to [time, dim1]
    ('equilibrium.time_slice.profiles_2d.grid.dim2', r'\TOP:RESULTS:GEQDSK:Z'),  # tile(Z, n_times), then transposed to [time, dim2]
    ('equilibrium.time_slice.profiles_2d.grid_type.index', r'\TOP:RESULTS:GEQDSK:BCENTR'),  # constant grid_type index 1, tiled to n_times = size(BCENTR); BCENTR used only for its time count
    ('equilibrium.time_slice.profiles_2d.psi', r'\TOP:RESULTS:GEQDSK:PSIRZ'),  # axes transposed [1,0,3,2] to match IMAS [time, dim1, dim2] layout

    ('equilibrium.vacuum_toroidal_field.b0', r'\TOP:RESULTS:GEQDSK:BCENTR'),
    ('equilibrium.vacuum_toroidal_field.r0', r'\TOP:RESULTS:GEQDSK:RZERO'),  # first element only, RZERO[0]

    ('equilibrium.time_slice.convergence.iterations_n', r'\TOP:MEASUREMENTS:CERROR'),  # iteration count = number of nonzero entries along CERROR's iteration dimension
    ('equilibrium.time_slice.convergence.grad_shafranov_deviation_value', r'\TOP:RESULTS:AEQDSK:ERROR'),
    ('equilibrium.time_slice.convergence.grad_shafranov_deviation_expression.index', None),  # EVAL: constant 3
]
