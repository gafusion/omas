import numpy as np
from scipy.interpolate import interp1d
from omas import *
from omas.omas_utils import printd, printe
from omas.machine_mappings._common import *
from omas.utilities.machine_mapping_decorator import machine_mapping_function
from omas.omas_core import ODS
from omas.omas_physics import omas_environment

__all__ = []
__regression_arguments__ = {'__all__': __all__, "requires_omfit": []}

# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def equilibrium_time_slice_data(ods, machine, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load EFIT boundary outline, X-point and strike point data

    This function replaces the py2tdi(nan_where, ...) expressions in _efit.json that
    TokSearch cannot handle, providing backend-agnostic access to EFIT data.

    :param ods: OMAS ODS instance
    :param machine: machine name
    :param pulse: shot number
    :param EFIT_tree: EFIT tree name (e.g., 'EFIT01', 'EFIT02')
    :param EFIT_run_id: run id extension for pulse number
    """
    printd(f'Loading EFIT boundary data from {EFIT_tree}...', topic='machine')

    # Get provider from ODS
    provider = ods.get_mds_provider(machine)
    pulse_id = get_pulse_id(pulse, EFIT_run_id)

    # The strike points are fetched undivided because the -0.89 sentinel is a property
    # of the stored value, not of the value scaled by 1/100.
    TDIs = {
        # Boundary data requiring NaN filtering
        'rbbbs': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.RBBBS',
        'zbbbs': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.ZBBBS',
        # X-point data requiring NaN filtering
        'rxpt1': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.RXPT1',
        'zxpt1': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.ZXPT1',
        'rxpt2': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.RXPT2',
        'zxpt2': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.ZXPT2',
        # Strike point data requiring NaN filtering
        'rvsid': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.RVSID',
        'zvsid': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.ZVSID',
        'rvsod': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.RVSOD',
        'zvsod': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.ZVSOD',
        'rvsiu': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.RVSIU',
        'zvsiu': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.ZVSIU',
        'rvsou': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.RVSOU',
        'zvsou': f'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.ZVSOU'
    }

    # Single provider call for all data
    efit_data = provider.raw(EFIT_tree, pulse_id, TDIs)
    with omas_environment(ods, cocosio=MDS_gEQDSK_COCOS_identify(ods, machine, pulse, EFIT_tree, EFIT_run_id)):
        # Boundary outline: RBBBS == 0 marks the unused tail of the fixed size arrays,
        # and masks Z as well so that both keep the same length once the NaNs are dropped
        rbbbs = np.atleast_2d(efit_data['rbbbs'])
        zbbbs = np.atleast_2d(efit_data['zbbbs'])
        unset = rbbbs == 0
        rbbbs[unset] = np.nan
        zbbbs[unset] = np.nan

        n_times = rbbbs.shape[0]
        for i in range(n_times):
            valid = ~np.isnan(rbbbs[i])
            for boundary in ['boundary', 'boundary_separatrix']:
                ods['equilibrium']['time_slice'][i][boundary]['outline']['r'] = rbbbs[i][valid]
                ods['equilibrium']['time_slice'][i][boundary]['outline']['z'] = zbbbs[i][valid]

        # X-points: a 0 means no X-point was found at that time
        for i in range(len(efit_data['rxpt1'])):
            n = 0
            for k, (r_node, z_node) in enumerate([('rxpt1', 'zxpt1'), ('rxpt2', 'zxpt2')]):
                r = np.atleast_1d(efit_data[r_node])
                z = np.atleast_1d(efit_data[z_node])
                # Do not append points that do not actually hold an X-point
                if r[i] <= 0.0:
                    continue
                for boundary in ['boundary', 'boundary_separatrix']:
                    ods['equilibrium']['time_slice'][i][boundary]['x_point'][n]['r'] = r[i]
                    ods['equilibrium']['time_slice'][i][boundary]['x_point'][n]['z'] = z[i]
                n += 1
        # Only add strike points with R > 0.0
        for i in range(len(efit_data['rvsid'])):
            n = 0
            for k, (r_node, z_node) in enumerate([('rvsid', 'zvsid'), ('rvsod', 'zvsod'), ('rvsiu', 'zvsiu'), ('rvsou', 'zvsou')]):
                r = np.atleast_1d(efit_data[r_node])
                z = np.atleast_1d(efit_data[z_node])
                # Do not append points that do not actually hold a strike point
                if r[i] <= 0.0:
                    continue
                ods['equilibrium']['time_slice'][i]['boundary_separatrix']['strike_point'][n]['r'] = r[i]
                ods['equilibrium']['time_slice'][i]['boundary_separatrix']['strike_point'][n]['z'] = z[i] / 1.e2
                n += 1
        printd(f'Successfully loaded EFIT data for {n_times} time slices', topic='machine')
    return ods

# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def psi_profiles(ods, machine, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load the 1D psi grid

    Replaces the py2tdi(geqdsk_psi, ...) expression for TokSearch compatibility.
    The constraint positions are handled by constraint_psi_to_real_psi in _common.py
    """
    printd(f'Loading PSI profiles from {EFIT_tree}...', topic='machine')

    provider = ods.get_mds_provider(machine)
    pulse_id = get_pulse_id(pulse, EFIT_run_id)

    TDIs = {
        'ssimag': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.SSIMAG',
        'ssibry': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.SSIBRY',
        'psin': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.PSIN'
    }

    efit_data = provider.raw(EFIT_tree, pulse_id, TDIs)

    # geqdsk_psi algorithm from python_tdi.py: a[:, None] + np.linspace(0, 1, n) * (b[:, None] - a[:, None])
    ssimag = efit_data['ssimag']
    ssibry = efit_data['ssibry']
    n = len(efit_data['psin'])
    geqdsk_psi = ssimag[:, None] + np.linspace(0, 1, n) * (ssibry[:, None] - ssimag[:, None])

    with omas_environment(ods, cocosio=MDS_gEQDSK_COCOS_identify(ods, machine, pulse, EFIT_tree, EFIT_run_id)):
        for i in range(len(geqdsk_psi)):
            ods['equilibrium']['time_slice'][i]['profiles_1d']['psi'] = geqdsk_psi[i]
    return ods

# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def fluxfun_profiles(ods, machine, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load the FLUXFUN profiles interpolated onto the GEQDSK psi grid

    Replaces py2tdi(interpolate_psi_1d, ...) expressions for TokSearch compatibility
    """
    printd(f'Loading FLUXFUN profiles from {EFIT_tree}...', topic='machine')

    provider = ods.get_mds_provider(machine)
    pulse_id = get_pulse_id(pulse, EFIT_run_id)

    TDIs = {
        'ssimag': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.SSIMAG',
        'ssibry': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.SSIBRY',
        'psin': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.PSIN',
        'psi': f'\\{EFIT_tree}::TOP.RESULTS.FLUXFUN.PSI',
        'j_tor': f'\\{EFIT_tree}::TOP.RESULTS.FLUXFUN.JEFF',
        'j_parallel': f'\\{EFIT_tree}::TOP.RESULTS.FLUXFUN.JLL',
        'volume': f'\\{EFIT_tree}::TOP.RESULTS.FLUXFUN.VOL'
    }

    efit_data = provider.raw(EFIT_tree, pulse_id, TDIs)

    # interpolate_psi_1d algorithm from python_tdi.py: the FLUXFUN quantities live on
    # their own psi grid and are interpolated onto the geqdsk_psi grid, holding the
    # end values constant outside the range
    ssimag = efit_data['ssimag']
    ssibry = efit_data['ssibry']
    n = len(efit_data['psin'])
    geqdsk_psi = ssimag[:, None] + np.linspace(0, 1, n) * (ssibry[:, None] - ssimag[:, None])
    x1 = efit_data['psi'].T

    with omas_environment(ods, cocosio=MDS_gEQDSK_COCOS_identify(ods, machine, pulse, EFIT_tree, EFIT_run_id)):
        for entry in ['j_tor', 'j_parallel', 'volume']:
            y1 = efit_data[entry].T
            for i in range(x1.shape[0]):
                fill_value = (y1[i][0], y1[i][-1])
                interpolator = interp1d(x1[i], y1[i], kind='cubic', bounds_error=False, fill_value=fill_value)
                ods['equilibrium']['time_slice'][i]['profiles_1d'][entry] = interpolator(geqdsk_psi[i])
    return ods

# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def grid_2d_data(ods, machine, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load 2D grid data that requires tiling operations
    
    Replaces py2tdi(tile, ...) expressions for grid dimensions
    """
    printd(f'Loading 2D grid data from {EFIT_tree}...', topic='machine')
    
    provider = ods.get_mds_provider(machine)
    pulse_id = get_pulse_id(pulse, EFIT_run_id)
    
    # Get grid data
    TDIs = {
        'r_grid': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.R',
        'z_grid': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.Z',
        'bcentr': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.BCENTR'
    }
    
    efit_data = provider.raw(EFIT_tree, pulse_id, TDIs)
    
    # Tile R and Z grids across time dimension (following tile algorithm from python_tdi.py)
    r_grid = efit_data['r_grid']
    z_grid = efit_data['z_grid']
    n_times = len(efit_data['bcentr'])
    
    # tile algorithm: np.array([a for k in range(n)])
    r_tiled = np.array([r_grid for k in range(n_times)])
    z_tiled = np.array([z_grid for k in range(n_times)])
    
    with omas_environment(ods, cocosio=MDS_gEQDSK_COCOS_identify(ods, machine, pulse, EFIT_tree, EFIT_run_id)):
        # Set 2D grid data
        for itime in range(n_times):
            ods['equilibrium']['time_slice'][itime]['profiles_2d'][0]['grid']['dim1'] = r_tiled[itime]
            ods['equilibrium']['time_slice'][itime]['profiles_2d'][0]['grid']['dim2'] = z_tiled[itime]
            
            ods['equilibrium']['time_slice'][itime]['profiles_2d'][0]['grid_type']['index'] = 1
    return ods
