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
                    ods['equilibrium']['time_slice'][i][boundary]['x_point'][n]['z'] = z[i] / 1.e2
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
def pf_current_measurements(ods, machine, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load PF current measurements that require stacking multiple signals
    
    Replaces py2tdi(stack_outer_2, ...) expressions for TokSearch compatibility
    """
    printd(f'Loading PF current measurements from {EFIT_tree}...', topic='machine')
    
    provider = ods.get_mds_provider(machine) 
    pulse_id = get_pulse_id(pulse, EFIT_run_id)
    
    # Get E-coil and F-coil current measurements
    TDIs = {
        'eccurt': f'\\{EFIT_tree}::TOP.MEASUREMENTS.ECCURT',
        'fccurt': f'\\{EFIT_tree}::TOP.MEASUREMENTS.FCCURT',
        'sigecc': f'\\{EFIT_tree}::TOP.MEASUREMENTS.SIGECC', 
        'sigfcc': f'\\{EFIT_tree}::TOP.MEASUREMENTS.SIGFCC',
        'fwtec': f'\\{EFIT_tree}::TOP.MEASUREMENTS.FWTEC',
        'fwtfc': f'\\{EFIT_tree}::TOP.MEASUREMENTS.FWTFC',
        'cecurr': f'\\{EFIT_tree}::TOP.MEASUREMENTS.CECURR',
        'ccbrsp': f'\\{EFIT_tree}::TOP.MEASUREMENTS.CCBRSP',
        'chiecc': f'\\{EFIT_tree}::TOP.MEASUREMENTS.CHIECC',
        'chifcc': f'\\{EFIT_tree}::TOP.MEASUREMENTS.CHIFCC'
    }
    
    efit_data = provider.raw(EFIT_tree, pulse_id, TDIs)
    
    # Stack E-coil and F-coil data along outer dimension (axis=1 as per stack_outer_2 function)
    measured = np.concatenate([efit_data['eccurt'], efit_data['fccurt']], axis=1)
    measured_error = np.concatenate([efit_data['sigecc'], efit_data['sigfcc']], axis=1)
    weight = np.concatenate([efit_data['fwtec'], efit_data['fwtfc']], axis=1)
    reconstructed = np.concatenate([efit_data['cecurr'], efit_data['ccbrsp']], axis=1)
    chi_squared = np.concatenate([efit_data['chiecc'], efit_data['chifcc']], axis=1)
    
    # Set PF current constraint data
    n_times = measured.shape[1] if len(measured.shape) > 1 else 1
    n_pf = measured.shape[0]
    with omas_environment(ods, cocosio=MDS_gEQDSK_COCOS_identify(ods, machine, pulse, EFIT_tree, EFIT_run_id)):
        for i in range(n_times):
            time_slice = ods['equilibrium']['time_slice'][i]
            
            # Set data for all PF coils
            for j in range(n_pf):
                if len(measured.shape) > 1:
                    time_slice['constraints']['pf_current'][j]['measured'] = measured[j, i]
                    time_slice['constraints']['pf_current'][j]['measured_error_upper'] = measured_error[j, i]
                    time_slice['constraints']['pf_current'][j]['weight'] = weight[j, i]
                    time_slice['constraints']['pf_current'][j]['reconstructed'] = reconstructed[j, i]
                    time_slice['constraints']['pf_current'][j]['chi_squared'] = chi_squared[j, i]
                else:
                    time_slice['constraints']['pf_current'][j]['measured'] = measured[j]
                    time_slice['constraints']['pf_current'][j]['measured_error_upper'] = measured_error[j]
                    time_slice['constraints']['pf_current'][j]['weight'] = weight[j]
                    time_slice['constraints']['pf_current'][j]['reconstructed'] = reconstructed[j]
                    time_slice['constraints']['pf_current'][j]['chi_squared'] = chi_squared[j]
    return ods


# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def psi_profiles(ods, machine, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load PSI profiles that require complex transformations
    
    Replaces py2tdi(efit_psi_to_real_psi_2d, ...) and py2tdi(geqdsk_psi, ...) expressions
    """
    printd(f'Loading PSI profiles from {EFIT_tree}...', topic='machine')
    
    provider = ods.get_mds_provider(machine)
    pulse_id = get_pulse_id(pulse, EFIT_run_id)
    
    # Get data for complex PSI transformations (simple PSI data handled by TDI expressions)
    TDIs = {
        'ssimag': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.SSIMAG',
        'ssibry': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.SSIBRY',
        'psin': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.PSIN',
        'rpress': f'\\{EFIT_tree}::TOP.MEASUREMENTS.RPRESS',
        'sizeroj': f'\\{EFIT_tree}::TOP.MEASUREMENTS.SIZEROJ'
    }
    
    efit_data = provider.raw(EFIT_tree, pulse_id, TDIs)
    
    # Convert EFIT PSI to real PSI (following geqdsk_psi algorithm from python_tdi.py)
    ssimag = efit_data['ssimag']  # a
    ssibry = efit_data['ssibry']  # b  
    psin = efit_data['psin']      # c
    
    # geqdsk_psi algorithm: a[:, None] + np.linspace(0, 1, n).T * (b[:, None] - a[:, None])
    n = len(psin)
    geqdsk_psi = ssimag[:, None] + np.linspace(0, 1, n).T * (ssibry[:, None] - ssimag[:, None])
    
    # Set PSI profiles
    n_times = len(ssimag) if hasattr(ssimag, '__len__') else 1
    with omas_environment(ods, cocosio=MDS_gEQDSK_COCOS_identify(ods, machine, pulse, EFIT_tree, EFIT_run_id)):
        for i in range(n_times):
            time_slice = ods['equilibrium']['time_slice'][i]
            
            # Set profiles_1d PSI (only this complex calculation needed)
            if i < len(geqdsk_psi):
                time_slice['profiles_1d']['psi'] = geqdsk_psi[i] if len(geqdsk_psi.shape) > 1 else geqdsk_psi
            
            # Set pressure constraint positions (efit_psi_to_real_psi_2d replacement)
            # efit_psi_to_real_psi_2d algorithm: (a.T * (c - b) + b).T
            if type(efit_data.get("rpress", Exception())) != Exception and len(efit_data['rpress']) > 0:
                rpress = -efit_data['rpress']  # a (note: negative sign from TDI)
                
                # Handle flexible dimensions: rpress needs time and space dimensions
                if len(rpress.shape) == 1:
                    # Only time dimension - add spatial dimension as singleton
                    rpress = rpress[:, np.newaxis]
                # Apply transformation for this time slice
                if i < rpress.shape[0]:
                    for j, press_pos in enumerate(rpress[i]):
                        pressure_psi = (press_pos * (ssibry[i] - ssimag[i]) + ssimag[i]).T
                        time_slice['constraints.pressure'][j]['position.psi'] = pressure_psi
            
            # Set j_tor constraint positions (efit_psi_to_real_psi_2d replacement)  
            if type(efit_data.get("sizeroj", Exception())) != Exception and len(efit_data['sizeroj']) > 0:
                sizeroj = efit_data['sizeroj']  # a
                if len(sizeroj.shape) == 1:
                    # Only time dimension - add spatial dimension as singleton
                    sizeroj = sizeroj[:, np.newaxis]
                
                # Apply transformation for this time slice
                if i < sizeroj.shape[0]:
                    for j, jtor_pos in enumerate(sizeroj[i]):
                        jtor_psi = (jtor_pos * (ssibry[i] - ssimag[i]) + ssimag[i]).T
                        time_slice[f'constraints.j_tor'][j]['position.psi'] = jtor_psi
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

# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def convergence_data(ods, machine, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load convergence data that requires complex axis operations
    
    Replaces py2tdi(get_largest_axis_value, ...) expressions
    """
    printd(f'Loading convergence data from {EFIT_tree}...', topic='machine')
    
    provider = ods.get_mds_provider(machine)
    pulse_id = get_pulse_id(pulse, EFIT_run_id)
    
    # Get convergence error data
    TDIs = {
        'cerror': f'\\{EFIT_tree}::TOP.MEASUREMENTS.CERROR',
        'cerror_dim': f'dim_of(\\{EFIT_tree}::TOP.MEASUREMENTS.CERROR,0)'
    }
    
    efit_data = provider.raw(EFIT_tree, pulse_id, TDIs)
    
    # get_largest_axis_value algorithm from python_tdi.py
    cerror_dim = efit_data['cerror_dim']  # a
    cerror = efit_data['cerror']          # b
    
    # get_largest_axis_value algorithm:
    # a = np.array([a for k in range(b.shape[0])])
    # a[b == 0] = 0  
    # return np.nanmax(a, axis=1)
    a = np.array([cerror_dim for k in range(cerror.shape[0])])
    a[cerror == 0] = 0
    max_axis_values = np.nanmax(a, axis=1)
    
    # Set convergence data
    n_times = len(max_axis_values) if hasattr(max_axis_values, '__len__') else 1
    
    for i in range(n_times):
        time_slice = ods['equilibrium']['time_slice'][i]
        iterations_n = max_axis_values[i] if hasattr(max_axis_values, '__len__') else max_axis_values
        time_slice['convergence']['iterations_n'] = iterations_n
    return ods

# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def pressure_measurements(ods, machine, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load pressure measurements that require ensure_2d transformation
    
    Replaces py2tdi(ensure_2d, -...) expressions for TokSearch compatibility
    """
    printd(f'Loading pressure measurements from {EFIT_tree}...', topic='machine')
    
    provider = ods.get_mds_provider(machine)
    pulse_id = get_pulse_id(pulse, EFIT_run_id)
    
    # Get pressure measurement data  
    TDIs = {
        'pressr': f'\\{EFIT_tree}::TOP.MEASUREMENTS.PRESSR',
        'sigpre': f'\\{EFIT_tree}::TOP.MEASUREMENTS.SIGPRE',
        'fwtpre': f'\\{EFIT_tree}::TOP.MEASUREMENTS.FWTPRE', 
        'cpress': f'\\{EFIT_tree}::TOP.MEASUREMENTS.CPRESS',
        'saipre': f'\\{EFIT_tree}::TOP.MEASUREMENTS.SAIPRE'
    }
    
    efit_data = provider.raw(EFIT_tree, pulse_id, TDIs)
    
    # Apply ensure_2d transformation: ensure array has at least 2 dimensions
    # Note: negative sign applied as per TDI expressions
    measured = -efit_data['pressr']
    measured_error = -efit_data['sigpre']
    weight = -efit_data['fwtpre']
    reconstructed = -efit_data['cpress']
    chi_squared = -efit_data['saipre']
    
    # Ensure 2D: if 1D, add singleton dimension
    for data in [measured, measured_error, weight, reconstructed, chi_squared]:
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
    
    # Set pressure constraint data
    n_times = measured.shape[0] if len(measured.shape) > 1 else 1
    n_pressure = measured.shape[1] if len(measured.shape) > 1 else len(measured)
    
    with omas_environment(ods, cocosio=MDS_gEQDSK_COCOS_identify(ods, machine, pulse, EFIT_tree, EFIT_run_id)):
        for i in range(n_times):
            time_slice = ods['equilibrium']['time_slice'][i]
            
            # Set data for all pressure measurements
            for j in range(n_pressure):
                if len(measured.shape) > 1:
                    time_slice['constraints']['pressure'][j]['measured'] = measured[i, j]
                    time_slice['constraints']['pressure'][j]['measured_error_upper'] = measured_error[i, j] 
                    time_slice['constraints']['pressure'][j]['weight'] = weight[i, j]
                    time_slice['constraints']['pressure'][j]['reconstructed'] = reconstructed[i, j]
                    time_slice['constraints']['pressure'][j]['chi_squared'] = chi_squared[i, j]
                else:
                    time_slice['constraints']['pressure'][j]['measured'] = measured[j]
                    time_slice['constraints']['pressure'][j]['measured_error_upper'] = measured_error[j]
                    time_slice['constraints']['pressure'][j]['weight'] = weight[j]
                    time_slice['constraints']['pressure'][j]['reconstructed'] = reconstructed[j]
                    time_slice['constraints']['pressure'][j]['chi_squared'] = chi_squared[j]
    return ods

# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def jtor_measurements(ods, machine, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load j_tor measurements that require convert_from_mega_2d transformation
    
    Replaces py2tdi(convert_from_mega_2d, -...) expressions for TokSearch compatibility
    """
    printd(f'Loading j_tor measurements from {EFIT_tree}...', topic='machine')
    
    provider = ods.get_mds_provider(machine)
    pulse_id = get_pulse_id(pulse, EFIT_run_id)
    
    # Get j_tor measurement data
    TDIs = {
        'vzeroj': f'\\{EFIT_tree}::TOP.MEASUREMENTS.VZEROJ'
    }
    
    efit_data = provider.raw(EFIT_tree, pulse_id, TDIs)
    
    # Apply convert_from_mega_2d transformation: divide by 1e6 and ensure 2D
    # Note: negative sign applied as per TDI expression
    measured = -efit_data['vzeroj'] / 1e6
    
    # Ensure 2D: if 1D, add singleton dimension  
    if len(measured.shape) == 1:
        measured = measured[:, np.newaxis]
    
    # Set j_tor constraint data
    n_times = measured.shape[0] if len(measured.shape) > 1 else 1
    n_jtor = measured.shape[1] if len(measured.shape) > 1 else len(measured)
    
    with omas_environment(ods, cocosio=MDS_gEQDSK_COCOS_identify(ods, machine, pulse, EFIT_tree, EFIT_run_id)):
        for i in range(n_times):
            time_slice = ods['equilibrium']['time_slice'][i]
            
            # Set data for all j_tor measurements
            for j in range(n_jtor):
                if len(measured.shape) > 1:
                    time_slice['constraints']['j_tor'][j]['measured'] = measured[i, j]
                else:
                    time_slice['constraints']['j_tor'][j]['measured'] = measured[j]
    return ods

# ================================
# Add test function call
if __name__ == '__main__':
    # Test with d3d pulse
    test_ods = ODS()
    equilibrium_time_slice_data(test_ods, 'd3d', 194844)
    print(f"Loaded equilibrium data with {len(test_ods['equilibrium']['time'])} time points")