import numpy as np
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
def equilibrium_time_slice_data(ods, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load EFIT equilibrium time slice data that requires complex TDI expressions
    
    This function replaces the complex py2tdi expressions in _efit.json that
    TokSearch cannot handle, providing backend-agnostic access to EFIT data.
    
    :param ods: OMAS ODS instance
    :param pulse: shot number
    :param EFIT_tree: EFIT tree name (e.g., 'EFIT01', 'EFIT02')  
    :param EFIT_run_id: run id extension for pulse number
    """
    printd(f'Loading EFIT equilibrium data from {EFIT_tree}...', topic='machine')
    
    # Get provider from ODS
    provider = ods.get_mds_provider('d3d')
    pulse_id = get_pulse_id(pulse, EFIT_run_id)
    
    # Fetch only data requiring NaN filtering (not available via simple TDI)
    printd('Loading EFIT data requiring NaN filtering...', topic='machine')
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
    
    # Get time data just for array sizing (time data is handled by simple TDI expressions)
    # We need this just to know how many time slices to process
    n_times = len(efit_data['rbbbs']) if len(efit_data['rbbbs'].shape) > 1 else 1
    
    # Process boundary data with NaN filtering
    rbbbs = efit_data['rbbbs'] 
    zbbbs = efit_data['zbbbs']
    
    # Set NaN where R boundary is 0 (invalid data)
    rbbbs[rbbbs == 0] = np.nan
    zbbbs[rbbbs == 0] = np.nan  # Use same mask for Z
    
    # Set boundary data for all time slices
    for i in range(n_times):
        if i < rbbbs.shape[0]:
            ods['equilibrium']['time_slice'][i]['boundary']['outline']['r'] = rbbbs[i, :]
            ods['equilibrium']['time_slice'][i]['boundary']['outline']['z'] = zbbbs[i, :]
    
    # Process X-point data with NaN filtering
    rxpt1 = efit_data['rxpt1']
    zxpt1 = efit_data['zxpt1'] 
    rxpt2 = efit_data['rxpt2']
    zxpt2 = efit_data['zxpt2']
    
    # Set NaN where X-point data is 0 (invalid)
    rxpt1[rxpt1 == 0] = np.nan
    zxpt1[rxpt1 == 0] = np.nan
    rxpt2[rxpt2 == 0] = np.nan  
    zxpt2[rxpt2 == 0] = np.nan
    
    # Set X-point data for all time slices (up to 2 X-points)
    for i in range(n_times):
        time_slice = ods['equilibrium']['time_slice'][i]
        
        # Set X-point array size
        if i < rxpt1.shape[0]:
            time_slice['boundary']['x_point'][0]['r'] = rxpt1[i]
            time_slice['boundary']['x_point'][0]['z'] = zxpt1[i]
            
        if i < rxpt2.shape[0]:
            time_slice['boundary']['x_point'][1]['r'] = rxpt2[i] 
            time_slice['boundary']['x_point'][1]['z'] = zxpt2[i]
    
    # Note: Global quantities, vacuum field, COCOS, and code info are handled by simple TDI expressions
    
    printd(f'Successfully loaded EFIT data for {n_times} time slices', topic='machine')


# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def pf_current_measurements(ods, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load PF current measurements that require stacking multiple signals
    
    Replaces py2tdi(stack_outer_2, ...) expressions for TokSearch compatibility
    """
    printd(f'Loading PF current measurements from {EFIT_tree}...', topic='machine')
    
    provider = ods.get_mds_provider('d3d') 
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
    
    # Stack E-coil and F-coil data along outer dimension
    measured = np.concatenate([efit_data['eccurt'], efit_data['fccurt']], axis=0)
    measured_error = np.concatenate([efit_data['sigecc'], efit_data['sigfcc']], axis=0)
    weight = np.concatenate([efit_data['fwtec'], efit_data['fwtfc']], axis=0)
    reconstructed = np.concatenate([efit_data['cecurr'], efit_data['ccbrsp']], axis=0)
    chi_squared = np.concatenate([efit_data['chiecc'], efit_data['chifcc']], axis=0)
    
    # Set PF current constraint data
    n_times = measured.shape[1] if len(measured.shape) > 1 else 1
    n_pf = measured.shape[0]
    
    for i in range(n_times):
        time_slice = ods['equilibrium']['time_slice'][i]
        
        # Set array size
        time_slice['constraints']['pf_current'][:] = n_pf
        
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


# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def psi_profiles(ods, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load PSI profiles that require complex transformations
    
    Replaces py2tdi(efit_psi_to_real_psi_2d, ...) and py2tdi(geqdsk_psi, ...) expressions
    """
    printd(f'Loading PSI profiles from {EFIT_tree}...', topic='machine')
    
    provider = ods.get_mds_provider('d3d')
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
    
    # Convert EFIT PSI to real PSI
    ssimag = efit_data['ssimag'].reshape(-1, 1)
    ssibry = efit_data['ssibry'].reshape(-1, 1)
    psin = efit_data['psin'].reshape(1, -1)
    
    # Calculate normalized PSI for profiles (geqdsk_psi function replacement)
    geqdsk_psi = ssimag + (ssibry - ssimag) * psin
    
    # Set PSI profiles
    n_times = len(ssimag) if hasattr(ssimag, '__len__') else 1
    
    for i in range(n_times):
        time_slice = ods['equilibrium']['time_slice'][i]
        
        # Set profiles_1d PSI (only this complex calculation needed)
        if i < len(geqdsk_psi):
            time_slice['profiles_1d']['psi'] = geqdsk_psi[i] if len(geqdsk_psi.shape) > 1 else geqdsk_psi
        
        # Set pressure constraint positions (efit_psi_to_real_psi_2d replacement)
        if 'rpress' in efit_data and len(efit_data['rpress']) > 0:
            rpress = -efit_data['rpress'] if i < len(efit_data['rpress']) else -efit_data['rpress'][0]
            pressure_psi = ssimag[i] + (ssibry[i] - ssimag[i]) * rpress
            time_slice['constraints']['pressure'][:]['position']['psi'] = pressure_psi
        
        # Set j_tor constraint positions (efit_psi_to_real_psi_2d replacement)  
        if 'sizeroj' in efit_data and len(efit_data['sizeroj']) > 0:
            sizeroj = efit_data['sizeroj'][i] if i < len(efit_data['sizeroj']) else efit_data['sizeroj'][0]
            jtor_psi = ssimag[i] + (ssibry[i] - ssimag[i]) * sizeroj
            time_slice['constraints']['j_tor'][:]['position']['psi'] = jtor_psi


# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def grid_2d_data(ods, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load 2D grid data that requires tiling operations
    
    Replaces py2tdi(tile, ...) expressions for grid dimensions
    """
    printd(f'Loading 2D grid data from {EFIT_tree}...', topic='machine')
    
    provider = ods.get_mds_provider('d3d')
    pulse_id = get_pulse_id(pulse, EFIT_run_id)
    
    # Get grid data
    TDIs = {
        'r_grid': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.R',
        'z_grid': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.Z',
        'bcentr': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.BCENTR'
    }
    
    efit_data = provider.raw(EFIT_tree, pulse_id, TDIs)
    
    # Tile R and Z grids across time dimension
    r_grid = efit_data['r_grid']
    z_grid = efit_data['z_grid']
    n_times = len(efit_data['bcentr'])
    
    # Create tiled grids: tile(R, size(BCENTR)) and tile(Z, size(BCENTR))
    r_tiled = np.tile(r_grid, (n_times, 1, 1)).transpose(1, 0, 2)
    z_tiled = np.tile(z_grid, (n_times, 1, 1)).transpose(1, 0, 2)
    
    # Set 2D grid data
    ods['equilibrium']['time_slice'][:]['profiles_2d'][:]['grid']['dim1'] = r_tiled
    ods['equilibrium']['time_slice'][:]['profiles_2d'][:]['grid']['dim2'] = z_tiled
    
    # Set grid type index (tiled constant 1)
    grid_type_index = np.tile(1, n_times).transpose(1, 0)
    ods['equilibrium']['time_slice'][:]['profiles_2d'][:]['grid_type']['index'] = grid_type_index


# ================================
@machine_mapping_function(__regression_arguments__, pulse=194844, EFIT_tree='EFIT01', EFIT_run_id='')
def convergence_data(ods, pulse, EFIT_tree='EFIT01', EFIT_run_id=''):
    """
    Load convergence data that requires complex axis operations
    
    Replaces py2tdi(get_largest_axis_value, ...) expressions
    """
    printd(f'Loading convergence data from {EFIT_tree}...', topic='machine')
    
    provider = ods.get_mds_provider('d3d')
    pulse_id = get_pulse_id(pulse, EFIT_run_id)
    
    # Get convergence error data
    TDIs = {
        'cerror': f'\\{EFIT_tree}::TOP.MEASUREMENTS.CERROR',
        'cerror_dim': f'dim_of(\\{EFIT_tree}::TOP.MEASUREMENTS.CERROR,0)'
    }
    
    efit_data = provider.raw(EFIT_tree, pulse_id, TDIs)
    
    # Get largest axis value (equivalent to get_largest_axis_value function)
    cerror_dim = efit_data['cerror_dim']
    cerror = efit_data['cerror']
    
    # Find the largest axis value
    max_axis_value = np.max(cerror_dim) if hasattr(cerror_dim, '__len__') else cerror_dim
    
    # Set convergence data
    n_times = len(cerror) if hasattr(cerror, '__len__') else 1
    
    for i in range(n_times):
        time_slice = ods['equilibrium']['time_slice'][i]
        time_slice['convergence']['iterations_n'] = max_axis_value


# ================================
# Add test function call
if __name__ == '__main__':
    # Test with d3d pulse
    test_ods = ODS()
    equilibrium_time_slice_data(test_ods, 194844)
    print(f"Loaded equilibrium data with {len(test_ods['equilibrium']['time'])} time points")