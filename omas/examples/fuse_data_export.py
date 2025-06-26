import time
import omas
import sys
from omas.omas_utils import printe
from omas.machine_mappings import d3d
from numpy import *
import argparse


def fuse_export(save_path, device, shot, EFIT_TREE, PROFILES_TREE, EFIT_RUN_ID, PROFILES_RUN_ID):
    ods = omas.ODS()

    tic = time.time()
    if device.lower() != "d3d":
        raise ValueError(f"Unsupported device {device}. Only 'd3d' supported at present.")
    printe("- Fetching ec_launcher data")
    d3d.ec_launcher_active_hardware(ods, shot)

    # printe("- Fetching nbi data")
    # d3d.nbi_active_hardware(ods, shot)

    printe("- Fetching core_profiles data")
    d3d.core_profiles_profile_1d(ods, shot, PROFILES_tree=PROFILES_TREE, 
                                 PROFILES_run_id=PROFILES_RUN_ID)

    printe("- Fetching wall data")
    d3d.wall(ods, shot)

    printe("- Fetching coils data")
    d3d.pf_active_hardware(ods, shot)
    d3d.pf_active_coil_current_data(ods, shot)

    printe("- Fetching flux loops data")
    d3d.magnetics_floops_data(ods, shot)

    printe("- Fetching magnetic probes data")
    d3d.magnetics_probes_data(ods, shot)

    printe("- Fetching Thomson scattering data")
    d3d.thomson_scattering_data(ods, shot)

    printe("- Fetching charge exchange data")
    d3d.charge_exchange_data(ods, shot, analysis_type="$(CER_analysis_type)")

    printe("- Fetching summary data")
    d3d.summary(ods, shot)

    printe("- Fetching equilibrium data")
    with ods.open(device, shot, options={'EFIT_tree': EFIT_TREE, "EFIT_run_id": EFIT_RUN_ID}):
        for k in range(len(ods["equilibrium.time"])):
            ods["equilibrium.time_slice"][k]["time"]
            ods["equilibrium.time_slice"][k]["global_quantities.ip"]
            ods["equilibrium.time_slice"][k]["profiles_1d.psi"]
            ods["equilibrium.time_slice"][k]["profiles_1d.f"]
            ods["equilibrium.time_slice"][k]["profiles_1d.pressure"]
            ods["equilibrium.time_slice"][k]["profiles_2d[0].psi"]
            ods["equilibrium.time_slice"][k]["profiles_2d[0].grid.dim1"]
            ods["equilibrium.time_slice"][k]["profiles_2d[0].grid.dim2"]
            ods["equilibrium.time_slice"][k]["profiles_2d[0].grid_type.index"] = 1
            ods["equilibrium.vacuum_toroidal_field.r0"]
            ods["equilibrium.vacuum_toroidal_field.b0"]

    printe(f"Data fetched via OMAS in {time.time()-tic:.2f} [s]")

    printe("Saving ODS to $filename", end="")
    tic = time.time()
    ods.save(save_path)
    printe(f" Done in {time.time()-tic:.2f} [s]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 3 mandatory arguments (positional)
    parser.add_argument('save_path')
    parser.add_argument('device')
    parser.add_argument('shot', type=int)
    parser.add_argument('EFIT_TREE')
    parser.add_argument('PROFILES_TREE')

    # 2 optional arguments (with -- prefix, default to None)
    parser.add_argument('--EFIT_RUN_ID', default=None)
    parser.add_argument('--PROFILES_RUN_ID', default=None)

    # Parse the arguments
    args = parser.parse_args()

    fuse_export(args.save_path, args.device, args.shot, args.EFIT_TREE, args.PROFILES_TREE, args.EFIT_RUN_ID, args.PROFILES_RUN_ID)
    