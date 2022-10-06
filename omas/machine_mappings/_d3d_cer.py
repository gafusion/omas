# -*-Python-*-
# Created by shaskey at 16 Oct 2015  11:23
#
# Get the data for the impurity and main ion CER systems

import scipy.constants as consts
import os
import copy
import numpy as np
import re
import warnings
from uncertainties.unumpy import uarray, std_devs, nominal_values
import pickle
import xarray
from omas.omas_machine import mdstree, mdsvalue
from omas.omas_core import ODS


def d3d_cer(
    shot,
    systems=None,
    Zeeman_correction=False,
    Zeeman_NN_loc="/fusion/projects/results/cer/haskeysr/Zeeman_NN_py3",
    Zeeman_inc_uncorr=False,
    fetch_VB=False,
    ods=None,
    return_xarray=False,
    split_CER_by_beam=True,
    verbose=False,
    impCERtang_uncor_CER_rotation=False,
    impCERtang_uncor_fix_geom=True,
    include_intensity_meas=False,
    nc_data_loc=None,
    nc_extra_label=None,
    get_FIDASIM_corrected=False,
):
    '''
    Possible CERFIT lines
      1       D I 4-2         4860.00 A
      2       B V 7-6         4944.65 A
      3       Ne X 11-10      5249.10 A
      4       N VII 9-8       5669.33 A
      5       He II 4-3       4685.68 A
      6       O VIII 8-7      2975.70 A
      7       D I 3-2         6561.03 A
      8       C VI 8-7        5290.50 A
      9       C VI 7-6        3433.65 A
     10       Ar XVI 14-13    4365.53 A
     11       Ar XVI 13-12    3463.56 A
     12       Ar XVIII 14-13  3448.92 A
     13       C VI 9-8        7716.80 A
     14       F IX 9-8        3429.39 A
     15       F IX 10-9       4794.50 A
     16       Ca XVIII 14-13  3448.92 A
     17       Ca XX 15-14     3462.80 A
     18       Li I 2-1        6707.78 A
     19       C IV 6h-7i      7726.2  A
     20       Li III 7-5      5167.0  A
    '''

    elementmass = {'D': 2, 'B': 11, 'Ne': 20, 'N': 14, 'He': 4, 'O': 16, 'C': 12, 'Ar': 40, 'F': 19, 'Ca': 40, 'Li': 7}
    romancharge = {
        'I': 1,
        'V': 5,
        'X': 10,
        'VII': 7,
        'II': 2,
        'VIII': 8,
        'VI': 6,
        'XVI': 16,
        'XVIII': 18,
        'IX': 9,
        'XX': 20,
        'IV': 4,
        'III': 3,
    }

    # os.environ["OMAS_DEBUG_TOPIC"] = 'machine'

    server = 'DIII-D'
    treename = 'IONS'

    if systems is None:
        systems = (('miCERtang', None), ('impCERvert', 'best'), ('impCERtang', 'best'))

    scratch = {}
    printe = printd = printi = printw = print

    print('  * Fetching CER data')

    def data_valid(shot, chord, meas_name):
        # List of chords with intensity calibraiton errors for FY15, FY16 shots after
        # CER upgraded with new fibers and cameras.
        disableChanVert = ['impCERvert_v03', 'impCERvert_v04', 'impCERvert_v05', 'impCERvert_v06', 'impCERvert_v23', 'impCERvert_v24']
        if (shot >= 162163) & (shot <= 167627) & (chord in disableChanVert) & (meas_name == 'n_12C6'):
            printe('Not fetching {} for {} for {}'.format(meas_name, chord, shot))
            return False
        return True

    def get_data(mdsstring, verbose, get_time=False):
        if verbose:
            print_fun = printi
        else:
            print_fun = printd
        # Always fetch calibration related data from the MDSplus tree
        mdsstring_orig = mdsstring
        mdsstring = mdsstring.lower()
        if (mdsstring.find('.calibration') >= 0) or (mdsstring.find('date_loaded') >= 0) or (nc_data_loc is None):
            print_fun(' ' * 6 + 'System: %s \t Measurement: %s \t MDS: %s' % (sys_name, mdsstring.split('.')[-1], mdsstring))
            tmp = mdsvalue('d3d', treename, shot, TDI=mdsstring)
            # Check if the returned data is valid, if not, return None
            try:
                dat = tmp.data()
                if get_time:
                    dim_of = tmp.dim_of(0)
            except Exception:
                dat = None
                if get_time:
                    dim_of = None
            if get_time:
                output = dat, dim_of
            else:
                output = dat
        else:
            # For fetching main ion data from outside MDSplus
            split_string = mdsstring_orig.split(':')
            channel_name = split_string[-2].split('.')[-1]
            ch_num = int(channel_name[-2:])
            ch_name = 'm' + channel_name[-2:]
            meas_name = split_string[-1]
            if get_FIDASIM_corrected:
                nc_loc = scratch.setdefault('MI_FSIM_{}'.format(shot), {})
            else:
                nc_loc = scratch.setdefault('MI_{}'.format(shot), {})
            if ch_name not in nc_loc:
                if get_FIDASIM_corrected:
                    end_txt = "FIDASIM_mdsplus"
                else:
                    end_txt = "mdsplus"
                if nc_extra_label is None:
                    nc_extra_txt = ''
                else:
                    nc_extra_txt = nc_extra_label + '_'
                input_nc_fname = "{}/{}/{}/{}{}_{}{}.nc".format(nc_data_loc, shot, ch_name, shot, ch_name, nc_extra_txt, end_txt)
                if os.path.isfile(input_nc_fname):
                    nc_loc[ch_name] = OMFITnc(input_nc_fname)
                else:
                    print("couldn't find netcdf file:{}".format(input_nc_fname))
            failed = True
            if ch_name in nc_loc:
                # Allow the measurement name as specified, or lower cased
                if (meas_name.lower() in nc_loc[ch_name]) or (meas_name in nc_loc[ch_name]):
                    use_name = meas_name.lower() if (meas_name.lower() in nc_loc[ch_name]) else meas_name
                    location = +nc_loc[ch_name][use_name]['data']
                    location_t = +nc_loc[ch_name]['time']['data']
                    if get_time:
                        output = +location, +location_t
                    else:
                        output = +location
                    failed = False
            if failed:
                if get_time:
                    output = None, None
                else:
                    output = None
        return output

    def get_measurement(loc, chord, mds_name, err, mult, time_axis):
        node = '{}.channel{:02d}:{}'.format(loc, chord, mds_name)
        dat = get_data('\IONS::TOP{}'.format(node), verbose)

        def check_fiducial():
            dat = get_data(fid_loc.format(cal_loc, chord), verbose)
            printe(' ' * 6 + 'Error with {} on {}: fiducial:{}'.format(mds_name, ch_format.format(chord), dat))

        if dat is not None:
            dat_err = None
            # Check for -1e30 cases and set them to NaN
            dat = np.atleast_1d(dat)
            mask = dat < -1.0e15
            dat[mask] = np.nan
            if (np.sum(mask) >= 1) and (mds_name in ['ROTC', 'ROT']):
                printe(
                    ' ' * 6 + 'Problem with {} on {}, {} of {} < -1e15'.format(mds_name, ch_format.format(chord), np.sum(mask), len(mask))
                )
                check_fiducial()
            if err != None:
                node_err = '{}.channel{:02d}:{}'.format(loc, chord, err)
                dat_err = get_data('\IONS::TOP{}'.format(node_err), verbose)
            if dat_err is None:
                uar2 = dat * mult
            else:
                # ignore data with zero value except where we expect zeros....
                uar2 = uarray(dat * mult, np.abs(dat_err) * mult)
        else:
            printe(' ' * 6 + 'Problem with {} on {}, data is None'.format(mds_name, ch_format.format(chord)))
            uar2 = time_axis * np.nan
            # If this is from rotation, check to see if the reason is because the fiducial calc failed
            if mds_name in ['ROTC', 'ROT']:
                check_fiducial()
        return uar2

    def imp_dens(time, stime, analysis_type, chord):
        """
        Fetches the impurity density calculation from IMPCON which is
        based on impurity density fits. Currenly this requires cross
        referencing the times with the data stored in the relevant
        analysis type tree to figure out the R and Z of each measurement
        """
        ni_data_out = time * np.nan
        meas_list_ni = {}
        if imp_ch_format != None:
            impurity_node = '\IONS::top.impdens.{}.nz{}'.format(analysis_type, imp_ch_format.format(chord))
            ni_data, ni_time = get_data(impurity_node, verbose, get_time=True)
        else:
            ni_data = None
            ni_time = None
        if ni_data is not None:
            # Now we need to find which items in ni_time correspond with the cerquick or cerauto analysis so we can
            # correct the time (the impdens data should be a subset of the temp and rotation data
            # The rounding is to get rid of odd numerical issues - i.e. sometimes the time will be 4090.00024414 instead of 4090
            t_tmp = np.round(time - 0.5 * stime, 3)
            for t_cur, ni_cur in zip(np.round(ni_time, 3), ni_data):
                tmp_mask = t_tmp == t_cur
                # For now we throw away data with multiple time (except for the first item)
                if np.sum(tmp_mask) == 1:
                    ni_data_out[tmp_mask] = ni_cur
            # Give it a 10% error for now
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in *')
            ni_data_w_err = uarray(ni_data_out, ni_data_out * 0.1)
        return ni_data_w_err

    def calculate_pinj(beam_geometry, beam_order, times, stimes, tssub_times, tssub_stimes, beam_dat_dict):
        # This calculates the energy from each beam during the time slice and subtraction time slice
        # for beams that have non zero geometry factors; however, the geometry factor is not applied to the powers
        cur_output = {}
        # For beams that the chord can see (based on geometry)
        valid_beams = []
        for tmp in beam_order[beam_geometry > 0.0]:
            # 4 characters for 30LT, 5 characters for 330lT
            # Shorten to 4 character standard
            if len(tmp) == 5:
                valid_beams.append((tmp[:2] + tmp[3]).lower())
            else:
                valid_beams.append(tmp[:3].lower())
        if not np.alltrue(np.diff(times) >= 0):
            printe('Times not monotonic')

        # Loop through the beams
        for cur_beam in valid_beams:
            # This is stored at 100kHz, do we need to upsample for better resolution?
            beam_time, beam_dat = beam_dat_dict[cur_beam]
            # For some shots i.e 145456 there is an issue with retrieving the beam information
            # Skip the calculation if this is the case. Should only be a problem for splitting CER measurements by beam_dat
            if beam_time is None:
                printe('Unable to get beam information for CER, only important if splitting CER measurements by beam')
            else:
                dt = beam_time[1] - beam_time[0]
                # This assumes that CER times in the RAW DA is in the middle of the integration time
                start_indices = np.searchsorted(beam_time, times - 0.5 * stimes)
                end_indices = np.searchsorted(beam_time, times + 0.5 * stimes)

                cur_output[cur_beam + '_ts'] = np.array(
                    [np.sum(beam_dat[start:end]) * dt / stime for start, end, stime in zip(start_indices, end_indices, stimes)]
                )
                # Check whether or not there is time slice subtraction
                if np.count_nonzero(tssub_stimes) > 0:
                    tssub_start_indices = np.searchsorted(beam_time, tssub_times)
                    tssub_end_indices = np.searchsorted(beam_time, tssub_times + tssub_stimes)
                    cur_output[cur_beam + '_tssub'] = np.array(
                        [
                            np.sum(beam_dat[start:end]) * dt / stime
                            for start, end, stime in zip(tssub_start_indices, tssub_end_indices, tssub_stimes)
                        ]
                    )
                    cur_output[cur_beam + '_tssub'][tssub_stimes == 0.0] = 0.0
                else:
                    cur_output[cur_beam + '_tssub'] = np.zeros(times.shape, dtype=float)
        return cur_output

    def use_zeeman_NN(
        Ti_obs_vals,
        modB_vals,
        theta_vals,
        NN=None,
        X_scaler=None,
        Y_scaler=None,
        species=None,
        loc='/fusion/projects/results/cer/haskeysr/Zeeman_NN/',
        plot=False,
    ):
        """
        NN, X_scaler, Y_scaler: neural network, X_scaler and Y_scaler objects from the neural network fitting
        Ti_obs_vals: Observed Ti values [eV]
        modB_vals: Magnetic field strength [T], same length and Ti_obs_vals
        theta_vals: Viewing angle [deg], same length and Ti_obs_vals
        """
        if NN is None:
            fname = '{}/Zeeman_Corr_NN_{}.pickle'.format(loc, species)
            print("Loading NN from:{}".format(fname))
            with open(fname, 'rb') as filehandle:
                NN_dat = pickle.load(filehandle)
            NN = NN_dat['NN']
            X_scaler = NN_dat['X_scaler']
            Y_scaler = NN_dat['Y_scaler']
        X = np.zeros((len(Ti_obs_vals), 4), dtype=float)
        X[:, 0] = +modB_vals  # .flatten()
        X[:, 1] = np.log10(Ti_obs_vals)  # .flatten())
        X[:, 2] = np.sin(np.deg2rad(theta_vals))  # .flatten()))
        X[:, 3] = np.cos(np.deg2rad(theta_vals))  # .flatten()))
        X_norm = X_scaler.transform(X, copy=True)
        Y_NN_norm = NN.predict(X_norm)
        Y_NN = Y_scaler.inverse_transform(Y_NN_norm, copy=True)
        Ti_real = Y_NN + Ti_obs_vals
        if plot:
            from matplotlib import pyplot

            fig, ax = pyplot.subplots(nrows=2, sharex=True)
            ax[0].plot(Ti_obs_vals, Ti_real, ',')
            ax[0].set_ylabel('Ti real')
            ax[1].plot(Ti_obs_vals, Y_NN, ',')
            ax[1].set_ylabel('Ti_real - Ti_obs')
            ax[-1].set_xlabel('Ti obs')
        return Ti_real

    def single_chord_zeeman_corr(in_dat, NN, X_scaler, Y_scaler, species='2H1'):
        """
        in_dat: an input CER dataset, should only have one channel
        NN, X_scaler, Y_scaler: neural network, X_scaler and Y_scaler objects from the neural network fitting
        """
        # Do this so that data ends up being 1D arrays instead of 1xY arrays
        in_dat_red = in_dat.isel(channel=0)
        times = in_dat_red['time'].values
        R_chord = in_dat_red['R'].values
        Z_chord = in_dat_red['Z'].values
        R_chord_mag_time = np.interp(mag_field['time'], times, in_dat_red['R'].values)
        Z_chord_mag_time = np.interp(mag_field['time'], times, in_dat_red['Z'].values)
        # Need a better way of doing this, time varying best index?
        R_inds = np.argmin(np.abs(mag_field['R_vals'][np.newaxis, :] - R_chord_mag_time[:, np.newaxis]), axis=1)
        Z_inds = np.argmin(np.abs(mag_field['Z_vals'][np.newaxis, :] - Z_chord_mag_time[:, np.newaxis]), axis=1)
        t_inds = np.arange(len(Z_inds))
        # Find Bt and Bp as a function of time for the chord measurement location, using nearest interpolation for R and Z, and linear for time
        Bt_vals_time = np.interp(times, mag_field['time'], mag_field['Bt_arr'][R_inds, Z_inds, t_inds])
        Bz_vals_time = np.interp(times, mag_field['time'], mag_field['Bz_arr'][R_inds, Z_inds, t_inds])
        Bp_vals_time = np.interp(times, mag_field['time'], mag_field['Bp_arr'][R_inds, Z_inds, t_inds])
        # Now we have the magnetic field strength
        modB = np.sqrt(Bt_vals_time**2 + Bp_vals_time**2)
        # Next we want to find the angle between the view and the magnetic field (theta)
        # Assemble time varying beam direction vectors
        # Note we use 90-phi because CER phi coords are stored in DIII-D machine coordinates which is clockwise starting at North when viewed from above
        # This changes to anti-clockwise, starting from East when viewed from above
        tor_angle = np.deg2rad((90.0 - in_dat_red['phi'].values) % 360.0)
        B = np.zeros((len(times), 3), dtype=float)
        B[:, 0] = -Bt_vals_time * np.sin(tor_angle)  # x component (east)
        B[:, 1] = Bt_vals_time * np.cos(tor_angle)  # y component (north)
        B[:, 2] = Bz_vals_time  # z component (up)
        B_hat = B / np.sqrt(np.sum(B**2, axis=1))[:, np.newaxis]
        # Chord viewing direction
        # Note we use 90-phi because LOS_pt1/2_phi are stored in DIII-D machine coordinates which is clockwise starting at North when viewed from above
        # This changes to anti-clockwise, starting from East when viewed from above
        c_pt1_rzphi = np.array([in_dat_red['LOS_pt1_R'].values, in_dat_red['LOS_pt1_Z'].values, 90.0 - in_dat_red['LOS_pt1_phi'].values])
        c_pt2_rzphi = np.array([in_dat_red['LOS_pt2_R'].values, in_dat_red['LOS_pt2_Z'].values, 90.0 - in_dat_red['LOS_pt2_phi'].values])
        # Same as B, x,y,z:east,north,up
        c_pt1_xyz = np.array(
            [c_pt1_rzphi[0] * np.cos(np.deg2rad(c_pt1_rzphi[2])), c_pt1_rzphi[0] * np.sin(np.deg2rad(c_pt1_rzphi[2])), c_pt1_rzphi[1]]
        )
        c_pt2_xyz = np.array(
            [c_pt2_rzphi[0] * np.cos(np.deg2rad(c_pt2_rzphi[2])), c_pt2_rzphi[0] * np.sin(np.deg2rad(c_pt2_rzphi[2])), c_pt2_rzphi[1]]
        )
        # Unit vector pointing from lens to chord-beam intersection
        chord_hat = c_pt1_xyz - c_pt2_xyz
        chord_hat = chord_hat / np.sqrt(np.sum(chord_hat**2))
        # Put into 2D array for easier dot product, time x {xyz}
        chord_hat_2D = B_hat * 0 + chord_hat[np.newaxis, :]
        # Angle between unit vectors using: a dot b = |a||b| cos(theta)
        angle_B_chord = np.rad2deg(np.arccos(np.sum((chord_hat_2D * B_hat), axis=1)))
        uncorr_name = 'T_orig_{}'.format(species)
        corr_name = 'T_{}'.format(species)
        # Backup the uncorrected temperature data
        in_dat[uncorr_name] = in_dat[corr_name] * 1
        Ti_obs, Ti_err = nominal_values(in_dat_red[corr_name].values), std_devs(in_dat_red[corr_name].values)
        # Only bother if there is any valid data to correct
        mask = np.isfinite(Ti_obs)
        if np.sum(mask) >= 1:
            Ti_real = use_zeeman_NN(Ti_obs[mask], modB[mask], angle_B_chord[mask], NN=NN, X_scaler=X_scaler, Y_scaler=Y_scaler)
            diff = Ti_obs[mask] - Ti_real
            print(
                "Channel:{}, modB [{:.2f},{:.2f}], theta [{:.2f},{:.2f}], val[{:.2f},{:.2f}], corr [{:.2f},{:.2f}]".format(
                    in_dat['channel'].values,
                    np.min(modB),
                    np.max(modB),
                    np.min(angle_B_chord),
                    np.max(angle_B_chord),
                    np.min(Ti_real),
                    np.max(Ti_real),
                    np.min(diff),
                    np.max(diff),
                )
            )
            in_dat[corr_name].values[0, mask] = uarray(Ti_real, Ti_err[mask])

    def fetch_system():
        output_list = []
        output_list_ni = []
        pinj_output_dat = {}
        for chord in val_chords:
            # for chord in range(3):
            chord_success = False
            node_time = '{}.channel{:02d}:{}'.format(loc, chord, 'time')
            node_stime = '{}.channel{:02d}:{}'.format(loc, chord, 'stime')
            t_start = get_data('\IONS::TOP{}'.format(node_time), verbose)
            if t_start is None:
                continue
            # If time exists we assume that measurements will exist....
            s_time = get_data('\IONS::TOP{}'.format(node_stime), verbose)
            lineid = get_data(lineid_loc.format(chord), verbose)
            if lineid is None:
                printe('Skipping %s because no lineid' % chord)
                continue
            if isinstance(lineid, bytes):
                lineid = lineid.decode("utf-8")
            wavelength = [get_data(wavelength_loc.format(chord), verbose)]

            # Time of measurement is actually the time recorded plus half the integration time
            t = t_start + 0.5 * s_time
            meas_dict = {}
            # What is the best way to handle a failure on this
            # Parse the lineID, we expect an:
            # element that begins with a capital followed by possibly a lower case letter,
            # Possible spaces, then capital roman numerals of unknown length for the Z
            # Possible spaces, then the transition which should be two numbers with a dash between them
            # However, to accomodate something like C IV 6h-7i, we include possible lower case letters the transition
            tmp = re.search('([A-Z][a-z]*) *([A-Z]*) *([0-9]*[a-z]*-[0-9]*[a-z]*)', lineid)
            element, charge, transition = tmp.group(1), tmp.group(2), tmp.group(3)
            element_mass_value = elementmass[element]
            if element == 'D':
                element = 'H'
            line_str = '{}{}{}'.format(element_mass_value, element, romancharge[charge])
            coords = {'channel': ['{s}_{c}'.format(s=sys_name, c=ch_format.format(chord))], 'time': t}
            for mds_name, err, key_name, mult, data_type, units in data_to_get:
                key_name = key_name.format(line_str=line_str)
                uar2 = get_measurement(loc, chord, mds_name, err, mult, t)
                if data_type == 'var':
                    if sys_name == 'miCERtang':
                        if np.sum(np.isfinite(nominal_values(uar2))) == 0:
                            continue
                        else:
                            meas_dict[key_name] = (['channel', 'time'], np.atleast_2d(uar2), {'units': units})
                    else:
                        meas_dict[key_name] = (['channel', 'time'], np.atleast_2d(uar2), {'units': units})
                else:
                    coords[key_name] = (['time'], uar2)
                    # coords[key_name] = uar2
            if len(meas_dict) >= 1:
                attrs = {
                    'system': sys_name,
                    'channel': ch_format.format(chord),
                    'analysis': analysis_type,
                    'FIT_message': '{} analysis'.format(analysis_type.upper()),
                }
                if sys_name == 'impCERvert':
                    attrs[
                        'FIT_V_pol_{}_message'.format(line_str)
                    ] = 'Only fit if interested in Er. Core channels are de-selected by default\ndue to inaccuracies because of atomic cross section effects.\nTalk to the CER group if this is not clear, or before enabling them.'

                if sys_name == 'impCERtang' and impCERtang_uncor_CER_rotation:
                    if impCERtang_uncor_fix_geom:
                        attrs[
                            'FIT_omega_tor_{}_message'.format(line_str)
                        ] = 'Using toroidal rotation without atomic physics correction but with\nLOS geometric correction'
                    else:
                        attrs[
                            'FIT_omega_tor_{}_message'.format(line_str)
                        ] = 'Using toroidal rotation without atomic physics correction or LOS\ngeometry correction'
                else:
                    attrs[
                        'FIT_omega_tor_{}_message'.format(line_str)
                    ] = 'Using toroidal rotation with atomic physics correction and LOS\ngeometry correction'
                LOS_pt1 = [get_data(k.format(chord), verbose) for k in LOS_pt1_loc]
                if None in LOS_pt1:
                    printe(' ' * 6 + 'Error getting LOS_pt1 for system:{}, channel:{}'.format(sys_name, chord))
                else:
                    coords['LOS_pt1_R'] = ('channel', [+LOS_pt1[0]])
                    coords['LOS_pt1_Z'] = ('channel', [+LOS_pt1[1]])
                    coords['LOS_pt1_phi'] = ('channel', [+LOS_pt1[2]])
                coords['wavelength'] = ('channel', wavelength)
                if VB_name in meas_dict:
                    # conversion from from ph/m2/sr/s/A to W/cm2/A for the VB measurement
                    # ph/s * hc/lambda -> W
                    # sr-1 * 4pi -> sr^0
                    # m-2 * 1e-4 -> cm-2
                    VB_conversion = consts.h * consts.c / (wavelength * 1.0e-10) * (4.0 * np.pi) * 1e-4
                    meas_dict[VB_name] = (meas_dict[VB_name][0], meas_dict[VB_name][1] * VB_conversion, {'units': 'W/cm**2/A'})
                for tmp_name, tmp_node in attrs_list:
                    attrs[tmp_name] = get_data(tmp_node.format(chord), verbose)

                # Is it worth adding this to co-ordinates along with beamgeometry ?

                # This is required for the older shots where this information was not stored in the tree
                if shot <= 124510:
                    beam_order_tmp = np.array(['30LT', '30RT', '330LT', '330RT'])
                else:
                    beam_order_tmp = beam_order
                attrs['beam_order'] = np.array([ii.replace(' ', '') for ii in beam_order_tmp])
                # Calculate the beam power for each measurement and put in coords

                # Add this kludge because sometimes there is signal even though all non-zero
                # beam geometry beams are off
                # the 330 mask is due to an error where the beam geometry for the 210's is sometimes non zero for the verticals
                # Maybe expand this later so that it will also work for the tangentials
                if orientation == 'vertical':
                    mask = np.array([tmp.find('330') >= 0 for tmp in attrs['beam_order']])
                    attrs['beamgeometry'] = attrs['beamgeometry'] * mask + mask * 0.001
                # pinj_output_dat will only have beam power data for beams which have non zero beam geometry
                pinj_output_dat[coords['channel'][0]] = calculate_pinj(
                    attrs['beamgeometry'],
                    attrs['beam_order'],
                    coords['time'],
                    coords['stime'][1],
                    coords['ttsub'][1],
                    coords['ttsub_stime'][1],
                    beam_dat_dict,
                )
                power_array = np.zeros((attrs['beam_order'].shape[0], len(t)), dtype=float)
                count = 0
                beam_order2 = np.array(
                    [i.lower().replace('330', '33').replace('150', '15').replace('210', '21') for i in attrs['beam_order']]
                )
                # This goes through all beams, (not just the ones that the chord sees) and
                # adds power information for ts and tssub. If the beam is not in pinj_output_dat for this chord
                # i.e it has no beamgeometry, then zero values are used
                for i in beam_order2:
                    beam_deg = i[0:2]
                    source = i[2]
                    ts_key = '{}{}_ts'.format(beam_deg, source)
                    tssub_key = '{}{}_tssub'.format(beam_deg, source)
                    # Use data from pinj_output_dat if it is there (beamgeometry is non zero)
                    # otherwise put zeros in
                    if tssub_key in pinj_output_dat[coords['channel'][0]]:
                        coords[ts_key] = (['time'], pinj_output_dat[coords['channel'][0]][ts_key])
                        coords[tssub_key] = (['time'], pinj_output_dat[coords['channel'][0]][tssub_key])
                    else:
                        coords[ts_key] = (['time'], t * 0)
                        coords[tssub_key] = (['time'], t * 0)
                    power_array[count, :] = coords[ts_key][1] - coords[tssub_key][1]
                    count += 1

                # At this point need to work out which beam each measurement is from. Should be recorded in MDSplus
                power_array = power_array * attrs['beamgeometry'][:, np.newaxis]
                total_power = np.sum(power_array, axis=0) + np.finfo(float).eps  # eps to avoid division by zero
                max_inds = np.argmax(power_array, axis=0)
                # We split beam and source so that it is easy to modify l, r and b
                beam_order3 = np.array([i[0:2] for i in beam_order2])
                beam_source = np.array([i[2] for i in beam_order2])
                active_beam = beam_order3[max_inds]
                active_source = beam_source[max_inds]
                max_power = power_array[max_inds, np.arange(max_inds.shape[0])]
                # Assume multiple beams if the fraction from the max beam is below .9
                multiple_beams_mask = (max_power / total_power) < 0.9
                active_source[multiple_beams_mask] = 'b'
                coords['active_beam'] = (['time'], np.core.defchararray.add(active_beam, active_source))

                # Add V_pol which will be filled in the second iteration
                # using data from Vtor, set intial data to Nan
                if sys_name == 'impCERvert':
                    dat = meas_dict['V_vert_{line_str}'.format(line_str=line_str)][1] * np.nan
                    meas_dict['V_pol_{line_str}'.format(line_str=line_str)] = (['channel', 'time'], dat, {'units': 'm/s'})

                # Create the list of measurements that need to be fit
                attrs['measurements'] = []
                attrs['measurements_to_fit'] = []
                for iii in [jjj.format(line_str=line_str) for jjj in measurements]:
                    if iii in meas_dict:
                        # really, we should have not fetched any deselected measurements to save time
                        # if root['SETTINGS']['PHYSICS']['DIII-D']['FETCH'].get(sys_name, {}).get(iii, True):
                        attrs['measurements'].append(iii)
                        attrs['measurements_to_fit'].append(iii)

                # Generate omega_tor if possible
                # Allow the option to use the uncorrected rotation data
                if impCERtang_uncor_CER_rotation and sys_name == 'impCERtang':
                    printw("Warning: Using uncorrected CER rotation data")
                    vtor_key = 'V_tor_uncor_{line_str}'.format(line_str=line_str)
                    # Apply geometry correction if required
                    if impCERtang_uncor_fix_geom:
                        R_tmp = meas_dict['R'][1][0, :]
                        # Note: phi stored in MDSplus for CER is zero at N, and +ve going clockwise when viewed from above ('DIII-D machine coords')
                        phi_tmp = meas_dict['phi'][1][0, :]
                        Z_tmp = meas_dict['Z'][1][0, :]
                        lens_rzphi = np.array(
                            [coords['LOS_pt1_R'][1][0], coords['LOS_pt1_Z'][1][0], coords['LOS_pt1_phi'][1][0]], dtype=float
                        )
                        # lens_xyz shape is 3
                        lens_xyz = np.array(
                            [
                                lens_rzphi[0] * np.sin(np.deg2rad(lens_rzphi[2])),
                                lens_rzphi[0] * np.cos(np.deg2rad(lens_rzphi[2])),
                                lens_rzphi[1],
                            ]
                        )
                        # plasma_xyz shape is 3 x n_times
                        plasma_xyz = np.array([R_tmp * np.sin(np.deg2rad(phi_tmp)), R_tmp * np.cos(np.deg2rad(phi_tmp)), Z_tmp])
                        # Calculate time varying angle between chord and phi_hat to correct LOS velocity for geometry
                        s = plasma_xyz - lens_xyz[:, np.newaxis]
                        mod_s = np.sqrt(np.sum(s**2, axis=0))
                        s_hat = s / mod_s[np.newaxis, :]
                        # Cannot just use phi stored in MDSplus for CER because it has zero pointing North, and +ve going clockwise when viewed from above
                        # tor_angle  below is CCW when viewed from above, and zero is at east (standard definition)
                        tor_angle = np.arctan2(plasma_xyz[1, :], plasma_xyz[0, :])
                        phi_hat = np.array([-np.sin(tor_angle), np.cos(tor_angle), tor_angle * 0])
                        chord_tor_corr = np.sum(phi_hat * s_hat, axis=0)
                    else:
                        chord_tor_corr = 1.0
                else:
                    vtor_key = 'V_tor_{line_str}'.format(line_str=line_str)
                    chord_tor_corr = 1.0
                omega_key = 'omega_tor_{line_str}'.format(line_str=line_str)
                vel_to_angular_list = [[vtor_key, omega_key]]

                # Include the geometrically corrected values for the main ion system
                if sys_name == 'miCERtang':
                    vel_to_angular_list.append(
                        ['V_tor_u_{line_str}'.format(line_str=line_str), 'omega_tor_u_{line_str}'.format(line_str=line_str)]
                    )
                for tmp_vtor, tmp_omega in vel_to_angular_list:
                    if (tmp_vtor in meas_dict) and ('R' in meas_dict):
                        # For the case where R=0, prevent this from raising an exception
                        # Note the outputs are masked based on 'ridiculous radius' of the measurement
                        # later in the script
                        R_vals = meas_dict['R'][1]
                        R_vals[R_vals == 0.0] = 1e-3
                        uar2 = (meas_dict[tmp_vtor][1] / chord_tor_corr) / R_vals
                        meas_dict[tmp_omega] = (['channel', 'time'], uar2, {'units': 'rad/s'})
                        attrs['measurements'].append(tmp_omega)
                        attrs['measurements_to_fit'].append(tmp_omega)

                # Get the impurity density data
                n_key = 'n_{line_str}'.format(line_str=line_str)
                get_imp_dens = True  # sys_name in ['impCERtang', 'impCERvert'] and root['SETTINGS']['PHYSICS']['DIII-D']['FETCH'].get(sys_name, {}).get(n_key, True)
                tmp_name = '{}_{}'.format(sys_name, ch_format.format(chord))
                if get_imp_dens and data_valid(shot, tmp_name, n_key):
                    uar = imp_dens(coords['time'], coords['stime'][1], analysis_type, chord)
                    meas_dict[n_key] = (['channel', 'time'], np.atleast_2d(uar), {'units': 'm^-3'})
                    attrs['measurements'].append(n_key)
                    attrs['measurements_to_fit'].append(n_key)
                if include_intensity_meas:
                    attrs['measurements'].append('Int_{line_str}'.format(line_str=line_str))
                    attrs['measurements_to_fit'].append('Int_{line_str}'.format(line_str=line_str))

                # Store a second line of sight for VB calculation
                coords['LOS_pt2_R'] = ('channel', [+meas_dict['R'][1].flatten()[0]])
                coords['LOS_pt2_Z'] = ('channel', [+meas_dict['Z'][1].flatten()[0]])
                coords['LOS_pt2_phi'] = ('channel', [+meas_dict['phi'][1].flatten()[0]])

                ds = xarray.Dataset(meas_dict, coords=coords, attrs=attrs)
                ds['LOS_pt2_R'].attrs['units'] = 'm'
                ds['LOS_pt2_Z'].attrs['units'] = 'm'
                ds['LOS_pt2_phi'].attrs['units'] = 'deg.'
                ds['active_beam'].attrs['units'] = ''

                # discard data with non-positive intensity or zero error
                # miCERtang is different because of a temporary bug loading the intensity data
                # ds.where was killing the units for individual measurements
                # needed to use the mask, indices, isel workaround
                mask = (ds['R'].values > 0.1) & (ds['R'].values < 3)
                if sys_name == 'miCERtang':
                    if f'Int_{line_str}' in ds:
                        mask = mask & (std_devs(ds[f'Int_{line_str}'].values) > 0)
                    indices = np.arange(mask.shape[1])[mask[0, :]]
                else:
                    mask = (
                        mask
                        & (nominal_values(ds['Int_{line_str}'.format(line_str=line_str)].values) > 0)
                        & (std_devs(ds['Int_{line_str}'.format(line_str=line_str)].values) > 0)
                    )
                    indices = np.arange(mask.shape[1])[mask[0, :]]
                # No point including this channel if we masked out all of the data
                if len(indices) == 0:
                    continue
                else:
                    ds = ds.isel(time=indices)
                    output_list.append(ds)
        return output_list, pinj_output_dat

    # Get the beam powers as a function of time
    # Here we are using pinjf which is power in watts
    # This should be replaced with a dedicated get_NBI at some point
    if 'beam_dat_{}'.format(shot) not in scratch:
        beam_dat = {}
        for i in ['30', '33', '21', '15']:
            for j in ['l', 'r']:
                try:
                    tmp = mdsvalue('d3d', 'nb', shot, TDI='.nb{}{}.pinjf_{}{}'.format(i, j, i, j))
                    beam_dat['{}{}'.format(i, j)] = [tmp.dim_of(0), tmp.data()]
                except Exception:
                    raise
                    beam_dat['{}{}'.format(i, j)] = [None, None]
        scratch['beam_dat_{}'.format(shot)] = beam_dat
    beam_dat_dict = scratch['beam_dat_{}'.format(shot)]

    # Get the CER tree structure through OMFIT so we can find the maximum channel number
    CER_tree = mdstree('d3d', treename, shot)

    # Loop through each of the systems (imp vert, imp tang, MI tang)
    pinj_dict = scratch.setdefault('pinj', {})
    for sys_name, analysis_type in systems:
        VB_name = 'VB_CER'
        print(' ' * 5 + 'Working on {}'.format(sys_name))
        results = {'30lt': [], '30rt': [], '21lt': [], '21rt': [], '33lt': [], '33rt': []}
        if nc_data_loc is not None and sys_name == 'miCERtang':
            analysis_type = 'best'
            # Clean the netCDF cache
            if 'MI_FSIM_{}'.format(shot) in scratch:
                del scratch['MI_FSIM_{}'.format(shot)]
            if 'MI_{}'.format(shot) in scratch:
                del scratch['MI_{}'.format(shot)]
        if analysis_type is None:
            print(' ' * 5 + 'Skipping {}'.format(sys_name))
            continue
        if analysis_type == 'best':
            tmp_tree = 'CERMAIN' if sys_name == 'miCERtang' else 'CER'
            tmp_cerfit = get_data('.{}.CERFIT:DATE_LOADED'.format(tmp_tree), verbose)
            if tmp_cerfit is not None:
                analysis_type = 'cerfit'
            else:
                tmp_cerauto = get_data('.{}.CERAUTO:DATE_LOADED'.format(tmp_tree), verbose)
                analysis_type = 'cerauto' if tmp_cerauto is not None else 'cerquick'
        print(' ' * 6 + '> Using: {} analysis for {}'.format(analysis_type, sys_name))

        # Settings for the various systems
        if sys_name == 'miCERtang':
            tree_name, orientation, id_letter = 'cermain', 'tangential', 'm'
            data_to_get = [
                ('TEMP', 'TEMP_ERR', 'T_u_{line_str}', 1, 'var', 'eV'),
                # Includes all corrections
                ('TEMPC', 'TEMPC_ERR', 'T_{line_str}', 1, 'var', 'eV'),
                # Includes all corrections
                ('ROTC', 'ROT_ERR', 'V_tor_{line_str}', 1000, 'var', 'm/s'),
                # Not corrected for atomic physics, ROT_ERR or ROTG_ERR?
                ('ROTG', 'ROT_ERR', 'V_tor_u_{line_str}', 1000, 'var', 'm/s'),
                # Line of sight velocity
                ('ROT', 'ROT_ERR', 'V_tor_uncor_{line_str}', 1000, 'var', 'm/s'),
                ('FULL', 'FULL_ERR', 'BE_full_{line_str}', 1.0, 'var', 'ph/m2/sr/s'),
                ('HALF', 'HALF_ERR', 'BE_half_{line_str}', 1.0, 'var', 'ph/m2/sr/s'),
                ('THIRD', 'THIRD_ERR', 'BE_third_{line_str}', 1.0, 'var', 'ph/m2/sr/s'),
                ('FIDA', 'FIDA_ERR', 'FIDA_{line_str}', 1.0, 'var', 'ph/m2/sr/s'),
                ('EBEAM', 'EBEAM_ERR', 'EBEAM_{line_str}', 1.0, 'var', 'V'),
                ('MODB', 'MODB_ERR', 'modB_{line_str}', 1.0, 'var', 'T'),
                # Hydrogen isotope fraction
                ('frac_1H1_act_therm', 'frac_1H1_act_therm_err', 'frac_1H1', 1.0, 'var', r'nH/(nD+nH)'),
            ]
            imp_ch_format = None
            measurements = ['T_{line_str}', 'T_u_{line_str}', 'BE_full_{line_str}', 'BE_half_{line_str}', 'BE_third_{line_str}', 'frac_1H1']
            fid_loc = '{}.channel{:02d}:fiducial'

        elif sys_name == 'impCERvert':
            tree_name, orientation, id_letter = 'cer', 'vertical', 'v'
            data_to_get = [
                ('TEMP', 'TEMP_ERR', 'T_{line_str}', 1, 'var', 'eV'),
                ('ROT', 'ROT_ERR', 'V_vert_{line_str}', 1000, 'var', 'm/s'),
            ]
            if fetch_VB:
                data_to_get.append(('VB', 'VB_ERR', VB_name, 1, 'var', 'photons/m^2/sr/s'))
            imp_ch_format = 'v{:d}'
            measurements = ['T_{line_str}', 'V_pol_{line_str}']  # ,VB_name]
            fid_loc = '{}.channel{:02d}:fiducual'

        elif sys_name == 'impCERtang':
            tree_name, orientation, id_letter = 'cer', 'tangential', 't'
            data_to_get = [
                ('TEMP', 'TEMP_ERR', 'T_{line_str}', 1, 'var', 'eV'),
                ('ROT', 'ROT_ERR', 'V_tor_uncor_{line_str}', 1000, 'var', 'm/s'),
                ('ROTC', 'ROT_ERR', 'V_tor_{line_str}', 1000, 'var', 'm/s'),
            ]
            if fetch_VB:
                data_to_get.append(('VB', 'VB_ERR', VB_name, 1, 'var', 'photons/m^2/sr/s'))
            imp_ch_format = 't{:d}'
            measurements = ['T_{line_str}']  # VB_name]
            fid_loc = '{}.channel{:02d}:fiducual'
        else:
            raise valueError('Unknown system name')

        # Get the valid channels from the CER_tree
        channel_num_list = [int(i.replace('CHANNEL', '')) for i in CER_tree[tree_name.upper()][analysis_type.upper()][orientation.upper()]]
        val_chords = np.sort(channel_num_list)

        # Specify where to get all of the data from
        loc = '.{}.{}.{}'.format(tree_name, analysis_type, orientation)
        cal_loc = '.{}.calibration.{}'.format(tree_name, orientation)
        lineid_loc = '{}.channel'.format(cal_loc) + '{:02d}:lineid'
        wavelength_loc = '{}.channel'.format(cal_loc) + '{:02}:WAVELENGTH'
        LOS_pt1_loc = ['{}.channel'.format(cal_loc) + '{:02}' + ':LENS_{}'.format(i) for i in ['R', 'Z', 'PHI']]

        beam_order_node = '{}.calibration:beam_order'.format(tree_name)
        beam_order = get_data(beam_order_node, verbose)
        beam_order = list(beam_order)
        if isinstance(beam_order[0], bytes):
            beam_order = [beam.decode('utf-8') for beam in beam_order]

        attrs_list = [
            ('beamgeometry', '{}.channel'.format(cal_loc) + '{:02}:BEAMGEOMETRY'),
            ('plasma_R', '{}.channel'.format(cal_loc) + '{:02}:PLASMA_R'),
            ('lineid', '{}.channel'.format(cal_loc) + '{:02}:LINEID'),
        ]
        #              ('wavelength', '{}.channel'.format(cal_loc) + '{:02}.WAVELENGTH'),]
        ch_format = id_letter + '{:02d}'

        # common data to get for all the systems
        common_data_to_get = [
            ('INTENSITY', 'INTENSITYERR', 'Int_{line_str}', 1, 'var', 'ph/m2/sr/s'),
            ('R', None, 'R', 1, 'var', 'm'),
            ('Z', None, 'Z', 1, 'var', 'm'),
            ('VIEW_PHI', None, 'phi', 1, 'var', 'deg'),
            ('STIME', None, 'stime', 1, 'coord', 'ms'),
            ('TTSUB', None, 'ttsub', 1, 'coord', ''),
            ('TTSUB_STIME', None, 'ttsub_stime', 1, 'coord', 'ms'),
        ]
        for i in common_data_to_get:
            data_to_get.append(i)

        # Fetch the measurements and impurity density (from IMPCON) for the current system
        # and put it in the OMFIT tree
        output_list, pinj_dict[sys_name] = fetch_system()
        RAW = {}

        # Whether or not to split up the CER system based on the beam the measurement came from
        output_ds_list = {}

        if split_CER_by_beam:
            for i in output_list:
                act_beam = i.active_beam.values
                possible_combs = np.unique(act_beam)
                for ii in possible_combs:
                    if ii not in output_ds_list:
                        output_ds_list[ii] = list()
                    mask = act_beam == ii
                    indices = np.arange(mask.shape[0])[mask]
                    new_ds = i.isel(time=indices)
                    new_ds = new_ds.assign_coords(channel=['{}_{}'.format(i.channel.values[0], ii)])
                    # Need to do this otherwise, the split systems share the same attributes dictionary
                    new_ds.attrs = copy.deepcopy(i.attrs)
                    output_ds_list[ii].append(new_ds)
        else:
            output_ds_list[sys_name] = output_list

        # Remove measurements with densities/temperatures < 0
        print(' ' * 6 + '> Removing invalid density and temperature values')
        for i in output_ds_list:
            for j in range(len(output_ds_list[i])):
                ds = output_ds_list[i][j]
                indices = [False] * len(ds['time'])
                for item in ds.variables:
                    if item.startswith('T_') or item.startswith('n_'):
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            invalid = np.isnan(nominal_values(ds.isel(channel=0)[item].values))
                            # For those values that are not zero, check they are also above zero
                            # Was throwing a numpy warning in regression test when there were Nan in this second equality
                            invalid[~invalid] = nominal_values(ds.isel(channel=0)[item].values)[~invalid] <= 0
                        printd('Converting %s values of %s<=0 in channel %s to nan ' % (sum(invalid), item, ds['channel'].values[0]))
                        ds.isel(channel=0)[item][invalid] = np.nan
                        indices |= ~invalid
                ds = ds.isel(time=indices)

        # Remove duplicate measurements (it occurs for miCER)
        for i in output_ds_list:
            for j in range(len(output_ds_list[i])):
                ds = output_ds_list[i][j]
                dummy, indices = np.unique(ds['time'].values, return_index=True)
                output_ds_list[i][j] = ds.isel(time=indices)

        # Apply the Zeeman correction
        if Zeeman_correction:
            # Load the neural network from pickle files
            if sys_name == 'miCERtang':
                fname = '{}/Zeeman_Corr_NN_2H1.pickle'.format(Zeeman_NN_loc)
                Zeeman_NN_species = '2H1'
            elif sys_name in ['impCERtang', 'impCERvert']:
                fname = '{}/Zeeman_Corr_NN_12C6.pickle'.format(Zeeman_NN_loc)
                # Technically this is for 12C6, n=8->7
                Zeeman_NN_species = '12C6'
            print("Loading Zeeman neural network correction for {} from:{}".format(Zeeman_NN_species, fname))
            with open(fname, 'rb') as filehandle:
                NN_dat = pickle.load(filehandle)
            Zeeman_NN = NN_dat['NN']
            Zeeman_NN_X_scaler = NN_dat['X_scaler']
            Zeeman_NN_Y_scaler = NN_dat['Y_scaler']
            printi("Loading magnetic field information")
            tmp = root['SCRIPTS']['DIII-D']['FETCH']['raw_EQ'].run(eq_name='EFIT01', store_in_tree=False)
            eq = tmp['ret_dat']['ds']
            eq = eq.sel(channel='EFIT01')
            OMFITprof_lib = root['LIB']['OMFITlib_general'].importCode()
            mag_field = OMFITprof_lib.mag_field_components(eq, make_plots=False)
            mag_field['time'] = eq['time'].values
            mag_field['R_vals'] = eq['R'].values
            mag_field['Z_vals'] = eq['Z'].values
            for tmp_subsys_name, tmp_subsys in output_ds_list.items():
                for tmp_cur_channel in tmp_subsys:
                    if 'T_{}'.format(Zeeman_NN_species) in tmp_cur_channel:
                        single_chord_zeeman_corr(
                            tmp_cur_channel, Zeeman_NN, Zeeman_NN_X_scaler, Zeeman_NN_Y_scaler, species=Zeeman_NN_species
                        )
                        if Zeeman_inc_uncorr:
                            tmp_cur_channel.attrs['measurements'].append('T_orig_{}'.format(Zeeman_NN_species))
                            tmp_cur_channel.attrs['measurements_to_fit'].append('T_orig_{}'.format(Zeeman_NN_species))
                    else:
                        printe("T_{} not available for this channel, cannot apply Zeeman correction".format(Zeeman_NN_species))
        if len(output_ds_list) >= 1:
            RAW.setdefault(sys_name, {})
            RAW[sys_name] = output_ds_list
        print(' ' * 5 + 'Finished {}, retrieved {} channels'.format(sys_name, len(output_list)))

    if return_xarray:
        return RAW

    # Mapping to OMAS

    def get_species(derived):
        """
        Identify species and ions that have density information
        """
        species = []
        for key in list(derived.data_vars.keys()):
            if not re.match('^[nT](_fast)?_([0-9]+[a-zA-Z]+[0-9]|e)$', key):
                continue
            s = key.split('_')[-1]
            if '_fast_' in key:
                s = 'fast_' + s
            species.append(s)
        species = np.atleast_1d(np.unique(species)).tolist()
        ions = [s for s in species if s not in ['e']]
        ions_with_dens = [i for i in ions if 'n_' + i in derived]
        ions_with_fast = [i.replace('fast_', '') for i in ions if 'fast_' in i]
        return species, ions, ions_with_dens, ions_with_fast

    def mZ(species):
        """
        Parse subscript strings and return ion mass and charge

        :param species: subscript strings such as `e`, `12C6`, `2H1`, 'fast_2H1`, ...

        :return: m and Z
        """
        species = str(species).replace('fast_', '')
        if species == 'e':
            Z = -1
            m = consts.m_e
        else:
            m = int(re.sub('([0-9]+)([a-zA-Z]+)([0-9]+)', r'\1', species))
            name = re.sub('([0-9]+)([a-zA-Z]+)([0-9]+)', r'\2', species)
            Z = int(re.sub('([0-9]+)([a-zA-Z]+)([0-9]+)', r'\3', species))
            m *= consts.m_u
        return m, Z

    cer_chan_ind = -1
    if ods is None:
        ods = ODS()
    sys = ods['charge_exchange']
    for k, diag in RAW.items():
        for sn, sub in diag.items():
            for ci, ch0 in enumerate(sub):
                ch = ch0.isel(channel=0)
                cer_chan_ind += 1
                chan = sys['channel.%d' % cer_chan_ind]
                chan['name'] = k + sn
                chan['identifier'] = str(ci)
                species, ions, ions_with_dens, ions_with_fast = get_species(ch)
                ion_spec = ions[0]
                m, z = mZ(ion_spec)
                chan['ion.0.a'] = m / consts.m_u
                chan['ion.0.label'] = ion_spec
                chan['position.r.data'] = ch['R'].values
                chan['position.r.time'] = ch['time'].values / 1e3
                chan['position.z.data'] = ch['Z'].values
                chan['position.z.time'] = ch['time'].values / 1e3
                if 'phi' in ch:  # Need to check this
                    chan['position.phi.data'] = ch['phi'].values / 360.0
                    chan['position.phi.time'] = ch['time'].values / 1e3
                if 'T_' + ion_spec in ch:
                    chan['ion.0.t_i.time'] = ch['time'].values / 1e3
                    chan['ion.0.t_i.data'] = ch['T_%s' % ion_spec].values
                if 'V_tor_' + ion_spec in ch:
                    chan['ion.0.velocity_tor.time'] = ch['time'].values / 1e3
                    chan['ion.0.velocity_tor.data'] = ch['V_tor_%s' % ion_spec].values
                if 'V_pol_' + ion_spec in ch:
                    chan['ion.0.velocity_pol.time'] = ch['time'].values / 1e3
                    chan['ion.0.velocity_pol.data'] = ch['V_pol_%s' % ion_spec].values
                chan['ion.0.z_n'] = z  # Charge of the nucleus
                chan['ion.0.z_ion'] = z  # Ionized state
    return ods


if __name__ == '__main__':
    d3d_cer(161409, systems=None, return_xarray=False)
    d3d_cer(161409, systems=None, return_xarray=True)
