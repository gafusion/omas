import numpy as np
from inspect import unwrap
from omas import *
from omas.omas_utils import printd
from omas.machine_mappings._common import *

__all__ = []


@machine_mapping_function(__all__)
def pf_active_hardware(ods):
    r"""
    Loads NSTX-U tokamak poloidal field coil hardware geometry

    :param ods: ODS instance
    """
    # From `OMFITmhdin(OMFITsrc + '/../modules/EFUND/TEMPLATES/mhdin_nstxu.dat').pretty_print()`
    # R        Z       dR      dZ    tilt1  tilt2
    # 0 in the last column really means 90 degrees
    # fmt: off
    fc_dat = np.array(
        [[0.324600011, 1.59060001, 0.0625, 0.463400006, 0.0, 0.0],
         [0.400299996, 1.80420005, 0.0337999985, 0.181400001, 0.0, 0.0],
         [0.550400019, 1.81560004, 0.0375000015, 0.1664, 0.0, 0.0],
         [0.799170017, 1.85264003, 0.162711993, 0.0679700002, 0.0, 0.0],
         [0.799170017, 1.93350995, 0.162711993, 0.0679700002, 0.0, 0.0],
         [1.49445999, 1.55263996, 0.186435997, 0.0679700002, 0.0, 0.0],
         [1.49445999, 1.63350999, 0.186435997, 0.0679700002, 0.0, 0.0],
         [1.80649996, 0.888100028, 0.115264997, 0.0679700002, 0.0, 0.0],
         [1.79460001, 0.807200015, 0.0915419981, 0.0679700002, 0.0, 0.0],
         [2.01180005, 0.648899972, 0.135900006, 0.0684999973, 0.0, 0.0],
         [2.01180005, 0.575100005, 0.135900006, 0.0684999973, 0.0, 0.0],
         [2.01180005, -0.648899972, 0.135900006, 0.0684999973, 0.0, 0.0],
         [2.01180005, -0.575100005, 0.135900006, 0.0684999973, 0.0, 0.0],
         [1.80649996, -0.888100028, 0.115264997, 0.0679700002, 0.0, 0.0],
         [1.79460001, -0.807200015, 0.0915419981, 0.0679700002, 0.0, 0.0],
         [1.49445999, -1.55263996, 0.186435997, 0.0679700002, 0.0, 0.0],
         [1.49445999, -1.63350999, 0.186435997, 0.0679700002, 0.0, 0.0],
         [0.799170017, -1.85264003, 0.162711993, 0.0679700002, 0.0, 0.0],
         [0.799170017, -1.93350995, 0.162711993, 0.0679700002, 0.0, 0.0],
         [0.550400019, -1.82959998, 0.0375000015, 0.1664, 0.0, 0.0],
         [0.400299996, -1.80420005, 0.0337999985, 0.181400001, 0.0, 0.0],
         [0.324600011, -1.59060001, 0.0625, 0.463400006, 0.0, 0.0]]
    )

    names = {'PF1AU':'PF1AU', 'PF1BU':'PF1BU', 'PF1CU':'PF1CU',
             'PF2U1':'PF2U', 'PF2U2':'PF2U',
             'PF3U1':'PF3U', 'PF3U2':'PF3U',
             'PF4U1':'PF4', 'PF4U2':'PF4',
             'PF5U1':'PF5', 'PF5U2':'PF5',
             'PF5L1':'PF5', 'PF5L2':'PF5',
             'PF4L1':'PF4', 'PF4L2':'PF4',
             'PF3L1':'PF3L', 'PF3L2':'PF3L',
             'PF2L1':'PF2L', 'PF2L2':'PF2L',
             'PF1CL':'PF1CL', 'PF1BL':'PF1BL', 'PF1AL':'PF1AL'}
    # fmt: on

    ods = pf_coils_to_ods(ods, fc_dat)

    for i, (name, fcid) in enumerate(names.items()):
        ods['pf_active.coil'][i]['name'] = ods['pf_active.coil'][i]['identifier'] = name
        ods['pf_active.coil'][i]['element.0.identifier'] = fcid


@machine_mapping_function(__all__)
def pf_active_coil_current_data(ods, pulse=203679):
    ods1 = ODS()
    unwrap(pf_active_hardware)(ods1)
    with omas_environment(ods, cocosio=1):
        fetch_assign(
            ods,
            ods1,
            pulse,
            channels='pf_active.coil',
            identifier='pf_active.coil.{channel}.element.0.identifier',
            time='pf_active.coil.{channel}.current.time',
            data='pf_active.coil.{channel}.current.data',
            validity=None,
            mds_server='nstxu',
            mds_tree='ENGINEERING',
            tdi_expression='\\ENGINEERING::TOP.ANALYSIS.I{signal}',
            time_norm=1.0,
            data_norm=1.0,
        )


@machine_mapping_function(__all__)
def magnetics_hardware(ods):
    r"""
    Load NSTX-U tokamak flux loops and magnetic probes hardware geometry

    :param ods: ODS instance
    """
    # From: `OMFITmhdin(OMFITsrc + '/../modules/EFUND/TEMPLATES/mhdin_nstxu.dat').pretty_print()`
    # From: `OMFITnstxMHD('/p/spitfire/s1/common/Greens/NSTX/Jan2015/01152015Av1.0/diagSpec01152015.dat').pretty_print()`
    # fmt: off
    # ==========
    # Flux loops
    # ==========
    R_flux_loop = [0.63270003, 1.00399995, 1.32000005, 1.72599995, 1.72590005,
                   1.71340001, 0.63730001, 0.97780001, 1.33570004, 1.72520006,
                   1.72490001, 1.71340001, 1.67719996, 1.67719996, 1.67719996,
                   1.67719996, 0.65170002, 0.8405, 1.00049996, 0.85860002,
                   0.64810002, 1.00329995, 0.28, 0.28, 0.28,
                   0.28, 0.28, 0.28, 0.28, 0.28,
                   0.28, 0.36000001, 0.36000001, 0.36000001, 0.36000001,
                   0.36000001, 0.36000001, 0.36000001, 0.36000001, 1.50740004,
                   1.47430003, 1.43480003, 1.40989995, 1.51139998, 1.47549999,
                   1.44560003, 1.40970004, 1.34590006, 1.29750001, 1.24969995,
                   1.19809997, 1.34640002, 1.29690003, 1.24759996, 1.19519997,
                   0.55000001, 0.34999999, 0.41999999, 0.41999999, 0.47999999,
                   0.47999999, 0.41999999, 0.41999999, 0.34999999, 0.55000001,
                   0.28]

    Z_flux_loop = [-1.73959994, -1.61829996, -1.44099998, -1.15489995, -0.81279999,
                   -0.36090002, 1.74000001, 1.648, 1.45140004, 1.11600006,
                   0.78990001, 0.3436, -0.62309998, -1.04920006, 0.60829997,
                   1.03550005, -1.70239997, -1.65279996, -1.58790004, 1.65090001,
                   1.70580006, 1.58840001, -0.25, -0.5, -0.75,
                   -1., 0., 0.25, 0.5, 0.75,
                   1., -1.39999998, -1.5, -1.70000005, -1.79999995,
                   1.39999998, 1.5, 1.70000005, 1.79999995, -0.66289997,
                   -0.76190001, -0.82690001, -0.9533, 0.65079999, 0.74980003,
                   0.83679998, 0.9382, -1.09379995, -1.15789998, -1.22449994,
                   -1.28859997, 1.08360004, 1.15059996, 1.21609998, 1.28460002,
                   1.95000005, 1.94000006, 1.85000002, 1.75, 1.70500004,
                   -1.70500004, -1.75, -1.85000002, -1.95500004, -1.96500003,
                   0.]

    name_flux_loop = ['FLEVVL2', 'FLEVVL3', 'FLEVVL4', 'FLEVVL5', 'FLEVVL6', 'FLEVVL7', 'FLEVVU2', 'FLEVVU3', 'FLEVVU4',
                      'FLEVVU5', 'FLEVVU6', 'FLEVVU7', 'FLIVVL1', 'FLIVVL2', 'FLIVVU1', 'FLIVVU2', 'FLOBDL1', 'FLOBDL2',
                      'FLOBDL3', 'FLOBDU1', 'FLOBDU2', 'FLOBDU3', 'FLOHL1', 'FLOHL2', 'FLOHL3', 'FLOHL4', 'FLOHM',
                      'FLOHU1', 'FLOHU2', 'FLOHU3', 'FLOHU4', 'FLPF1AL1', 'FLPF1AL2', 'FLPF1AL3', 'FLPF1AL4', 'FLPF1AU1',
                      'FLPF1AU2', 'FLPF1AU3', 'FLPF1AU4', 'FLPPPL1', 'FLPPPL2', 'FLPPPL3', 'FLPPPL4', 'FLPPPU1', 'FLPPPU2',
                      'FLPPPU3', 'FLPPPU4', 'FLSPPL1', 'FLSPPL2', 'FLSPPL3', 'FLSPPL4', 'FLSPPU1', 'FLSPPU2', 'FLSPPU3',
                      'FLSPPU4', 'FLMDLU2', 'FLMDLU1', 'FLPF1BU2', 'FLPF1BU1', 'FLCSCU4', 'FLCSCL4', 'FLBF1BL1', 'FLBF1BL2',
                      'FLMDLL1', 'FLMDLL2', 'FLEXTRA']
    # ===============
    # Magnetic probes
    # ===============
    R_magnetic = [0.30199999, 0.30199999, 0.30199999, 0.30199999, 0.30199999,
                  0.30199999, 0.30199999, 0.30199999, 0.30199999, 0.30199999,
                  0.30199999, 0.30199999, 0.30199999, 0.30199999, 0.30199999,
                  0.30199999, 0.30199999, 0.30199999, 0.30199999, 0.30199999,
                  0.30199999, 0.30199999, 0.30199999, 0.30199999, 0.30199999,
                  0.30199999, 0.30199999, 0.30199999, 0.30199999, 0.30199999,
                  1.48860002, 1.46099997, 1.42910004, 1.49109995, 1.4619,
                  1.42879999, 1.30540001, 1.25629997, 1.20720005, 1.32050002,
                  1.26779997, 1.21809995, 1.39709997, 1.42490005, 1.45720005,
                  1.48959994, 1.51740003, 1.39709997, 1.42490005, 1.45720005,
                  1.48959994, 1.51740003, 1.18340003, 1.22280002, 1.27139997,
                  1.32959998, 1.18340003, 1.22280002, 1.27139997, 1.32959998,
                  0.47279999, 0.47279999, 0.47279999, 0.47279999, 0.47279999,
                  0.47279999, 0.47279999, 0.47279999, 0.39399999, 0.39399999,
                  0.39399999, 0.39399999, 0.39399999, 0.39399999, 0.39399999,
                  0.39399999, 0.39399999, 0.39399999, 0.39399999, 0.39399999,
                  0.39399999, 0.39399999, 0.39399999, 0.39399999, 0.68159998,
                  0.68159998, 0.9055, 0.9055, 1.13660002, 1.13660002,
                  0.68159998, 0.68159998, 0.9055, 0.9055, 1.13660002,
                  1.13660002, 0.79000002, 0.79000002, 1.02110004, 1.02110004,
                  0.79000002, 0.79000002, 1.02110004, 1.02110004, 0.47279999,
                  0.47279999, 0.47279999, 0.47279999]

    Z_magnetic = [-0.033, -0.13600001, -0.27200001, -0.40799999, -0.54400003,
                  -0.68000001, 0.13600001, 0.27200001, 0.40799999, 0.54400003,
                  0.68000001, -0.68000001, -0.40799999, -0.27200001, -0.033,
                  0.13600001, 0.40799999, 0.68000001, -0.95300001, -0.95300001,
                  0.81699997, 0.81699997, -0.81699997, -0.81699997, 0.95300001,
                  0.95300001, -0.95300001, -0.95300001, 0.95200002, 0.95200002,
                  -0.71109998, -0.79619998, -0.89450002, 0.69859999, 0.78549999,
                  0.88410002, -1.15240002, -1.22529995, -1.29550004, 1.13600004,
                  1.20930004, 1.27859998, -0.96899998, -0.88459998, -0.78659999,
                  -0.6886, -0.60420001, 0.96899998, 0.88459998, 0.78659999,
                  0.6886, 0.60420001, -1.31700003, -1.26119995, -1.1925,
                  -1.11020005, 1.31700003, 1.26119995, 1.1925, 1.11020005,
                  -1.64900005, -1.64900005, 1.64900005, 1.64900005, 1.64900005,
                  1.64900005, -1.64900005, -1.64900005, 1.36399996, 1.36399996,
                  1.46200001, 1.46200001, 1.55999994, 1.55999994, -1.36399996,
                  -1.36399996, -1.46200001, -1.46200001, -1.55999994, -1.55999994,
                  1.46200001, 1.46200001, -1.46200001, -1.46200001, -1.61530006,
                  -1.61530006, -1.52709997, -1.52709997, -1.43610001, -1.43610001,
                  1.61530006, 1.61530006, 1.52709997, 1.52709997, 1.43610001,
                  1.43610001, -1.57260001, -1.57260001, -1.48160005, -1.48160005,
                  1.57260001, 1.57260001, 1.48160005, 1.48160005, 1.64900005,
                  1.64900005, -1.64900005, -1.64900005]

    A_magnetic = [90., 90., 90., 90., 90.,
                  90., 90., 90., 90., 90.,
                  90., 90., 90., 90., 90.,
                  90., 90., 90., 0., 90.,
                  0., 90., 90., 0., 90.,
                  0., 90., 0., 90., 0.,
                  71.6707993, 66.3263016, 70.4561996, 107.652, 113.017998,
                  107.425003, 57.6887016, 53.9482994, 54.1119995, 126.530998,
                  126.248001, 124.803001, 71.5, 74.0999985, 70.,
                  69.2235031, 71.8610001, 109.301003, 108.685997, 105.824997,
                  106.981003, 107.504997, 52.9403, 55.6543999, 54.1320992,
                  57.5690994, 126.985001, 127.282997, 128.231995, 126.612999,
                  0., 90., 0., 90., 0.,
                  90., 0., 90., 90., 0.,
                  90., 0., 90., 0., 90.,
                  0., 90., 0., 90., 0.,
                  90., 0., 90., 0., 19.2000008,
                  116., 26.4097996, 110.654999, 17.1231995, 111.585999,
                  -22.0321007, 66.8696976, -21.3855991, 69.2611008, -21.3295002,
                  68.0830002, 21.5, 111.5, 21.5, 111.5,
                  -21.5, 68.5, -21.5, 68.5, 0.,
                  90., 0., 90.]

    S_magnetic = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                  0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                  0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                  0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                  0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                  0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                  0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                  0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                  0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                  0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]

    name_magnetic = ['1DMCSCL1', '1DMCSCL2', '1DMCSCL3', '1DMCSCL4', '1DMCSCL5', '1DMCSCL6', '1DMCSCU2',
                     '1DMCSCU3', '1DMCSCU4', '1DMCSCU5', '1DMCSCU6', '1DMCSC2L6', '1DMCSC2L4', '1DMCSC2L2',
                     '1DMCSC2L1', '1DMCSC2U2', '1DMCSC2U4', '1DMCSC2U6', '2DMCSCL2N', '2DMCSCL2T', '2DMCSCU1N',
                     '2DMCSCU1T', '2DMCSCL1T', '2DMCSCL1N', '2DMCSCU2T', '2DMCSCU2N', '2DMCSC2L2T',
                     '2DMCSC2L2N', '2DMCSC2U2T', '2DMCSC2U2N', 'L1DMPPPGL1', 'L1DMPPPGL2', 'L1DMPPPGL3',
                     'L1DMPPPGU1', 'L1DMPPPGU2', 'L1DMPPPGU3', 'L1DMSPPGL1', 'L1DMSPPGL2', 'L1DMSPPGL3',
                     'L1DMSPPGU1', 'L1DMSPPGU2', 'L1DMSPPGU3', 'L1DMPPPGL4', 'L1DMPPPGL5', 'L1DMPPPGL6',
                     'L1DMPPPGL7', 'L1DMPPPGL8', 'L1DMPPPGU4', 'L1DMPPPGU5', 'L1DMPPPGU6', 'L1DMPPPGU7',
                     'L1DMPPPGU8', 'L1DMSPPGL4', 'L1DMSPPGL5', 'L1DMSPPGL6', 'L1DMSPPGL7', 'L1DMSPPGU4',
                     'L1DMSPPGU5', 'L1DMSPPGU6', 'L1DMSPPGU7', '2DMIBDHL6T', '2DMIBDHL6N', '2DMIBDHU5T',
                     '2DMIBDHU5N', '2DMIBDHU6T', '2DMIBDHU6N', '2DMIBDHL5T', '2DMIBDHL5N', '2DMIBDVU1T',
                     '2DMIBDVU1N', '2DMIBDVU2T', '2DMIBDVU2N', '2DMIBDVU3T', '2DMIBDVU3N', '2DMIBDVL1T',
                     '2DMIBDVL1N', '2DMIBDVL2T', '2DMIBDVL2N', '2DMIBDVL3T', '2DMIBDVL3N', '2DMIBDV2U2T',
                     '2DMIBDV2U2N', '2DMIBDV2L2T', '2DMIBDV2L2N', '2DMOBDL1T', '2DMOBDL1N', '2DMOBDL3T',
                     '2DMOBDL3N', '2DMOBDL5T', '2DMOBDL5N', '2DMOBDU1T', '2DMOBDU1N', '2DMOBDU3T',
                     '2DMOBDU3N', '2DMOBDU5T', '2DMOBDU5N', '2DMOBDL2T', '2DMOBDL2N', '2DMOBDL4T',
                     '2DMOBDL4N', '2DMOBDU2T', '2DMOBDU2N', '2DMOBDU4T', '2DMOBDU4N', '2DMIBDH2U6T',
                     '2DMIBDH2U6N', '2DMIBDH2L6T', '2DMIBDH2L6N']
    # fmt: on

    with omas_environment(ods, cocosio=1):
        for k, (r, z, name) in enumerate(zip(R_flux_loop, Z_flux_loop, name_flux_loop)):
            ods[f'magnetics.flux_loop.{k}.name'] = name
            ods[f'magnetics.flux_loop.{k}.identifier'] = 'F_' + name
            ods[f'magnetics.flux_loop.{k}.position[0].r'] = r
            ods[f'magnetics.flux_loop.{k}.position[0].z'] = z
            ods[f'magnetics.flux_loop.{k}.type.index'] = 1

        for k, (r, z, a, s, name) in enumerate(zip(R_magnetic, Z_magnetic, A_magnetic, S_magnetic, name_magnetic)):
            ods[f'magnetics.b_field_pol_probe.{k}.name'] = name
            ods[f'magnetics.b_field_pol_probe.{k}.identifier'] = 'B_' + name
            ods[f'magnetics.b_field_pol_probe.{k}.position.r'] = r
            ods[f'magnetics.b_field_pol_probe.{k}.position.z'] = z
            ods[f'magnetics.b_field_pol_probe.{k}.length'] = s
            ods[f'magnetics.b_field_pol_probe.{k}.poloidal_angle'] = -a / 180 * np.pi
            ods[f'magnetics.b_field_pol_probe.{k}.toroidal_angle'] = 0.0 / 180 * np.pi
            ods[f'magnetics.b_field_pol_probe.{k}.type.index'] = 1
            ods[f'magnetics.b_field_pol_probe.{k}.turns'] = 1


@machine_mapping_function(__all__)
def magnetics_floops_data(ods, pulse=203679):
    ods1 = ODS()
    unwrap(magnetics_hardware)(ods1)
    with omas_environment(ods, cocosio=1):
        fetch_assign(
            ods,
            ods1,
            pulse,
            channels='magnetics.flux_loop',
            identifier='magnetics.flux_loop.{channel}.identifier',
            time='magnetics.flux_loop.{channel}.flux.time',
            data='magnetics.flux_loop.{channel}.flux.data',
            validity='magnetics.flux_loop.{channel}.flux.validity',
            mds_server='nstxu',
            mds_tree='NSTX',
            tdi_expression='\{signal}',
            time_norm=0.001,
            data_norm=1.0,
        )


@machine_mapping_function(__all__)
def magnetics_probes_data(ods, pulse=203679):
    ods1 = ODS()
    unwrap(magnetics_hardware)(ods1)
    with omas_environment(ods, cocosio=1):
        fetch_assign(
            ods,
            ods1,
            pulse,
            channels='magnetics.b_field_pol_probe',
            identifier='magnetics.b_field_pol_probe.{channel}.identifier',
            time='magnetics.b_field_pol_probe.{channel}.field.time',
            data='magnetics.b_field_pol_probe.{channel}.field.data',
            validity='magnetics.b_field_pol_probe.{channel}.field.validity',
            mds_server='nstxu',
            mds_tree='NSTX',
            tdi_expression='\{signal}',
            time_norm=0.001,
            data_norm=1.0,
        )


@machine_mapping_function(__all__)
def MDS_gEQDSK_psi_nstx(ods, pulse=203679, EFIT_tree='EFIT01'):
    return MDS_gEQDSK_psi(ods, 'nstxu', pulse, EFIT_tree)


@machine_mapping_function(__all__)
def MDS_gEQDSK_bbbs_nstx(ods, pulse=203679, EFIT_tree='EFIT01'):
    TDIs = {
        'r': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.RBBBS',
        'z': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.ZBBBS',
        'n': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.NBBBS',
    }
    res = mdsvalue('nstxu', pulse=pulse, treename=EFIT_tree, TDI=TDIs).raw()
    res['n'] = res['n'].astype(int)
    for k in range(len(res['n'])):
        ods[f'equilibrium.time_slice.{k}.boundary.outline.r'] = res['r'][k, : res['n'][k]]
        ods[f'equilibrium.time_slice.{k}.boundary.outline.z'] = res['z'][k, : res['n'][k]]



# =====================
if __name__ == '__main__':
    run_machine_mapping_functions(__all__, globals(), locals())
