# https://confluence.iter.org/display/IMP/UDA+data+mapping+tutorial

import numpy as np
import json
import re
from omas.omas_utils import o2u, l2o, p2l, i2o, u2n, printe
from omas import omas_environment, omas_info_node, imas_json_dir
from omas.omas_physics import cocos_signals
from omas.mappings.mds_mapping_functions import *

# ===================


with open(imas_json_dir + '/../mappings/cocos_rules.json', 'r') as f:
    cocos_rules = json.load(f)
# generate TDI for cocos_rules
for item in cocos_rules:
    if 'eval2TDI' in cocos_rules[item]:
        cocos_rules[item]['TDI'] = eval(cocos_rules[item]['eval2TDI'])

# ===================


def machine_to_omas(ods, machine, pulse, treename, location):
    '''
    Routine to convert machine data to ODS

    :param ods: input ODS to populate

    :param machine: machine

    :param pulse: pulse

    :param treename: MDS+ treename

    :param location: ODS location to be populated

    :return: updated ODS
    '''
    location = l2o(p2l(location))
    mapped = machine_mappings(machine)[location]

    # evaluate TDI
    if 'TDI' in mapped:
        try:
            TDI = mapped['TDI'].format(**locals())
            data = OMFITmdsValue(server=machine, shot=pulse, treename=treename, TDI=TDI).data()
            if data is None:
                raise ValueError('data is None')
        except Exception:
            printe(TDI.replace('\\n', '\n'))
            raise
    else:
        raise ValueError(f"Could not fetch data for {location}. Must define ['TDI']")

    # transpose manipulation
    if mapped.get('TRANSPOSE', False):
        data = np.transpose(data, mapped['TRANSPOSE'])

    # transpose filter
    nanfilter = lambda x: x
    if mapped.get('NANFILTER', False):
        nanfilter = lambda x: x[~np.isnan(x)]

    # cocos
    cocosio = 11
    if mapped.get('COCOSIO', False):
        cocosio = int(OMFITmdsValue(server=machine, shot=pulse, treename=treename, TDI=mapped['COCOSIO'].format(**locals())).data()[0])

    # assign data to ODS
    with omas_environment(ods, cocosio=cocosio):
        csize = mapped.get('COORDINATES', [])
        osize = len([c for c in mapped.get('COORDINATES', []) if c != '1...N'])
        dsize = len(data.shape)
        if dsize - osize == 0 or ':' not in location:
            if data.size == 1:
                data = np.asscalar(data)
            ods[location] = nanfilter(data)
        else:
            for k in range(data.shape[0]):
                ods[u2n(location, [k] + [0] * 10)] = nanfilter(data[k, ...])

    return ods


_machine_mappings = {}


def machine_mappings(machine):
    '''
    Function to load the json mapping files
    This function sanity-checks and the mapping file and adds extra info required for mapping

    :param machine: machine for which to load the mapping files

    :return: dictionary with mapping transformations
    '''
    if machine not in _machine_mappings:
        with open(imas_json_dir + '/../mappings/d3d.json', 'r') as f:
            _machine_mappings[machine] = json.load(f)
        mappings = _machine_mappings[machine]
        umappings = [o2u(loc) for loc in mappings]

        # generate TDI and sanity check mappings
        for location in mappings:
            # sanity check format
            if l2o(p2l(location)) != location:
                raise ValueError(f'{location} mapping should be specified as {l2o(p2l(location))}')

            # generate DTI functions based on eval2DTI
            if 'eval2TDI' in mappings[location]:
                mappings[location]['TDI'] = eval(mappings[location]['eval2TDI'])

            # make sure required coordinates info are present in the mapping
            info = omas_info_node(location)
            if 'coordinates' in info:
                mappings[location]['COORDINATES'] = list(map(i2o, info['coordinates']))
                for coordinate in mappings[location]['COORDINATES']:
                    if coordinate == '1...N':
                        continue
                    elif coordinate not in umappings:
                        raise ValueError(f'missing coordinate {coordinate} for {location}')

            # add cocos transformation info
            if o2u(location) in cocos_signals and cocos_signals[o2u(location)] is not None:
                cocos_defined = False
                for cocos in cocos_rules:
                    if re.findall(cocos, mappings[location]['TDI']):
                        mappings[location]['COCOSIO'] = cocos_rules[cocos]['TDI']
                        cocos_defined = True
                        break
                if not cocos_defined:
                    raise ValueError(f'{location} must have COCOS specified')

    return _machine_mappings[machine]


# ===================

if __name__ == '__main__':
    import os
    import tempfile

    os.chdir(tempfile.gettempdir())
    from omas import ODS
    from omfit.classes.omfit_mds import OMFITmdsValue
    from omfit.classes.omfit_eqdsk import OMFITgeqdsk

    machine = 'd3d'
    pulse = 168830
    treename = 'EFIT01'

    ods = ODS()
    for location in machine_mappings(machine):
        print(location)
        machine_to_omas(ods, machine, pulse, treename, location)

    g = OMFITgeqdsk(None).from_omas(ods, 100)
    g.plot()
    from matplotlib import pyplot

    pyplot.show()
