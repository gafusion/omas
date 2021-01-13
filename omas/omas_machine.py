# https://confluence.iter.org/display/IMP/UDA+data+mapping+tutorial

import numpy as np
import json
import re
from omas.omas_utils import o2u, l2o, p2l, i2o, u2n, printe, printd, convert_int
from omas import ODS, omas_environment, omas_info_node, imas_json_dir, omas_rcparams
from omas.omas_core import dynamic_ODS
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
    from omfit.classes.omfit_mds import OMFITmdsValue

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

    if location.endswith(':'):
        return int(data[0])

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


def load_omas_machine(machine, pulse, consistency_check=True, imas_version=omas_rcparams['default_imas_version'], cls=ODS):
    printd('Loading from %s' % machine, topic='machine')
    ods = cls(imas_version=imas_version, consistency_check=consistency_check)
    treename = 'EFIT01'
    for location in machine_mappings(machine):
        if location.endswith(':'):
            continue
        print(location)
        machine_to_omas(ods, machine, pulse, treename, location)
    return ods


class dynamic_omas_machine(dynamic_ODS):
    """
    Class that provides dynamic data loading from machine mappings
    This class is not to be used by itself, but via the ODS.open() method.
    """

    def __init__(self, machine, pulse, verbose=True):
        self.kw = {'machine': machine, 'pulse': pulse}
        self.active = False
        self.cache = {}

    def open(self):
        printd('Dynamic open  %s' % self.kw, topic='dynamic')
        self.active = True
        return self

    def close(self):
        printd('Dynamic close %s' % self.kw, topic='dynamic')
        self.active = False
        self.cache.clear()
        return self

    def __getitem__(self, key):
        if not self.active:
            raise RuntimeError('Dynamic link broken: %s' % self.kw)
        if o2u(key) not in self.cache:
            printd('Dynamic read  %s: %s' % (self.kw, key), topic='dynamic')
            out = machine_to_omas(ODS(), self.kw['machine'], self.kw['pulse'], 'EFIT01', o2u(key))
            self.cache[o2u(key)] = out
        if isinstance(self.cache[o2u(key)], int):
            return self.cache[o2u(key)]
        else:
            return self.cache[o2u(key)][key]

    def __contains__(self, location):
        ulocation = o2u(location)
        if not self.active:
            raise RuntimeError('Dynamic link broken: %s' % self.kw)
        if ulocation.endswith(':'):
            return False
        return ulocation in machine_mappings(self.kw['machine'])

    def keys(self, location):
        ulocation = o2u(location)
        if ulocation + '.:' in machine_mappings(self.kw['machine']):
            return list(range(self[ulocation + '.:']))
        else:
            return np.unique(
                [
                    convert_int(k[len(ulocation) :].lstrip('.').split('.')[0])
                    for k in machine_mappings(self.kw['machine'])
                    if k.startswith(ulocation)
                ]
            )


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
