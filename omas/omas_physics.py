from __future__ import print_function, division, unicode_literals

import numpy

__all__=[]

def add_to__all__(f):
    __all__.append(f.__name__)
    return f

#constants class that mimics scipy.constants
class constants(object):
    e=1.6021766208e-19

@add_to__all__
def core_profiles_update_pressures(ods):
    '''
    updates pressures from kinetic profiles in place

        `core_profiles.profiles_1d.:.pressure_thermal`
        `core_profiles.profiles_1d.:.pressure_ion_total`
        `core_profiles.profiles_1d.:.pressure_perpendicular`
        `core_profiles.profiles_1d.:.pressure_parallel`

    :param ods: input ods

    :return: updated ods
    '''
    for time_index in ods['core_profiles']['profiles_1d']:
        prof1d=ods['core_profiles']['profiles_1d'][time_index]
        __p__=prof1d['electrons']['density']*prof1d['electrons']['temperature']*constants.e
        prof1d['pressure_thermal']=__p__
        prof1d['pressure_ion_total']=__p__*0.0
        prof1d['pressure_perpendicular']=__p__/3.
        prof1d['pressure_parallel']=__p__/3.
        for therm_fast,density in [('therm','density'),('fast','density_fast')]:
            for k in range(len(prof1d['ion'])):
                if (len(prof1d['ion']) and density in prof1d['ion'][k] and numpy.sum(numpy.abs(prof1d['ion'][k][density])) > 0):
                    if therm_fast == 'therm':
                        __p__=prof1d['ion'][k]['density']*prof1d['ion'][k]['temperature']*constants.e
                        prof1d['pressure_ion_total']+=__p__
                        prof1d['pressure_perpendicular']+=__p__/3.
                        prof1d['pressure_parallel']+=__p__/3.
                    else:
                        prof1d['pressure_perpendicular']+=prof1d['ion'][k]['pressure_fast_perpendicular']
                        prof1d['pressure_parallel']+=prof1d['ion'][k]['pressure_fast_parallel']
        prof1d['pressure_thermal']+=prof1d['pressure_ion_total']
    return ods
