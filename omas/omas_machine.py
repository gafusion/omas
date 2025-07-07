# https://confluence.iter.org/display/IMP/UDA+data+mapping+tutorial

import subprocess
import functools
import shutil
from omas.omas_utils import *
from omas.omas_core import ODS, dynamic_ODS, omas_environment, omas_info_node, imas_json_dir, omas_rcparams
from omas.omas_physics import cocos_signals
from omas.machine_mappings import d3d, nstx, nstxu, east
from omas.machine_mappings.d3d import __regression_arguments__
from omas.utilities.machine_mapping_decorator import machine_mapping_function
from omas.utilities.omas_mds import mdsvalue, check_for_pulse_id
try:
    from MDSplus.connection import MdsIpException
    from MDSplus.mdsExceptions import TreeNODATA, TreeNNF
except:
    pass

try:
    from omas.machine_mappings import mast
except ImportError:
    pass

__all__ = [
    'machine_expression_types',
    'machines',
    'machine_mappings',
    'load_omas_machine',
    'test_machine_mapping_functions',
    'reload_machine_mappings'
]

machine_expression_types = ['VALUE', 'EVAL', 'ENVIRON', 'PYTHON', 'TDI', 'eval2TDI']

_url_dir = os.sep.join([omas_rcparams['tmp_omas_dir'], 'machine_mappings', '{branch}', 'omas_machine_mappings_url_{branch}'])


# ===================
# mapping engine
# ===================


def python_tdi_namespace(branch):
    """
    Returns the namespace of the python_tdi.py file

    :param branch: remote branch to load

    :return: namespace
    """
    # return cached python tdi function namespace
    if branch in _python_tdi_namespace:
        return _python_tdi_namespace[branch]
    _python_tdi_namespace[branch] = {}

    # get local mapping functions
    if not branch:
        exec('from omas.machine_mappings.python_tdi import *', _python_tdi_namespace[branch])

    # get mapping functions from GitHub
    else:
        printd(f'omas python tdi mappings from branch: `{branch}`', topic='machine')

        # make sure remote branch is transfered
        machines(None, branch)

        # import from temporary directory
        dir = _url_dir.format(branch=branch)
        if dir + os.sep + '..' not in sys.path:
            sys.path.insert(0, dir + os.sep + '..')

        exec(f'from omas_machine_mappings_url_{branch}.python_tdi import *', _python_tdi_namespace[branch])

    return _python_tdi_namespace[branch]

def remove_nans(x):
    import numpy as np
    if np.isscalar(x):
        if np.isnan(x):
            raise ValueError("Behavior of Nan filter undefined for scalar nan values")
        else:
            return x
    else:
        return x[~np.isnan(x)]

def machine_to_omas(ods, machine, pulse, location, options={}, branch='', user_machine_mappings=None, cache=None):
    """
    Routine to convert machine data to ODS

    :param ods: input ODS to populate

    :param machine: machine name

    :param pulse: pulse number

    :param location: ODS location to be populated

    :param options: dictionary with options to use when loading the data

    :param branch: load machine mappings and mapping functions from a specific GitHub branch

    :param user_mappings: allow specification of external mappings

    :param cache: if cache is a dictionary, this will be used to establish a cash

    :return: updated ODS and data before being assigned to the ODS
    """

    pulse = int(pulse)

    if user_machine_mappings is None:
        user_machine_mappings = {}

    location = l2o(p2l(location))
    for branch in [branch, 'master']:
        mappings = machine_mappings(machine, branch, user_machine_mappings)
        options_with_defaults = copy.copy(mappings['__options__'])
        options_with_defaults.update(options)
        options_with_defaults.update({'machine': machine, 'pulse': pulse, 'location': location})
        try:
            if not location.endswith(".*"): # location = "core_profiles.*"
                mapped = mappings[location]
            break
        except KeyError as e:
            if branch == 'master':
                raise e
            else:
                print(f"Failed to load {location} from head. Attempting to resolve using the master branch.")
                print(f"Error was:")
                print(e)
    idm = (machine, branch)
    failed_locations = {}
    if location.endswith(".*"):
        temp_cache = cache
        if temp_cache is None:
            temp_cache = {}
        root = location.split(".*")[0]
        for key in mappings:
            if root in key and key not in ods:
                try:
                    resolve_mapped(ods, machine, pulse, mappings, key, idm, options_with_defaults, branch, cache=temp_cache)
                except (TreeNODATA, MdsIpException) as e:
                    if hasattr(e, "eval2TDI"):
                        failed_locations[key] = e.eval2TDI
                    else:
                        failed_locations[key] = e.TDI
                except TreeNNF as e:
                    failed_locations[key] = e.TDI
                    if key != 'equilibrium.time_slice.:.constraints.j_tor.:.measured':
                        raise e
        if len(failed_locations) > 0:
            import yaml
            print("Failed to load the following keys: ")
            print(failed_locations)
            with open("failed_locs", "w") as failed_locs_file:
                yaml.dump(failed_locations, failed_locs_file, yaml.CDumper)
            return ods
    else:
        return resolve_mapped(ods, machine, pulse,  mappings, location, idm, options_with_defaults, branch, cache=cache)

def resolve_mapped(ods, machine, pulse,  mappings, location, idm, options_with_defaults, branch, cache=None):
    """
    Routine to resolve a mapping

    :param ods: input ODS to populate

    :param machine: machine name

    :param pulse: pulse number

    :param mappings: Dictionary of available mappings

    :param location: ODS location to be resolved

    :param idm: Tuple with machine and branch

    :param options_with_defaults: dictionary with options to use when loading the data including default settings

    :param branch: load machine mappings and mapping functions from a specific GitHub branch

    :param cache: if cache is a dictionary, this will be used to establish a cash

    :return: updated ODS and data before being assigned to the ODS
    """
    mapped = mappings[location]
    # cocosio
    cocosio = None
    if 'COCOSIO' in mapped:
        if isinstance(mapped['COCOSIO'], int):
            cocosio = mapped['COCOSIO']
    elif 'COCOSIO_PYTHON' in mapped:
        call = mapped['COCOSIO_PYTHON'].format(**options_with_defaults)
        if cache and call in cache:
            cocosio = cache[call]
        else:
            namespace = {}
            namespace.update(_namespace_mappings[idm])
            namespace['__file__'] = machines(machine, branch)[:-5] + '.py'
            tmp = compile(call, machines(machine, branch)[:-5] + '.py', 'eval')
            cocosio = eval(tmp, namespace)
            if isinstance(cache, dict):
                cache[call] = cocosio
    elif 'COCOSIO_TDI' in mapped:
        TDI = mapped['COCOSIO_TDI'].format(**options_with_defaults)
        treename = mapped['treename'].format(**options_with_defaults) if 'treename' in mapped else None
        cocosio = int(mdsvalue(machine, treename, pulse, TDI).raw())

    # CONSTANT VALUE
    if 'VALUE' in mapped:
        data0 = data = mapped['VALUE']

    # EVAL
    elif 'EVAL' in mapped:
        data0 = data = eval(mapped['EVAL'].format(**options_with_defaults), _namespace_mappings[idm])

    # ENVIRONMENTAL VARIABLE
    elif 'ENVIRON' in mapped:
        data0 = data = os.environ.get(mapped['ENVIRON'].format(**options_with_defaults))
        if data is None:
            raise ValueError(
                f'Environmental variable {mapped["ENVIRON"].format(**options_with_defaults)} is not defined'
            )

    # PYTHON
    elif 'PYTHON' in mapped:
        call = mapped['PYTHON'].format(**options_with_defaults)
        # python functions tend to set multiple locations at once
        # it is thus very beneficial to cache that
        if cache and call in cache:
            ods = cache[call]
        else:
            namespace = {}
            namespace.update(_namespace_mappings[idm])
            namespace['ods'] = ODS()
            namespace['__file__'] = machines(machine, branch)[:-5] + '.py'
            printd(f"Calling `{call}` in {os.path.basename(namespace['__file__'])}", topic='machine')
            # Add the callback for mapping updates
            # By supplyinh the function to the decorator we avoid a ringinclusion
            call_w_update_mapping = call[:-1] + ", update_callback=update_mapping)"
            exec( machine + "." + call_w_update_mapping)
            if isinstance(cache, dict):
                cache[call] = ods
        if location.endswith(':'):
            return (
                int(len(ods[u2n(location[:-2], [0] * 100)])),
                {'raw_data': ods, 'processed_data': ods, 'cocosio': cocosio, 'branch': mappings['__branch__']},
            )
        else:
            return ods, {'raw_data': ods, 'processed_data': ods, 'cocosio': cocosio, 'branch': mappings['__branch__']}

    # MDSplus
    elif 'TDI' in mapped:
        try:
            if 'treename' in  mapped:
                pulse_id = check_for_pulse_id(pulse, mapped['treename'], options_with_defaults)
            else:
                pulse_id = pulse
            TDI = mapped['TDI'].format(**options_with_defaults)
            treename = mapped['treename'].format(**options_with_defaults) if 'treename' in mapped else None
            data0 = data = mdsvalue(machine, treename, pulse_id, TDI).raw()
            if data is None:
                raise ValueError('data is None')
        except Exception as e:
            printe(mapped['TDI'].format(**options_with_defaults).replace('\\n', '\n'))
            if "eval2TDI" in mapped:
                e.eval2TDI = mapped['eval2TDI']
            e.TDI = mapped['TDI']
            raise e

    else:
        raise ValueError(f"Could not fetch data for {location}. Must define one of {machine_expression_types}")

    # handle size definition for array of structures
    if location.endswith(':'):
        return int(data), {'raw_data': data0, 'processed_data': data, 'cocosio': cocosio, 'branch': mappings['__branch__']}

    # transpose manipulation
    if mapped.get('TRANSPOSE', False):
        for k in range(len(mapped['TRANSPOSE']) - len(data.shape)):
            data = numpy.array([data])
        data = numpy.transpose(data, mapped['TRANSPOSE'])

    # transpose filter
    nanfilter = lambda x: x
    if mapped.get('NANFILTER', False):
        #lambda x: x[~numpy.isnan(x)]
        nanfilter = remove_nans

    # assign data to ODS
    if not hasattr(data, 'shape'):
        ods[location] = data
    else:
        with omas_environment(ods, cocosio=cocosio):
            dsize = len(data.shape)  # size of the data fetched from MDSplus
            csize = len(mapped.get('COORDINATES', []))  # number of coordinates
            osize = len([c for c in mapped.get('COORDINATES', []) if c != '1...N'])  # number of named coordinates
            asize = location.count(':') + csize  # data size required from MDSplus to make the assignement
            if asize != dsize:
                raise Exception(
                    f"Experiment data {data.shape} does not fit in `{location}` [{', '.join([':'] * location.count(':') + mapped.get('COORDINATES', []))}]"
                )
            if dsize - osize == 0 or ':' not in location:
                if data.size == 1:
                    data = data.item()
                ods[location] = nanfilter(data)
            else:
                for k in itertools.product(*list(map(range, data.shape[: location.count(':')]))):
                    ods[u2n(location, k)] = nanfilter(data[k])

    return ods, {'raw_data': data0, 'processed_data': data, 'cocosio': cocosio, 'branch': mappings['__branch__']}


_machine_mappings = {}
_namespace_mappings = {}
_user_machine_mappings = {}
_python_tdi_namespace = {}


def machine_mappings(machine, branch, user_machine_mappings=None, return_raw_mappings=False, raise_errors=False):
    """
    Function to load the json mapping files (local or remote)
    Allows for merging external mapping rules defined by users.
    This function sanity-checks and the mapping file and adds extra info required for mapping

    :param machine: machine for which to load the mapping files

    :param branch: GitHub branch from which to load the machine mapping information

    :param user_machine_mappings: Dictionary of mappings that users can pass to this function to temporarily use their mappings
                                  (useful for development and testinig purposes)

    :param return_raw_mappings: Return mappings without following __include__ statements nor resoliving `eval2TDI` directives

    :param raise_errors: raise errors or simply print warnings if something isn't right

    :return: dictionary with mapping transformations
    """
    if user_machine_mappings is None:
        user_machine_mappings = {}

    idm = (machine, branch)

    if (
        return_raw_mappings
        or idm not in _machine_mappings
        or list(_user_machine_mappings.keys()) + list(user_machine_mappings.keys()) != _machine_mappings[idm]['__user_machine_mappings__']
    ):

        # figure out mapping file
        filename = machines(machine, branch)

        # load mappings from file following __include__ directives
        if not os.stat(filename).st_size:
            top = {}
        else:
            with open(filename, 'r') as f:
                try:
                    top = json.load(f)
                except json.decoder.JSONDecodeError as _excp:
                    raise ValueError(f'Error reading {filename}\n' + str(_excp))
        go_deep = ['__cocos_rules__', '__options__']
        mappings = {k: {} for k in go_deep}
        mappings.setdefault('__include__', ['_common'])
        if not return_raw_mappings:
            for item in top.get('__include__', ['_common']):
                include_filename = os.path.split(filename)[0] + os.sep + f'{item}.json'
                with open(include_filename, 'r') as f:
                    try:
                        sub = json.load(f)
                    except json.decoder.JSONDecodeError as _excp:
                        raise ValueError(f'Error reading {include_filename}\n' + str(_excp))
                    for k in go_deep:
                        mappings[k].update(sub.setdefault(k, {}))
                        del sub[k]
                    for k in sub:
                        sub[k]['__include__'] = item
                    mappings.update(sub)
            for k in go_deep:
                mappings[k].update(top.setdefault(k, {}))
                del top[k]
        mappings.update(top)

        # merge mappings and user_machine_mappings
        mappings['__user_machine_mappings__'] = []
        for umap in [_user_machine_mappings, user_machine_mappings]:
            umap = copy.copy(umap)
            mappings['__user_machine_mappings__'].extend(list(umap.keys()))
            for item in ['__cocos_rules__', '__options__']:
                mappings[item].update(umap.pop(item, {}))
            mappings.update(umap)

        # return raw json mappings if so requested
        if return_raw_mappings:
            mappings.pop('__user_machine_mappings__')
            return mappings

        # ============= below this line we process the raw mappings =============

        mappings['__filename__'] = filename
        mappings['__branch__'] = branch

        # read the machine specific python mapping functions
        _namespace_mappings[idm] = {}
        if os.path.exists(os.path.splitext(filename)[0] + '.py'):
            with open(os.path.splitext(filename)[0] + '.py', 'r') as f:
                try:
                    exec(f.read(), _namespace_mappings[idm])
                except Exception as _excp:
                    raise _excp.__class__(f"Error in {os.path.splitext(filename)[0] + '.py'}\n" + str(_excp))

        # generate TDI for cocos_rules
        for item in mappings['__cocos_rules__']:
            if 'eval2TDI' in mappings['__cocos_rules__'][item]:
                try:
                    mappings['__cocos_rules__'][item]['TDI'] = eval(
                        mappings['__cocos_rules__'][item]['eval2TDI'].replace('\\', '\\\\'), python_tdi_namespace(branch)
                    )
                except Exception as _excp:
                    text = f"Error evaluating eval2TDI in ['__cocos_rules__'][{item!r}]: {mappings['__cocos_rules__'][item]['eval2TDI']}:\n{_excp!r}"
                    if raise_errors:
                        raise _excp.__class__(text)
                    else:
                        printe(text)

        # generate TDI and sanity check mappings
        for location in mappings:
            # sanity check format
            if l2o(p2l(location)) != location:
                raise ValueError(f'{location} mapping should be specified as {l2o(p2l(location))}')

            # generate DTI functions based on eval2DTI
            if 'eval2TDI' in mappings[location]:
                mappings[location]['TDI'] = eval(mappings[location]['eval2TDI'].replace('\\', '\\\\'), python_tdi_namespace(branch))

            # make sure required coordinates info are present in the mapping
            # this COORDINATES info is also used later to assing data in the ODS
            info = omas_info_node(location)
            if 'coordinates' in info:
                mappings[location]['COORDINATES'] = list(map(i2o, info['coordinates']))
                for coordinate in mappings[location]['COORDINATES']:
                    if '1...' in coordinate:
                        continue
                    elif coordinate not in mappings:
                        text = f'Missing coordinate {coordinate} for {location}'
                        if raise_errors:
                            raise ValueError(text)
                        else:
                            printe(text)

            # add cocos transformation info
            has_COCOS = o2u(location) in cocos_signals and cocos_signals[o2u(location)] is not None
            if 'COCOSIO' not in mappings[location] and has_COCOS:
                cocos_defined = False
                for cocos_rule in mappings['__cocos_rules__']:
                    for exp in ['TDI', 'PYTHON']:
                        if exp in mappings[location] and re.findall(cocos_rule, mappings[location][exp]):
                            for cocos_exp in ['PYTHON', 'TDI']:
                                if cocos_exp in mappings['__cocos_rules__'][cocos_rule]:
                                    mappings[location]['COCOSIO_' + cocos_exp] = mappings['__cocos_rules__'][cocos_rule][cocos_exp]
                                    cocos_defined = True
                if not cocos_defined:
                    text = f'{location} must have COCOSIO specified'
                    if raise_errors:
                        raise ValueError(text)
                    else:
                        printe(text)
            if 'COCOSIO' in mappings[location] and not has_COCOS:
                text = f'{location} should not have COCOS specified, or COCOS definition should be added to omas_cocos file'
                if raise_errors:
                    raise ValueError(text)
                else:
                    printe(text)

        # cache
        _machine_mappings[idm] = mappings

    return _machine_mappings[idm]


def reload_machine_mappings(verbose=True):
    """
    Flushes internal caches of machine mappings.
    This will force the mapping files to be re-read when they are first accessed.

    :param verbose: print to screen when mappings are reloaded
    """
    # reset machine mapping caches
    for cache in [_machine_mappings, _namespace_mappings, _python_tdi_namespace, _machines_dict, _user_machine_mappings]:
        cache.clear()

    # in case users did a `from omas.machine_mappings import ...`
    for mod in list(sys.modules):
        if mod.startswith('omas.machine_mappings'):
            del sys.modules[mod]

    if verbose:
        print('Reloaded OMAS machine mapping info')


# ===================
# list machines and update machine files
# ===================
_machines_dict = {}


def machines(machine=None, branch=''):
    """
    Function to get machines that have their mappings defined
    This function takes care of remote transfer the needed files (both .json and .py) if a remote branch is requested

    :param machine: string with machine name or None

    :param branch: GitHub branch from which to load the machine mapping information

    :return: if `machine==None` returns dictionary with list of machines and their json mapping files
             if `machine` is a string, then returns json mapping filename
    """

    # return cached results
    if branch in _machines_dict:
        if machine is None:
            return _machines_dict[branch]
        elif machine in _machines_dict[branch]:
            return _machines_dict[branch][machine]

    # local mappings
    if not branch:
        dir = omas_dir + 'machine_mappings'

    # remote mappings from GitHub
    else:
        if branch == 'master':
            svn_branch = 'trunk'
        else:
            svn_branch = 'branches/' + branch

        dir = _url_dir.format(branch=branch)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        subprocess.Popen(
            f'''
svn export --force https://github.com/gafusion/omas.git/{svn_branch}/omas/machine_mappings/ {dir}
''',
            stdout=subprocess.PIPE,
            shell=True,
        ).communicate()[0]

    # go through machine files
    _machines_dict[branch] = {}
    for filename in glob.glob(f'{dir}/*.json'):
        m = os.path.splitext(os.path.split(filename)[1])[0]
        if not m.startswith('_'):
            _machines_dict[branch][m] = os.path.abspath(filename)

    # return list of supported machines
    if machine is None:
        return _machines_dict[branch]

    # return filename with mappings for this machine
    else:
        if machine not in _machines_dict[branch]:
            raise NotImplementedError(f'Machine mapping file `{machine}.json` does not exist')
        return _machines_dict[branch][machine]


def update_mapping(machine, location, value, cocosio=None, default_options=None, update_path=False):
    """
    Utility function that updates the local mapping file of a given machine with the mapping info of a given location

    :param machine: machine name

    :param location: ODS location to be updated

    :param value: dictionary with mapping info

    :param cocosio: if integer and location has COCOS transform it adds it

    :param update_path: use the same value for the arrays of structures leading to this location

    :return: dictionary with updated raw mappings
    """
    ulocation = l2u(p2l(location))
    value = copy.copy(value)
    if 'COORDINATES' in value:
        del value['COORDINATES']
    if cocosio and ulocation in cocos_signals and cocos_signals[ulocation] is not None:
        assert isinstance(cocosio, int)
        value['COCOSIO'] = cocosio

    # operate on the raw mappings
    new_raw_mappings = machine_mappings(machine, '', None, return_raw_mappings=True)

    # assign default options
    updated_defaults = False
    if default_options:
        for item in default_options:
            if item not in new_raw_mappings['__options__'] and item not in ['machine', 'pulse', 'location']:
                new_raw_mappings['__options__'][item] = default_options[item]
                updated_defaults = True

    # if the definition is the same do not do anythinig
    # use `sorted(repr(dict))` as a cheap recursive dictionary diff
    # sorted is needed because starting with Python3.7 dictionaries are sorted and we cannot guarantee that value and mappings have same sorting
    if not updated_defaults and ulocation in new_raw_mappings and sorted(repr(value)) == sorted(repr(new_raw_mappings[ulocation])):
        return new_raw_mappings

    # add definition for new/updated location and update the .json file
    new_raw_mappings[ulocation] = value
    filename = machines(machine, '')
    with open(filename, 'w') as f:
        json.dump(new_raw_mappings, f, indent=1, separators=(',', ': '), sort_keys=True)
    print(f'Updated {machine} mapping for {ulocation}')

    # add the same call for arrays of structures going upstream
    if update_path:
        for uloc in [':'.join(ulocation.split(':')[: k + 1]) + ':' for k, l in enumerate(ulocation.split(':')[:-1])]:
            if uloc in new_raw_mappings:
                continue
            if 'COCOSIO' in value:
                value = copy.copy(value)
                del value['COCOSIO']
            update_mapping(machine, uloc, value, None, None, update_path=False)

    return new_raw_mappings



def test_machine_mapping_functions(machine, __all__, global_namespace, local_namespace, compare_to_toksearch=False, skip_omfit_tests=False):
    """
    Function used to test python mapping functions

    :param __all__: list of functions to test

    :param namespace: testing namespace
    
    :param compare_to_toksearch: if True, execute functions twice (once with mdsvalue, 
                                once with toksearch) and compare time data for consistency
                                
    :param skip_omfit_tests: if True, skip all functions that require OMFIT dependencies
                            (useful for environments without OMFIT installed)
    """
    from pprint import pprint

    old_OMAS_DEBUG_TOPIC = os.environ.get('OMAS_DEBUG_TOPIC', None)
    os.environ['OMAS_DEBUG_TOPIC'] = 'machine'

    # call machine mapping to make sure the json file is properly formatted
    # machine = os.path.splitext(os.path.split(local_namespace['__file__'])[1])[0]
    print(f'Sanity check of `{machine}` mapping files: ... ', end='')
    machine_mappings(machine, '', raise_errors=True)
    print('OK')

    __regression_arguments__ = global_namespace['__regression_arguments__']
    
    def _execute_function_with_backend(func_name, regression_kw, backend='mdsvalue'):
        """Helper function to execute a mapping function with specified backend"""
        ods = ODS(mds_backend=backend)
        func = eval(func_name, global_namespace, local_namespace)
        try:
            try:
                regression_kw_copy = regression_kw.copy()
                regression_kw_copy["update_callback"] = update_mapping
                func(ods, **regression_kw_copy)
            except Exception:
                raise
        except TypeError as _excp:
            if re.match('.*missing [0-9]+ required positional argument.*', str(_excp)):
                raise _excp.__class__(
                    str(_excp)
                    + '\n'
                    + 'For testing purposes, make sure to provide default arguments for your mapping functions via the decorator @machine_mapping_function(__regression_arguments__, ...)'
                )
            else:
                raise
        return ods
    
    def _find_time_data_for_function(ods, func_name):
        """Helper function to find time data in ODS using machine mapping JSON"""
        try:
            # Load the machine mappings to get the JSON data
            mappings = machine_mappings(machine, '', raise_errors=False)
            
            # Find all IMAS fields that map to this Python function
            function_call_pattern = f"{func_name}(ods, "
            mapped_time_fields = []
            
            for imas_field, mapping_info in mappings.items():
                if isinstance(mapping_info, dict) and 'PYTHON' in mapping_info:
                    python_call = mapping_info['PYTHON']
                    # Check if this field maps to our function and contains '.time'
                    if function_call_pattern in python_call and '.time' in imas_field:
                        mapped_time_fields.append(imas_field)
            
            # Convert IMAS field notation to ODS key notation
            # (e.g., "core_profiles.time" stays the same, "bolometer.channel.:.power.time" becomes "bolometer.channel.0.power.time")
            ods_time_keys = []
            for field in mapped_time_fields:
                # Replace array notation .:. with .0. for first element
                ods_key = field.replace('.:.', '.0.')
                ods_time_keys.append(ods_key)
            
            # Check which of these keys actually exist in the ODS
            available_keys = ods.flat().keys()
            for ods_key in ods_time_keys:
                if ods_key in available_keys:
                    try:
                        data = ods[ods_key]
                        if hasattr(data, '__len__') and len(data) > 0:
                            # Successfully found time data using JSON mapping
                            return ods_key, data
                    except:
                        continue
            
            # If no mapped time fields found, fall back to any time data
            time_keys = [key for key in available_keys if 'time' in key.lower() and not key.lower().endswith('homogeneous_time')]
            for key in time_keys:
                try:
                    data = ods[key]
                    if hasattr(data, '__len__') and len(data) > 0:
                        return key, data
                except:
                    continue
                    
        except Exception as e:
            # If JSON parsing fails, fall back to simple search
            all_keys = ods.flat().keys()
            time_keys = [key for key in all_keys if 'time' in key.lower() and not key.lower().endswith('homogeneous_time')]
            for key in time_keys:
                try:
                    data = ods[key]
                    if hasattr(data, '__len__') and len(data) > 0:
                        return key, data
                except:
                    continue
        
        return None, None
    
    def _compare_backends(func_name, regression_kw):
        """Compare function execution between mdsvalue and toksearch backends"""
        print(f"\n--- Backend Comparison for {func_name} ---")
        
        try:
            # Execute with mdsvalue backend
            print("  Executing with mdsvalue backend...")
            ods_mds = _execute_function_with_backend(func_name, regression_kw, 'mdsvalue')
            time_key_mds, time_data_mds = _find_time_data_for_function(ods_mds, func_name)
            
            if time_key_mds is None:
                print("  ⚠ No time data found in mdsvalue results")
                return
            
            print(f"    ✓ mdsvalue: {time_key_mds} with {len(time_data_mds)} points")
            
            # Execute with toksearch backend  
            print("  Executing with toksearch backend...")
            ods_toks = _execute_function_with_backend(func_name, regression_kw, 'toksearch')
            time_key_toks, time_data_toks = _find_time_data_for_function(ods_toks, func_name)
            
            if time_key_toks is None:
                print("  ⚠ No time data found in toksearch results")
                return
                
            print(f"    ✓ toksearch: {time_key_toks} with {len(time_data_toks)} points")
            
            # Compare time data
            if time_key_mds == time_key_toks:
                if len(time_data_mds) == len(time_data_toks):
                    # Compare values (allow small numerical differences)
                    if numpy.allclose(time_data_mds, time_data_toks, rtol=1e-10, atol=1e-12):
                        print(f"    ✅ Time data MATCHES between backends")
                    else:
                        max_diff = numpy.max(numpy.abs(time_data_mds - time_data_toks))
                        print(f"    ⚠ Time data differs (max diff: {max_diff:.2e})")
                else:
                    print(f"    ⚠ Different time array lengths: {len(time_data_mds)} vs {len(time_data_toks)}")
            else:
                print(f"    ⚠ Different time keys: {time_key_mds} vs {time_key_toks}")
                
        except Exception as e:
            expected_errors = ["toksearch package is required", "No module named 'toksearch'"]
            if any(err in str(e) for err in expected_errors):
                print("    ⚠ toksearch backend: Expected failure (package not installed)")
            else:
                print(f"    ✗ Backend comparison failed: {e}")
    
    try:
        # Get list of functions that require OMFIT
        requires_omfit = __regression_arguments__.get("requires_omfit", [])
        
        # Filter functions based on skip_omfit_tests parameter
        functions_to_test = []
        omfit_functions_skipped = []
        
        for func_name in __all__:
            if skip_omfit_tests and func_name in requires_omfit:
                omfit_functions_skipped.append(func_name)
            else:
                functions_to_test.append(func_name)
        
        # Report what functions are being tested
        if skip_omfit_tests and omfit_functions_skipped:
            print(f"⚠️  Skipping {len(omfit_functions_skipped)} OMFIT-dependent functions:")
            for func_name in omfit_functions_skipped:
                print(f"    - {func_name}")
            print(f"✅ Testing {len(functions_to_test)} compatible functions")
            print()
        
        for func_name in functions_to_test:
            regression_kw = {item: value for item, value in __regression_arguments__.get(func_name, {}).items() if item != '__all__'}
            print('=' * len(func_name))
            print(func_name)
            pprint(regression_kw)
            print('=' * len(func_name))
            
            # Execute with default backend for main test
            ods = _execute_function_with_backend(func_name, regression_kw)
            
            tmp = numpy.unique(list(map(o2u, ods.flat().keys()))).tolist()
            if not len(tmp):
                print('No data assigned to ODS')
                return
                
            n = max(map(lambda x: len(x), tmp))
            for item in tmp:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error')
                        print(f'{item.ljust(n)}   {numpy.array(ods[item]).shape}')
                except Exception:
                    print(f'{item.ljust(n)}   mixed')
            
            # Optional backend comparison - skip functions that require OMFIT
            if compare_to_toksearch and machine == 'd3d':  # Only for d3d for now
                if func_name in requires_omfit:
                    print(f"    ⚠️  Skipping backend comparison for {func_name} (requires OMFIT)")
                else:
                    _compare_backends(func_name, regression_kw)
    finally:
        if old_OMAS_DEBUG_TOPIC is None:
            del os.environ['OMAS_DEBUG_TOPIC']
        else:
            os.environ['OMAS_DEBUG_TOPIC'] = old_OMAS_DEBUG_TOPIC


# ===================
# Loading machine data in ODSs
# ===================
class dynamic_omas_machine(dynamic_ODS):
    """
    Class that provides dynamic data loading from machine mappings
    This class is not to be used by itself, but via the ODS.open() method.
    """

    def __init__(self, machine, pulse, options={}, branch='', user_machine_mappings=None, verbose=True):
        self.kw = {'machine': machine, 'pulse': int(pulse), 'options': options, 'branch': branch, 'user_machine_mappings': user_machine_mappings}
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
            ods, _ = machine_to_omas(
                ODS(),
                self.kw['machine'],
                self.kw['pulse'],
                o2u(key),
                self.kw['options'],
                self.kw['branch'],
                self.kw['user_machine_mappings'],
                self.cache,
            )
            self.cache[o2u(key)] = ods
        if isinstance(self.cache[o2u(key)], int):
            return self.cache[o2u(key)]
        else:
            return self.cache[o2u(key)][key]

    def __contains__(self, location):
        if not self.active:
            raise RuntimeError('Dynamic link broken: %s' % self.kw)
        ulocation = o2u(location)
        if ulocation.endswith(':'):
            return False
        return ulocation in machine_mappings(self.kw['machine'], self.kw['branch'], self.kw['user_machine_mappings'])

    def keys(self, location):
        ulocation = (o2u(location) + ".").lstrip('.')
        if ulocation + ':' in machine_mappings(self.kw['machine'], self.kw['branch'], self.kw['user_machine_mappings']):
            try:
                return list(range(self[ulocation + ':']))
            except Exception as _excp:
                printe(f'{ulocation}: issue:' + repr(_excp))
                return []
        else:
            tmp = numpy.unique(
                [
                    convert_int(k[len(ulocation) :].lstrip('.').split('.')[0])
                    for k in machine_mappings(self.kw['machine'], self.kw['branch'], self.kw['user_machine_mappings'])
                    if not k.startswith('_') and k.startswith(ulocation) and len(k[len(ulocation) :].lstrip('.').split('.')[0])
                ]
            )
            if ':' in tmp:
                raise ValueError(f"Please specify number of structures for `{o2u(location)}.:` in {self.kw['machine']}.json")
            return tmp


def load_omas_machine(
    machine,
    pulse,
    options={},
    consistency_check=True,
    imas_version=omas_rcparams['default_imas_version'],
    cls=ODS,
    branch='',
    user_machine_mappings=None,
):
    printd('Loading from %s' % machine, topic='machine')
    ods = cls(imas_version=imas_version, consistency_check=consistency_check)
    for location in [location for location in machine_mappings(machine, branch, user_machine_mappings) if not location.startswith('__')]:
        if location.endswith(':'):
            continue
        print(location)
        machine_to_omas(ods, machine, pulse, location, options, branch)
    return ods
