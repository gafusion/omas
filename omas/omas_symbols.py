"""
Formatted symbols for physics quantities

Run this script standalone to reorganize the symbols/units definitions in this script
"""
import re
from .omas_utils import l2u, p2l, u2o, l2o

__all__ = ['latexit']

_symbols = {}
# symbols start
_symbols['.b_field_r'] = r'$B_{r}$'
_symbols['.b_field_tor'] = r'$B_{\phi}$'
_symbols['.b_field_z'] = r'$B_{z}$'
_symbols['.electrons.density_thermal'] = '$n_e$'
_symbols['.ion.:.density'] = r'$n_{i:}$'
_symbols['.n_e'] = '$n_e$'
_symbols['.potential_floating'] = r'$\phi_f'
_symbols['.potential_plasma'] = r'$\phi_{plasma}'
_symbols['.pressure'] = '$P$'
_symbols['.psi'] = r'$\psi$'
_symbols['.rho_tor_norm'] = 'r$\rho$'
_symbols['.saturation_current_ion'] = '$I_{sat}$'
_symbols['.t_e'] = '$T_e$'
_symbols['.t_i'] = '$T_i$'
_symbols['.zeff'] = r'$Z_{\rm eff}$'
# symbols end

_units = {}
# units start
_units['m^-3'] = '$m^{-3}$'
# units end


def nth_repl(s, sub, repl, n):
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find + len(sub) :]
    return s


class PhysicsSymbols(dict):
    def __init__(self, *args, **kw):
        dict.__init__(self, *args, **kw)
        self.update(_symbols)
        self.update(_units)

    def __getitem__(self, location):
        # return if perfect match
        if location in self:
            return dict.__getitem__(self, location)

        # strip .values or .data from location
        location_in = p2l(location)
        if location_in[-1] in ['values', 'data']:
            location_in = location_in[:-1]
        location = l2u(location_in).rstrip('.:')

        # search through the symbols sorted so that longer ending paths have precedence
        for symbol in reversed(sorted(self.keys(), key=lambda x: x[::-1])):
            if location.endswith(symbol) or ('.' not in location and location.endswith(symbol.lstrip('.'))):
                tmp = dict.__getitem__(self, symbol)
                if ':' in tmp:  # if : is found in the result, then substitute it with the index of the location provided by the user
                    indexes = [k for k in location_in if isinstance(k, int)]
                    for k in reversed(range(tmp.count(':'))):
                        tmp = nth_repl(tmp, ':', str(indexes[-1 - k]), tmp.count(':') - k)
                return tmp

        raise KeyError(location)


latexit = PhysicsSymbols()
