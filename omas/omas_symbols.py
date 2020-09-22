"""
Formatted symbols for physics quantities

Run this script standalone to reorganize the symbols/units definitions in this script
"""
import re
from .omas_utils import l2u, p2l

__all__ = ['latexit']

_symbols = {}
# symbols start
_symbols['.n_e'] = '$n_e$'
_symbols['.potential_floating'] = r'$\phi_f'
_symbols['.potential_plasma'] = r'$\phi_{plasma}'
_symbols['.psi'] = '$\\psi$'
_symbols['.rho_tor_norm'] = '$\\rho$'
_symbols['.saturation_current_ion'] = '$I_{sat}$'
_symbols['.t_e'] = '$T_e$'
_symbols['.t_i'] = '$T_i$'
_symbols['.zeff'] = '$Z_{\\rm eff}$'
# symbols end

_units = {}
# units start
_units['m^-3'] = '$m^{-3}$'
# units end


class PhysicsSymbols(dict):
    def __getitem__(self, location):
        if location in self:
            return dict.__getitem__(self, location)
        location = l2u(p2l(location))
        for symbol in self:
            if '.' not in location and location.endswith(symbol.lstrip('.')):
                return dict.__getitem__(self, symbol)
            elif location.endswith(symbol):
                return dict.__getitem__(self, symbol)
        raise KeyError(location)


latexit = PhysicsSymbols()
latexit.update(_symbols)
latexit.update(_units)
