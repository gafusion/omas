#--------------
# uncertainties
import uncertainties.core
_orig_uncertainties_Variable = uncertainties.core.Variable
class Variable(_orig_uncertainties_Variable):
    '''
    OMFIT modified Variable to implemet caching of 0 nominal_value, tag=None objects
    '''
    # Cache Variable type variables here
    var_cache = {}

    # Want to get actual existing object if it exists
    def __new__(cls,value=None,std_dev=None,tag=None):
        if value == 0 and tag is None and repr(std_dev) in Variable.var_cache:
            return Variable.var_cache[repr(std_dev)]
        else:
            return super(Variable, cls).__new__(Variable,value,std_dev,tag=tag)

    # Need this to get unpickling to call __new__
    def __getnewargs__(self):
        return self.nominal_value, self.std_dev, self.tag

    # Slightly modified from original __init__ to point to super(_orig...)
    # and to cache result
    def __init__(self,value,std_dev,tag=None):
        """
        The nominal value and the standard deviation of the variable
        are set.
        The value is converted to float.
        The standard deviation std_dev can be NaN. It should normally
        be a float or an integer.
        'tag' is a tag that the user can associate to the variable.  This
        is useful for tracing variables.
        The meaning of the nominal value is described in the main
        module documentation.
        """
        if value == 0 and tag is None and repr(std_dev) in Variable.var_cache:
            return
        #! The value, std_dev, and tag are assumed by __copy__() not to
        # be copied.  Either this should be guaranteed here, or __copy__
        # should be updated.

        # Only float-like values are handled.  One reason is that the
        # division operator on integers would not produce a
        # differentiable functions: for instance, Variable(3, 0.1)/2
        # has a nominal value of 3/2 = 1, but a "shifted" value
        # of 3.1/2 = 1.55.
        value = float(value)

        # If the variable changes by dx, then the value of the affine
        # function that gives its value changes by 1*dx:

        # ! Memory cycles are created.  However, they are garbage
        # collected, if possible.  Using a weakref.WeakKeyDictionary
        # takes much more memory.  Thus, this implementation chooses
        # more cycles and a smaller memory footprint instead of no
        # cycles and a larger memory footprint.
        super(_orig_uncertainties_Variable, self).__init__(value, {self: 1.})

        self.std_dev = std_dev  # Assignment through a Python property

        self.tag = tag
        if value==0 and tag is None:
            Variable.var_cache[repr(std_dev)] = self

    # Slightly modified from original __repr__ to point to super(_orig...)
    def __repr__(self):

        num_repr  = super(_orig_uncertainties_Variable, self).__repr__()

        if self.tag is None:
            return num_repr
        else:
            return "< %s = %s >" % (self.tag, num_repr)

uncertainties.core.Variable = Variable
import uncertainties
import uncertainties.unumpy as unumpy
if not hasattr(uncertainties, 'Variable'):
    setattr(uncertainties, 'Variable', uncertainties.core.Variable)
if not hasattr(uncertainties, 'AffineScalarFunc'):
    setattr(uncertainties, 'AffineScalarFunc', uncertainties.core.AffineScalarFunc)
if not hasattr(uncertainties, 'CallableStdDev'):
    class _uncertainties__CallableStdDev(float):
        def __call__(self):
            return self
    setattr(uncertainties, 'CallableStdDev', _uncertainties__CallableStdDev)
from uncertainties.unumpy import nominal_values, std_devs, uarray
from uncertainties import ufloat_fromstr
