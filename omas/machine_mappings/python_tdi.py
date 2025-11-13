def tile(a, n):
    import numpy as np

    a = a.data()
    return np.array([a for k in range(n)])

def stack_outer_2(a, b):
    import numpy as np

    a = a.data()
    b = b.data()
    return np.concatenate([a, b],axis=1)

def nan_where(a, b, n):
    import numpy as np

    a = a.data()
    b = b.data()
    a[b == n] = np.nan
    return a

def get_largest_axis_value(a, b):
    import numpy as np

    a = a.data()
    b = b.data()
    a = np.array([a for k in range(b.shape[0])])
    a[b == 0] = 0
    return np.nanmax(a, axis=1)

def MDS_gEQDSK_COCOS_identify(bt, ip):
    import numpy as np

    bt = np.mean(bt)
    ip = np.mean(ip)
    g_cocos = {(+1, +1): 1, (+1, -1): 3, (-1, +1): 5, (-1, -1): 7, (+1, 0): 1, (-1, 0): 3}
    sign_Bt = int(np.sign(bt))
    sign_Ip = int(np.sign(ip))
    return g_cocos.get((sign_Bt, sign_Ip), None)

def geqdsk_psi(a, b, c):
    import numpy as np

    a = a.data()
    b = b.data()
    c = c.data()
    n = len(c)
    M = a[:, None] + np.linspace(0, 1, n).T * (b[:, None] - a[:, None])
    return M

def efit_psi_to_real_psi_2d(a, b, c):
    import numpy as np

    # a = ensure_2d(a)
    a = a.data()
    if len(a.shape) < 2:
        a = np.atleast_2d(a)
    b = b.data()
    c = c.data()
    return (a.T * (c - b) + b).T

def convert_from_mega_2d(a):
    import numpy as np

    #return ensure_2d(a)*1.e6
    a = a.data()
    if len(a.shape) < 2:
        a = np.atleast_2d(a)
    return a*1.e6

def ensure_2d(a):
    import numpy as np
    
    a = a.data()
    if len(a.shape) < 2:
        return np.atleast_2d(a)
    else:
        return a

def interpolate_psi_1d(x1, y1, a, b, c):
    import numpy as np
    from scipy.interpolate import interp1d
   
    #x2 = geqdsk_psi(a, b, c)
    a = a.data()
    b = b.data()
    c = c.data()
    n = len(c)
    x2 = a[:, None] + np.linspace(0, 1, n).T * (b[:, None] - a[:, None])
    y2 = np.zeros(np.shape(x2))
    x1 = x1.data().T
    y1 = y1.data().T
    for i in range(len(x1[:, 0])):
        # Use constant extrapolation: first and last values for out-of-bounds
        fill_value = (y1[i][0], y1[i][-1])
        y2[i] = interp1d(x1[i], y1[i], kind='cubic', bounds_error=False, fill_value=fill_value)(x2[i])
    return y2

def py2tdi(func, *args):
    import inspect
    import re

    function = inspect.getsource(func)
    function_name = re.findall(r'def (\w+)\(', function)[0]

    function = function.strip().replace('\n', '\\n').replace('{', '{{').replace('}', '}}')

    arguments = ', '.join(map(str, args))
    TDI = f"""ADDFUN("{function_name}","exec('''{function}''')"),{function_name}({arguments})"""
    return TDI
