def tile(a, n):
    import numpy as np

    a = a.data()
    return np.array([a for k in range(n)])

def stack_outer(*args):
    if len(args) < 2:
        raise ValueError("stack_outer only makes sense with 2 ore more arguments")
    import numpy as np
    a = np.array(args[0].data)
    for array in args[1:]:
        a = np.concatenate([a,array],axis=1)


def nan_where(a, b, n):
    import numpy as np

    a = a.data()
    b = b.data()
    a[b == n] = np.NaN
    return a


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

def efit_psi_to_psi(a, b, c):
    a = a.data()
    b = b.data()
    c = c.data()
    return (a - b)/(c - b)

def py2tdi(func, *args):
    import inspect
    import re

    function = inspect.getsource(func)
    function_name = re.findall(r'def (\w+)\(', function)[0]

    function = function.strip().replace('\n', '\\n').replace('{', '{{').replace('}', '}}')

    arguments = ', '.join(map(str, args))
    TDI = f"""ADDFUN("{function_name}","exec('''{function}''')"),{function_name}({arguments})"""
    return TDI
