import json
import os
from omas.omas_utils import printd
import numpy as np

__all__ = [
    'mdstree',
    'mdsvalue',
    'get_pulse_id'
]

def get_pulse_id(pulse, run_id=None):
    """
    Converts the pulse number into a MDSplus run_id

    :param pulse: Regular shot number

    :param run_id: Extension that contains the pulse number. E.g."01". Should be of type string or None

    :return: Pulse id, i.e. shot number with run_id extension if available
    """
    if run_id is None:
        return pulse
    else:
        return int(str(pulse) + run_id)

def check_for_pulse_id(pulse, treename, options_with_defaults):
    """
    Checks if the tree has a run_id associated with it and returns the pulse id

    :param pulse: Regular shot number

    :param treename: Name of tree being loaded

    :param options_with_defaults: Dictionary with options for the current machine


    """
    if 'EFIT' in treename:
        return get_pulse_id(pulse, options_with_defaults["EFIT_run_id"])
    else:
        return pulse


_mds_connection_cache = {}

_last_open_tree = {} # Tracks the last opened tree to avoid reopening trees

def open_tree_cached(conn, server, shot, tree):
    global _last_open_tree
    if (shot, tree) == _last_open_tree.get(server, None):
        return
    else:
        conn.openTree(tree, shot)
        _last_open_tree[server] = (shot, tree)

# ===================
# MDSplus functions
# ===================
def tunnel_mds(server, treename):
    """
    Resolve MDSplus server
    NOTE: This function makes use of the optional `omfit_classes` dependency to establish a SSH tunnel to the MDSplus server.

    :param server: MDSplus server address:port

    :param treename: treename (in case treename affects server to be used)

    :return: string with MDSplus server and port to be used
    """
    try:
        import omfit_classes.omfit_mds
    except (ImportError, ModuleNotFoundError):
        return server.format(**os.environ)
    else:
        server0 = omfit_classes.omfit_mds.translate_MDSserver(server, treename)
        tunneled_server = omfit_classes.omfit_mds.tunneled_MDSserver(server0, quiet=False)
        return tunneled_server

    return server.format(**os.environ)





class mdsvalue(dict):
    """
    Execute MDSplus TDI functions
    """

    def __init__(self, server, treename, pulse, TDI, old_MDS_server=False):
        self.treename = treename
        self.pulse = pulse
        self.TDI = TDI
        if 'nstx' in server:
            old_MDS_server = True
        try:
            # handle the case that server is just the machine name
            machine_mappings_path = os.path.join(os.path.dirname(__file__), "../", "machine_mappings")
            machine_mappings_path = os.path.join(machine_mappings_path, server + ".json")
            with open(machine_mappings_path, "r") as machine_file:
                    server = json.load(machine_file)["__mdsserver__"]
        except Exception:
            # hanlde case where server is actually a URL
            if '.' not in server:
                raise
        self.server = tunnel_mds(server, self.treename)
        old_servers = ['skylark.pppl.gov:8500', 'skylark.pppl.gov:8501', 'skylark.pppl.gov:8000']
        if server in old_servers or self.server in old_servers:
            old_MDS_server = True
        self.old_MDS_server = old_MDS_server

    def data(self):
        return self.raw(f'data({self.TDI})')

    def dim_of(self, dim):
        return self.raw(f'dim_of({self.TDI},{dim})')

    def units(self):
        return self.raw(f'units({self.TDI})')

    def error(self):
        return self.raw(f'error({self.TDI})')

    def error_dim_of(self, dim):
        return self.raw(f'error_dim_of({self.TDI},{dim})')

    def units_dim_of(self, dim):
        return self.raw(f'units_dim_of({self.TDI},{dim})')

    def size(self, dim):
        return self.raw(f'size({self.TDI})')

    def raw(self, TDI=None):
        """
        Fetch data from MDSplus with connection caching

        :param TDI: string, list or dict of strings
            MDSplus TDI expression(s) (overrides the one passed when the object was instantiated)

        :return: result of TDI expression, or dictionary with results of TDI expressions
        """
        try:
            import time

            t0 = time.time()
            import MDSplus

            def mdsk(value):
                """
                Translate strings to MDSplus bytes
                """
                return str(str(value).encode('utf8'))

            if TDI is None:
                TDI = self.TDI

            try:
                out_results = None

                # try connecting and re-try on fail
                for fallback in [0, 1]:
                    if self.server not in _mds_connection_cache:
                        _mds_connection_cache[self.server] = MDSplus.Connection(self.server)
                        _last_open_tree[self.server] = None
                    try:
                        conn = _mds_connection_cache[self.server]
                        if self.treename is not None:
                            open_tree_cached(conn, self.server, self.pulse, self.treename)
                        break
                    except Exception as _excp:
                        if self.server in _mds_connection_cache:
                            _last_open_tree[self.server] = None
                            _mds_connection_cache[self.server].reconnect()
                        if fallback:
                            raise
                
                # list of TDI expressions
                if isinstance(TDI, (list, tuple)):
                    TDI = {expr: expr for expr in TDI}

                # dictionary of TDI expressions
                if isinstance(TDI, dict):
                    # old versions of MDSplus server do not support getMany
                    if self.old_MDS_server:
                        results = {}
                        for tdi in TDI:
                            try:
                                data = mdsvalue(self.server, self.treename, self.pulse, TDI[tdi]).raw()

                                # Convert float32 arrays to float64 to preserve precision
                                if isinstance(data, np.ndarray) and data.dtype == np.float32:
                                    data = data.astype(np.float64)
                                results[tdi] = data
                            except Exception as _excp:
                                results[tdi] = Exception(str(_excp))
                        out_results = results

                    # more recent MDSplus server
                    else:
                        conns = conn.getMany()
                        for name, expr in TDI.items():
                            conns.append(name, expr)
                        res = conns.execute()
                        results = {}
                        for name, expr in TDI.items():
                            try:
                                data = MDSplus.Data.data(res[mdsk(name)][mdsk('value')])
                            except KeyError:
                                try:
                                    data = MDSplus.Data.data(res[str(name)][str('value')])
                                except KeyError:
                                    try:
                                        results[name] = Exception(MDSplus.Data.data(res[mdsk(name)][mdsk('error')]))
                                        continue
                                    except KeyError:
                                        results[name] = Exception(MDSplus.Data.data(res[str(name)][str('error')]))
                                        continue

                            # Convert float32 arrays to float64 to preserve precision
                            if isinstance(data, np.ndarray) and data.dtype == np.float32:
                                data = data.astype(np.float64)
                            results[name] = data
                        out_results = results

                # single TDI expression
                else:
                    out_results = MDSplus.Data.data(conn.get(TDI))

                    # Convert float32 arrays to float64 to preserve precision
                    if isinstance(out_results, np.ndarray) and out_results.dtype == np.float32:
                        out_results = out_results.astype(np.float64)

                # return values
                return out_results

            except Exception as _excp:
                txt = []
                for item in ['server', 'treename', 'pulse']:
                    txt += [f' - {item}: {getattr(self, item)}']
                txt += [f' - TDI: {TDI}']
                raise _excp.__class__(str(_excp) + '\n' + '\n'.join(txt))

        finally:
            if out_results is not None:
                if isinstance(out_results, dict):
                    if all(isinstance(out_results[k], Exception) for k in out_results):
                        printd(f'{TDI} \tall NO\t {time.time() - t0:3.3f} secs', topic='machine')
                    elif any(isinstance(out_results[k], Exception) for k in out_results):
                        printd(f'{TDI} \tsome OK/NO\t {time.time() - t0:3.3f} secs', topic='machine')
                    else:
                        printd(f'{TDI} \tall OK\t {time.time() - t0:3.3f} secs', topic='machine')
                else:
                    printd(f'{TDI} \tOK\t {time.time() - t0:3.3f} secs', topic='machine')
            else:
                printd(f'{TDI} \tNO\t {time.time() - t0:3.3f} secs', topic='machine')

class mdstree(dict):
    """
    Class to handle the structure of an MDSplus tree.
    Nodes in this tree are mdsvalue objects
    """

    def __init__(self, server, treename, pulse):
        for TDI in sorted(mdsvalue(server, treename, pulse, rf'getnci("***","FULLPATH")').raw())[::-1]:
            try:
                TDI = TDI.decode('utf8')
            except AttributeError:
                pass
            TDI = TDI.strip()
            path = TDI.replace('::TOP', '').lstrip('\\').replace(':', '.').split('.')
            h = self
            for p in path[1:-1]:
                h = h.setdefault(p, mdsvalue(server, treename, pulse, ''))
            if path[-1] not in h:
                h[path[-1]] = mdsvalue(server, treename, pulse, TDI)
            else:
                h[path[-1]].TDI = TDI
