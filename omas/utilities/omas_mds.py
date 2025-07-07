import json
import os
from omas.omas_utils import printd

__all__ = [
    'mdstree',
    'mdsvalue',
    'mds_provider',
    'toksearch_provider', 
    'BaseProvider',
    'get_pulse_id',
    'get_mds_backend',
    'set_default_mds_backend',
    'create_mds_provider',
    'auto_mds_provider'
]

# Global variable to store the default backend
_default_mds_backend = 'mdsvalue'

def get_mds_backend():
    """
    Get the current default MDS backend
    
    :return: string with backend name ('mdsvalue' or 'toksearch')
    """
    return _default_mds_backend

def set_default_mds_backend(backend):
    """
    Set the default MDS backend for data retrieval
    
    :param backend: string with backend name ('mdsvalue' or 'toksearch')
    """
    global _default_mds_backend
    valid_backends = ['mdsvalue', 'toksearch']
    if backend not in valid_backends:
        raise ValueError(f"Backend must be one of {valid_backends}, got '{backend}'")
    _default_mds_backend = backend
    printd(f"Default MDS backend set to: {backend}", topic='machine')

def create_mds_provider(server, backend=None, **kwargs):
    """
    Factory function to create the appropriate MDS provider instance
    
    :param server: MDSplus server address or machine name
    :param backend: specific backend to use ('mdsvalue' or 'toksearch'), 
                   uses default if None
    :param kwargs: additional arguments passed to provider constructor
    
    :return: Instance of the appropriate provider class
    """
    if backend is None:
        backend = _default_mds_backend
    
    if backend == 'mdsvalue':
        return mds_provider(server, **kwargs)
    elif backend == 'toksearch':
        return toksearch_provider(server, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

def auto_mds_provider(server, **kwargs):
    """
    Convenience function that uses the current default backend
    
    :param server: MDSplus server address or machine name
    :param kwargs: additional arguments passed to provider constructor
    
    :return: Instance of the default provider class
    """
    return create_mds_provider(server, **kwargs)

# Legacy factory functions for backward compatibility
def create_mds_backend(server, treename, pulse, TDI, backend=None, **kwargs):
    """Legacy function - creates old-style backend instance"""
    if backend is None:
        backend = _default_mds_backend
    
    if backend == 'mdsvalue':
        return mdsvalue(server, treename, pulse, TDI, **kwargs)
    elif backend == 'toksearch':
        raise ValueError("Legacy toksearch class removed - use toksearch_provider instead")
    else:
        raise ValueError(f"Unknown backend: {backend}")

def auto_mds_backend(server, treename, pulse, TDI, **kwargs):
    """Legacy function - creates old-style backend instance"""
    return create_mds_backend(server, treename, pulse, TDI, **kwargs)

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





class BaseProvider:
    """
    Base class for MDS data providers
    Provider is instantiated once per server and reused for multiple signal fetches
    """

    def __init__(self, server, **kwargs):
        self.server = server
        self._init_server_connection(**kwargs)

    def _init_server_connection(self, **kwargs):
        """Initialize server connection - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _init_server_connection")

    def data(self, treename, pulse, TDI):
        """Get data from the signal"""
        return self.raw(treename, pulse, f'data({TDI})')

    def dim_of(self, treename, pulse, TDI, dim):
        """Get dimension of the signal"""
        return self.raw(treename, pulse, f'dim_of({TDI},{dim})')

    def units(self, treename, pulse, TDI):
        """Get units of the signal"""
        return self.raw(treename, pulse, f'units({TDI})')

    def error(self, treename, pulse, TDI):
        """Get error/uncertainty of the signal"""
        return self.raw(treename, pulse, f'error({TDI})')

    def error_dim_of(self, treename, pulse, TDI, dim):
        """Get error of dimension"""
        return self.raw(treename, pulse, f'error_dim_of({TDI},{dim})')

    def units_dim_of(self, treename, pulse, TDI, dim):
        """Get units of dimension"""
        return self.raw(treename, pulse, f'units_dim_of({TDI},{dim})')

    def size(self, treename, pulse, TDI, dim):
        """Get size of dimension"""
        return self.raw(treename, pulse, f'size({TDI})')

    def raw(self, treename, pulse, TDI):
        """Fetch raw data - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement raw() method")
    
    def mds_tree(self, treename, pulse):
        """
        Create tree structure for the given tree and pulse
        
        :param treename: MDSplus tree name
        :param pulse: pulse number
        :return: dictionary representing the tree structure
        """
        # Get all node paths in the tree
        node_paths = self.raw(treename, pulse, rf'getnci("***","FULLPATH")')
        
        # Build hierarchical tree structure
        tree = {}
        for path in sorted(node_paths)[::-1]:
            try:
                path = path.decode('utf8')
            except AttributeError:
                pass
            path = path.strip()
            
            # Parse the path into components
            path_parts = path.replace('::TOP', '').lstrip('\\').replace(':', '.').split('.')
            # Filter out empty parts
            path_parts = [part for part in path_parts if part]
            
            # Navigate/create the tree structure
            current = tree
            for part in path_parts[:-1]:
                current = current.setdefault(part, {})
            
            # Set the final node with its TDI expression
            if path_parts[-1] not in current:
                current[path_parts[-1]] = {'TDI': path}
            else:
                current[path_parts[-1]]['TDI'] = path
        # Return only the elements of the requested tree
        return tree[treename]


class mdsvalue(dict):
    """
    Execute MDSplus TDI functions
    """

    def __init__(self, server, treename, pulse, TDI, **kwargs):
        super().__init__()
        self.treename = treename
        self.pulse = pulse
        self.TDI = TDI
        self.server = server
        self._init_server_connection(**kwargs)

    def data(self):
        """Get data from the signal"""
        return self.raw(f'data({self.TDI})')

    def dim_of(self, dim):
        """Get dimension of the signal"""
        return self.raw(f'dim_of({self.TDI},{dim})')

    def units(self):
        """Get units of the signal"""
        return self.raw(f'units({self.TDI})')

    def error(self):
        """Get error/uncertainty of the signal"""
        return self.raw(f'error({self.TDI})')

    def error_dim_of(self, dim):
        """Get error of dimension"""
        return self.raw(f'error_dim_of({self.TDI},{dim})')

    def units_dim_of(self, dim):
        """Get units of dimension"""
        return self.raw(f'units_dim_of({self.TDI},{dim})')

    def size(self, dim):
        """Get size of dimension"""
        return self.raw(f'size({self.TDI})')

    def _init_server_connection(self, old_MDS_server=False, **kwargs):
        # Check if server is already resolved (modern provider-based usage)
        if '.' in self.server and ':' in self.server:
            # Server appears to be already resolved (has dots and port)
            old_servers = ['skylark.pppl.gov:8500', 'skylark.pppl.gov:8501', 'skylark.pppl.gov:8000']
            if self.server in old_servers:
                old_MDS_server = True
            self.old_MDS_server = old_MDS_server
            return
        
        # Legacy path: resolve server from scratch (for backward compatibility only)
        # NOTE: This is inefficient and should be avoided - use mds_provider instead
        if 'nstx' in self.server:
            old_MDS_server = True
        try:
            # handle the case that server is just the machine name
            machine_mappings_path = os.path.join(os.path.dirname(__file__), "../", "machine_mappings")
            machine_mappings_path = os.path.join(machine_mappings_path, self.server + ".json")
            with open(machine_mappings_path, "r") as machine_file:
                    self.server = json.load(machine_file)["__mdsserver__"]
        except Exception:
            # handle case where server is actually a URL
            if '.' not in self.server:
                raise
        self.server = tunnel_mds(self.server, self.treename)
        old_servers = ['skylark.pppl.gov:8500', 'skylark.pppl.gov:8501', 'skylark.pppl.gov:8000']
        if self.server in old_servers:
            old_MDS_server = True
        self.old_MDS_server = old_MDS_server


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
                    if (self.server, self.treename, self.pulse) not in _mds_connection_cache:
                        conn = MDSplus.Connection(self.server)
                        if self.treename is not None:
                            conn.openTree(self.treename, self.pulse)
                        _mds_connection_cache[(self.server, self.treename, self.pulse)] = conn
                    try:
                        conn = _mds_connection_cache[(self.server, self.treename, self.pulse)]
                        break
                    except Exception as _excp:
                        if (self.server, self.treename, self.pulse) in _mds_connection_cache:
                            del _mds_connection_cache[(self.server, self.treename, self.pulse)]
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
                                results[tdi] = mdsvalue(self.server, self.treename, self.pulse, TDI[tdi]).raw()
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
                                results[name] = MDSplus.Data.data(res[mdsk(name)][mdsk('value')])
                            except KeyError:
                                try:
                                    results[name] = MDSplus.Data.data(res[str(name)][str('value')])
                                except KeyError:
                                    try:
                                        results[name] = Exception(MDSplus.Data.data(res[mdsk(name)][mdsk('error')]))
                                    except KeyError:
                                        results[name] = Exception(MDSplus.Data.data(res[str(name)][str('error')]))
                        out_results = results

                # single TDI expression
                else:
                    out_results = MDSplus.Data.data(conn.get(TDI))

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



# ===============================
# New Provider-based Architecture
# ===============================

class mds_provider(BaseProvider):
    """
    MDSplus provider - instantiated once per server, reused for multiple signals
    """
    
    def _init_server_connection(self, old_MDS_server=False, **kwargs):
        """Initialize MDSplus server connection"""
        # Store original server name for reference
        self.original_server = self.server
        
        if 'nstx' in self.server:
            old_MDS_server = True
        try:
            # handle the case that server is just the machine name
            machine_mappings_path = os.path.join(os.path.dirname(__file__), "../", "machine_mappings")
            machine_mappings_path = os.path.join(machine_mappings_path, self.server + ".json")
            with open(machine_mappings_path, "r") as machine_file:
                    self.server = json.load(machine_file)["__mdsserver__"]
        except Exception:
            # handle case where server is actually a URL
            if '.' not in self.server:
                raise
        
        # Resolve server URL with tunneling (done once at provider initialization)
        self.resolved_server = tunnel_mds(self.server, None)
        
        old_servers = ['skylark.pppl.gov:8500', 'skylark.pppl.gov:8501', 'skylark.pppl.gov:8000']
        if self.resolved_server in old_servers:
            old_MDS_server = True
        self.old_MDS_server = old_MDS_server
    

    def raw(self, treename, pulse, TDI):
        """
        Fetch data from MDSplus with connection caching
        
        :param treename: MDSplus tree name
        :param pulse: pulse number  
        :param TDI: string, list or dict of strings - MDSplus TDI expression(s)
        
        :return: result of TDI expression, or dictionary with results of TDI expressions
        """
        try:
            import time
            
            t0 = time.time()
            import MDSplus
            
            def mdsk(value):
                """Translate strings to MDSplus bytes"""
                return str(str(value).encode('utf8'))
            
            try:
                out_results = None
                
                # Resolve server for this specific tree
                server = self.resolved_server
                
                # try connecting and re-try on fail
                for fallback in [0, 1]:
                    if (server, treename, pulse) not in _mds_connection_cache:
                        conn = MDSplus.Connection(server)
                        if treename is not None:
                            conn.openTree(treename, pulse)
                        _mds_connection_cache[(server, treename, pulse)] = conn
                    try:
                        conn = _mds_connection_cache[(server, treename, pulse)]
                        break
                    except Exception as _excp:
                        if (server, treename, pulse) in _mds_connection_cache:
                            del _mds_connection_cache[(server, treename, pulse)]
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
                                # Use direct MDSplus connection for legacy servers (avoid creating temporary mdsvalue)
                                result = MDSplus.Data.data(conn.get(TDI[tdi]))
                                results[tdi] = result
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
                                results[name] = MDSplus.Data.data(res[mdsk(name)][mdsk('value')])
                            except KeyError:
                                try:
                                    results[name] = MDSplus.Data.data(res[str(name)][str('value')])
                                except KeyError:
                                    try:
                                        results[name] = Exception(MDSplus.Data.data(res[mdsk(name)][mdsk('error')]))
                                    except KeyError:
                                        results[name] = Exception(MDSplus.Data.data(res[str(name)][str('error')]))
                        out_results = results
                
                # single TDI expression
                else:
                    out_results = MDSplus.Data.data(conn.get(TDI))
                
                # return values
                return out_results
            
            except Exception as _excp:
                txt = []
                for item in ['server', 'treename', 'pulse']:
                    value = locals().get(item, getattr(self, item, 'unknown'))
                    txt += [f' - {item}: {value}']
                txt += [f' - TDI: {TDI}']
                raise _excp.__class__(str(_excp) + '\n' + '\n'.join(txt))
        
        finally:
            if 'out_results' in locals() and out_results is not None:
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


class toksearch_provider(BaseProvider):
    """
    TokSearch provider - instantiated once per server, reused for multiple signals
    """
    
    def _init_server_connection(self, **kwargs):
        """Initialize toksearch connection and import MdsSignal"""
        # Import and cache MdsSignal class
        try:
            from toksearch import MdsSignal
            self.MdsSignal = MdsSignal
        except ImportError:
            raise ImportError("toksearch package is required for toksearch backend")
        
        # Set up server mapping for MdsSignal
        server_map = {
            'd3d': 'remote://atlas.gat.com:8000'
        }
        
        # Get the actual server location
        if self.server in server_map:
            self.mds_server = server_map[self.server]
        else:
            # Assume it's already a proper server specification
            self.mds_server = self.server
    
    def raw(self, treename, pulse, TDI):
        """
        Fetch data using toksearch MdsSignal backend
        
        :param treename: MDSplus tree name
        :param pulse: pulse number
        :param TDI: string, list or dict of strings - TDI expression(s) to fetch
        
        :return: result of TDI expression, or dictionary with results of TDI expressions
        """
        try:
            import time
            
            t0 = time.time()
            
            # Handle different TDI input types
            if isinstance(TDI, (list, tuple)):
                TDI = {expr: expr for expr in TDI}
            
            # Dictionary of TDI expressions
            if isinstance(TDI, dict):
                results = {}
                for name, expr in TDI.items():
                    try:
                        signal = self.MdsSignal(expr, treename, location=self.mds_server)
                        result = signal.gather(pulse)
                        # Extract the data component
                        if isinstance(result, dict) and 'data' in result:
                            results[name] = result['data']
                        else:
                            results[name] = result
                        # Clean up the signal
                        signal.cleanup_shot(pulse)
                    except Exception as _excp:
                        results[name] = Exception(str(_excp))
                return results
            
            # Single TDI expression
            else:
                signal = self.MdsSignal(TDI, treename, location=self.mds_server)
                result = signal.gather(pulse)
                # Clean up the signal
                signal.cleanup_shot(pulse)
                # Extract the data component
                if isinstance(result, dict) and 'data' in result:
                    return result['data']
                else:
                    return result
                    
        except Exception as _excp:
            txt = []
            for item in ['server', 'treename', 'pulse']:
                value = locals().get(item, getattr(self, item, 'unknown'))
                txt += [f' - {item}: {value}']
            txt += [f' - TDI: {TDI}']
            raise _excp.__class__(str(_excp) + '\n' + '\n'.join(txt))
        
        finally:
            if 'results' in locals():
                if isinstance(results, dict):
                    if all(isinstance(results[k], Exception) for k in results):
                        printd(f'{TDI} \tall NO\t {time.time() - t0:3.3f} secs', topic='machine')
                    elif any(isinstance(results[k], Exception) for k in results):
                        printd(f'{TDI} \tsome OK/NO\t {time.time() - t0:3.3f} secs', topic='machine')
                    else:
                        printd(f'{TDI} \tall OK\t {time.time() - t0:3.3f} secs', topic='machine')
                else:
                    printd(f'{TDI} \tOK\t {time.time() - t0:3.3f} secs', topic='machine')
            else:
                printd(f'{TDI} \tNO\t {time.time() - t0:3.3f} secs', topic='machine')