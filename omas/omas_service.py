def omas_service(verbose=True):
    from omas.omas_core import dynamic_ODS_factory
    import Pyro5.api
    import socket

    daemon = Pyro5.api.Daemon(port=0)
    uri = daemon.register(dynamic_ODS_factory, 'dynamic_ODS_factory')
    uri = str(uri).replace('localhost', socket.gethostname())
    if verbose:
        print("OMAS service uri ready at " + uri)
    daemon.requestLoop()
    return uri


omas_service_script = __file__

__all__ = ['omas_service', 'omas_service_script']


if __name__ == '__main__':
    omas_service()
