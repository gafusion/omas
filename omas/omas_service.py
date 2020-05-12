from omas.omas_core import dynamic_ODS_factory
import Pyro5.api

daemon = Pyro5.api.Daemon()
uri = daemon.register(dynamic_ODS_factory)

print("Ready. Object uri =", uri)
daemon.requestLoop()
