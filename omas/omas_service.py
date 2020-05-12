from omas.omas_core import dynamic_ODS_factory
import Pyro5.server

daemon = Pyro5.api.Daemon(port=39921)
uri = daemon.register(dynamic_ODS_factory,'dynamic_ODS_factory')

print("Ready. Object uri =", uri)
daemon.requestLoop()
