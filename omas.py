from __future__ import absolute_import, print_function, division, unicode_literals

from omas_utils import *

class omas(dict):
    '''
    OMAS class
    '''
    def __init__(self, location='', imas_version=None):
        '''
        :param imas_version: IMAS version to use as a constrain for the nodes names
        '''
        if imas_version is None:
            imas_version=os.path.split(sorted(glob.glob(imas_json_dir+os.sep+'*'))[-1])[-1]
            printd('OMAS class instantiated with IMAS version: '+imas_version)
        self.imas_version=imas_version
        self.name=''
        self.parent=None
        self.structure={}

    @property
    def location(self):
        h=self
        location=''
        while str(h.name):
            location='.'.join(filter(None,[str(h.name),location]))
            h=h.parent()
            if h is None:
                break
        return location

    def __setitem__(self, key, value):
        #if this is the head
        if not self.location:
            self.structure=load_structure(key.split(separator)[0])

        #consistency checking
        location='.'.join(filter(None,[self.location,str(key)]))
        structure={}
        structure_location=re.sub('\.[0-9]+','[:]',location)
        for item in self.structure.keys():
            if item.startswith(structure_location):
                structure[item]=self.structure[item]
        if not len(structure):
            raise(Exception('`%s` is not a valid IMAS location'%location))

        if isinstance(value,omas):
            old_name=str(getattr(value,'name',''))
            value.name=key
            #deepcopy necessary to keep the location straight
            if old_name and old_name!=key:
                try:
                    value1=copy.deepcopy(value)
                except Exception:
                    raise
                finally:
                    value.name=old_name
                value=value1
            value.parent=weakref.ref(self)
            value.structure=structure

        return dict.__setitem__(self, key, value)

    def __getitem__(self, key):
        if key not in self: #dynamic path creation
            self.__setitem__(key, omas(imas_version=self.imas_version))
        return dict.__getitem__(self, key)

    def paths(self, **kw):
        '''
        Traverse the ods and return paths that have data

        :return: list of paths that have data
        '''
        paths=kw.setdefault('paths',[])
        path=kw.setdefault('path',[])
        for kid in self.keys():
            if isinstance(self[kid], omas):
                self[kid].paths(paths=paths,path=path+[kid])
            else:
                paths.append(path+[kid])
        return paths

    def get(self, path):
        '''
        Get data from path

        :param path: path in the ods

        :return: data at path in ods
        '''
        h=self
        for step in path:
            h=h[step]
        return h

def ods_sample():
    ods=omas()
    ods['equilibrium']['time_slice'][0]['time']=1000.
    ods['equilibrium']['time_slice'][0]['global_quantities']['ip']=1.5

    # issue with x_point structure?
    if False:
        ods['equilibrium']['time_slice'][1]['time']=2000.
        ods['equilibrium']['time_slice'][1]['boundary']['x_point'][0]['z']=0.

    ods2=omas()
    ods2['equilibrium']['time_slice'][2]=ods['equilibrium']['time_slice'][0]

    print(ods['equilibrium']['time_slice'][0]['global_quantities'].location)
    print(ods['equilibrium']['time_slice'][2]['global_quantities'].location)

    pprint(ods.paths())
    pprint(ods2.paths())
    return ods

from omas_imas import *

#------------------------------
if __name__ == '__main__':

    ods=ods_sample()

