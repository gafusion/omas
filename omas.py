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
        self.location=''
        self.structure={}
        if location:
            h=self
            for step in location.split(separator):
                h=h[step]
            self.location=h.location
            self.structure=h.structure

    def __setitem__(self, key, value):
        #if this is the head
        if not self.location:
            self.structure=load_structure(key.split(separator)[0])

        #consistency checking
        if isinstance(value,omas):
            value.location='.'.join(filter(None,[self.location,str(key)]))
            structure={}
            structure_location=re.sub('\.[0-9]+','[:]',value.location)
            for item in self.structure.keys():
                if item.startswith(structure_location):
                    structure[item]=self.structure[item]
            if not len(structure):
                raise(Exception('`%s` is not a valid IMAS location'%value.location))
            value.structure=structure

        return dict.__setitem__(self, key, value)

    def __getitem__(self, key):
        #dynamic path creation
        if key not in self:
            self.__setitem__(key, omas(imas_version=self.imas_version))
        return dict.__getitem__(self, key)

    def traverse(self, **kw):
        '''
        traverse the hierarchy and returns paths that have data

        :return: list of paths that have data
        '''
        paths=kw.setdefault('paths',[])
        path=kw.setdefault('path',[])
        for kid in self.keys():
            if isinstance(self[kid], omas):
                self[kid].traverse(paths=paths,path=path+[kid])
            else:
                paths.append(path+[kid])
        return paths

    def get(self, path):
        '''
        get data from path

        :param path: path in the ods hierarchy

        :return: data at path in ods hierarchy
        '''
        h=self
        for step in path:
            h=h[step]
        return h

def ods_sample():
    ods=omas()

    ods['equilibrium']['time_slice'][0]['boundary']=5
    ods['equilibrium']['time_slice'][1]['boundary']=5

    pprint(ods.traverse())
    return ods

from omas_imas import *

#------------------------------
if __name__ == '__main__':

    ods=ods_sample()
    save_to_imas(ods)
