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

def ods_sample():
    #ods=omas(location='equilibrium.time_slice.0')

    ods=omas()
    print(len(ods['equilibrium'].structure))
    print(len(ods['equilibrium']['time_slice'].structure))
    print(len(ods['equilibrium']['time_slice'][0]['boundary'].structure))

    #ods['boundary']['x_point'][0]['r']=1

    #print(ods['equilibrium']['time_slice'][0]['boundary']['x_point'][0]['r'].structure)
    #ods=omas(location='equilibrium.time_slice.0')
    #print(ods['boundary']['x_point'][0]['r'].structure)

#------------------------------
if __name__ == '__main__':

    ods_sample()