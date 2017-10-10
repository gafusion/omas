from __future__ import absolute_import, print_function, division, unicode_literals

from omas_utils import *

def _omas_key_dict_preprocessor(key):
    '''
    converts a omas string path to a list of keys that make the path

    :param key: omas string path

    :return: list of keys that make the path
    '''
    if not isinstance(key,(list,tuple)):
        key=str(key)
        key=re.sub('\]','',re.sub('\[','.',key)).split('.')
    else:
        key=map(str,key)
    try:
        key[0]=int(key[0])
    except ValueError:
        pass
    return key

class omas(dict):
    '''
    OMAS class
    '''
    def __new__(cls, *args, **kw):
        instance = dict.__new__(cls, *args, **kw)
        instance.imas_version=None
        instance.name=''
        instance.parent=None
        instance.structure={}
        return instance

    def __init__(self, location='', imas_version=None):
        '''
        :param imas_version: IMAS version to use as a constrain for the nodes names
        '''
        if imas_version is None:
            imas_version=os.path.split(sorted(glob.glob(imas_json_dir+os.sep+'*'))[-1])[-1]
            printd('OMAS class instantiated with IMAS version: '+imas_version)
        self.imas_version=imas_version

    @property
    def location(self):
        h=self
        location=''
        if not hasattr(h,'name'):
            pass
        while str(h.name):
            location='.'.join(filter(None,[str(h.name),location]))
            h=h.parent()
            if h is None:
                break
        return location

    def __setitem__(self, key, value):
        #handle individual keys as well as full paths
        key=_omas_key_dict_preprocessor(key)

        #if the user has entered path rather than a single key
        if len(key)>1:
            pass_on_value=value
            value=omas(imas_version=self.imas_version)

        #if this is the head
        if not self.location:
            self.structure=load_structure(key[0].split(separator)[0])

        #consistency checking
        location='.'.join(filter(None,[self.location,str(key[0])]))
        structure={}
        structure_location=re.sub('\.[0-9]+','[:]',location)
        for item in self.structure.keys():
            if item.startswith(structure_location):
                structure[item]=self.structure[item]
        if not len(structure):
            print(self.location)
            print(re.sub('\.[0-9]+','.:',self.location))
            options=numpy.unique(map(lambda x:re.sub('\[:\]','.:',x)[len(re.sub('\.[0-9]+','.:',self.location))+1:].split('.')[0],self.structure))
            if len(options)==1 and options[0]==':':
                options='A numerical index is needed'
            else:
                options='Did you mean: %s'%options
            spaces='           '+' '*(len(self.location)+1)
            raise(Exception('`%s` is not a valid IMAS location\n'%location+spaces+'^\n'+spaces+'%s'%options))

        #if the value is a dictionary structure
        if isinstance(value,omas):
            old_name=str(getattr(value,'name',''))
            value.name=key[0]
            #deepcopy necessary to keep the location straight
            if old_name and old_name!=key[0]:
                try:
                    value1=copy.deepcopy(value)
                except Exception:
                    raise
                finally:
                    value.name=old_name
                value=value1
            value.parent=weakref.ref(self)
            value.structure=structure

        #if the user has entered path rather than a single key
        if len(key)>1:
            if key[0] not in self:
                dict.__setitem__(self, key[0], value)
            return self[key[0]].__setitem__('.'.join(key[1:]), pass_on_value)
        else:
            return dict.__setitem__(self, key[0], value)

    def __getitem__(self, key):
        #handle individual keys as well as full paths
        key=_omas_key_dict_preprocessor(key)

        #dynamic path creation
        if key[0] not in self:
            self.__setitem__(key[0], omas(imas_version=self.imas_version))

        if len(key)>1:
            #if the user has entered path rather than a single key
            return dict.__getitem__(self, key[0])['.'.join(key[1:])]
        else:
            return dict.__getitem__(self, key[0])

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

    def flat(self):
        '''
        :return: flat dictionary representation of the data
        '''
        tmp={}
        for path in self.paths():
            tmp['.'.join(map(str,path))]=self[path]
        return tmp

    def __getstate__(self):
        #switching between weak/strong reference for .parent attribute
        state = self.__dict__.copy()
        if state['parent'] is not None:
            state['parent'] = state['parent']()
        return state

    def __setstate__(self, state):
        #switching between weak/strong reference for .parent attribute
        self.__dict__ = state.copy()
        if self.__dict__['parent'] is not None:
            self.__dict__['parent'] = weakref.ref(self.__dict__['parent'])

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

    ods['equilibrium.time_slice.1.time']=2000.
    ods['equilibrium.time_slice.1.global_quantities.ip']=2.
    ods['equilibrium.time_slice[2].time']=3000.
    ods['equilibrium.time_slice[2].global_quantities.ip']=3.
    print(ods['equilibrium.time_slice']['1.global_quantities.ip'])
    print(ods[['equilibrium','time_slice',1,'global_quantities','ip']])
    print(ods[('equilibrium','time_slice','1','global_quantities','ip')])

    pprint(ods.paths())
    pprint(ods2.paths())

    tmp=pickle.dumps(ods)
    ods=pickle.loads(tmp)

    save_omas_pkl(ods,'test.pkl')
    ods=load_omas_pkl('test.pkl')

    tmp=ods.flat()
    pprint(tmp)

    return ods

def save_omas_pkl(ods, filename, **kw):
    with open(filename,'w') as f:
        pickle.dump(ods,f,**kw)

def load_omas_pkl(filename):
    with open(filename,'r') as f:
        return pickle.load(f)

from omas_imas import *
from omas_s3 import *
#------------------------------
if __name__ == '__main__':

    ods=ods_sample()
