from __future__ import print_function, division, unicode_literals

from .omas_utils import *

__all__=['omas_rcparams',
         'omas',                     'ods_sample',            'different_ods',
         'save_omas',                'load_omas',             'test_omas_suite',
         'save_omas_pkl',            'load_omas_pkl',         'test_omas_pkl',
         'save_omas_json',           'load_omas_json',        'test_omas_json',
         'save_omas_nc',             'load_omas_nc',          'test_omas_nc',
         'save_omas_imas',           'load_omas_imas',        'test_omas_imas',
         'save_omas_s3',             'load_omas_s3',          'test_omas_s3',
         'aggregate_imas_html_docs', 'create_json_structure', 'create_html_documentation',
         'imas_json_dir',            'default_imas_version'
         ]

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
        key=list(map(str,key))
    try:
        key[0]=int(key[0])
    except ValueError:
        pass
    return key

class omas(dict):
    '''
    OMAS class
    '''
    def __new__(cls, consistency_check=True, *args, **kw):
        instance = dict.__new__(cls, *args, **kw)
        instance.imas_version=None
        instance.name=''
        instance.parent=None
        instance.structure={}
        instance._consistency_check=consistency_check
        return instance

    def __init__(self, *args, **kw):
        '''
        :param imas_version: IMAS version to use as a constrain for the nodes names

        :param consistency_check: whether to enforce consistency with IMAS schema
        '''
        imas_version=kw.pop('imas_version',None)
        if imas_version is None:
            imas_version=os.path.split(sorted(glob.glob(imas_json_dir+os.sep+'*'))[-1])[-1]
        self.imas_version=re.sub('_','.',imas_version)

    @property
    def consistency_check(self):
        '''
        property that sets whether consistency with IMAS schema is enabled or not

        :return: True/False
        '''
        return self._consistency_check

    @consistency_check.setter
    def consistency_check(self,value):
        self._consistency_check=value
        for item in self:
            if isinstance(self[item],omas):
                self[item].consistency_check=value

    @property
    def location(self):
        '''
        property that returns the path in the IMAS schema

        :return: string with location of current object
        '''
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

        structure={}
        #if structural checks are enabled
        if self.consistency_check:
            #if this is the head
            if not self.location:
                self.structure=load_structure(key[0].split(separator)[0])

            #consistency checking
            location='.'.join(filter(None,[self.location,str(key[0])]))
            structure_location=re.sub('\.[0-9:]+','[:]',location)
            for item in self.structure.keys():
                if item.startswith(structure_location):
                    structure[item]=self.structure[item]
            if not len(structure):
                options=numpy.unique(list(map(lambda x:re.sub('\[:\]','.:',x)[len(re.sub('\.[0-9]+','.:',self.location))+1:].split('.')[0],self.structure)))
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

        #data slicing
        if key[0]==':':
            data=[]
            for k in self.keys():
                data.append( self['.'.join([str(k)]+key[1:])] )
            return numpy.array(data)

        #dynamic path creation
        elif key[0] not in self:
            self.__setitem__(key[0], omas(imas_version=self.imas_version))

        if len(key)>1:
            #if the user has entered path rather than a single key
            return dict.__getitem__(self, key[0])['.'.join(key[1:])]
        else:
            return dict.__getitem__(self, key[0])

    def __delitem__(self, key):
        #handle individual keys as well as full paths
        key=_omas_key_dict_preprocessor(key)
        print(key)
        if len(key)>1:
            #if the user has entered path rather than a single key
            del self[key[0]]['.'.join(key[1:])]
        else:
            return dict.__delitem__(self, key[0])

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

    def __deepcopy__(self,memo={}):
        if hasattr(self,'_consistency_check'):
            _consistency_checkBKP=self._consistency_check
        else:
            _consistency_checkBKP=omas_rcparams['consistency_check']
        try:
            self.consistency_check=False
            tmp=pickle.loads(pickle.dumps(self,pickle.HIGHEST_PROTOCOL))
            tmp.consistency_check=_consistency_checkBKP
        finally:
            self.consistency_check=_consistency_checkBKP
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

    def __getnewargs__(self):
        return (False,)

#--------------------------------------------
# save and load OMAS with Python pickle
#--------------------------------------------
def save_omas_pkl(ods, filename, **kw):
    '''
    Save OMAS data set to Python pickle

    :param ods: OMAS data set

    :param filename: filename to save to

    :param kw: keywords passed to pickle.dump function
    '''
    printd('Saving to %s'%(filename),topic='pkl')

    with open(filename,'wb') as f:
        pickle.dump(ods,f,**kw)

def load_omas_pkl(filename):
    '''
    Load OMAS data set from Python pickle

    :param filename: filename to save to

    :returns: ods OMAS data set
    '''
    printd('Loading from %s'%(filename),topic='pkl')

    with open(filename,'rb') as f:
        return pickle.load(f)

def test_omas_pkl(ods):
    '''
    test save and load Python pickle

    :param ods: ods

    :return: ods
    '''
    filename='test.pkl'
    save_omas_pkl(ods,filename)
    ods1=load_omas_pkl(filename)
    return ods1

#--------------------------------------------
# tools
#--------------------------------------------
def ods_sample():
    '''
    create sample ODS data
    :return:
    '''
    ods=omas()

    #info ODS is used for keeping track of IMAS metadata
    ods['info.user']=unicode(os.environ['USER'])
    ods['info.tokamak']='ITER'
    ods['info.imas_version']=unicode(os.environ.get('IMAS_VERSION','3.10.1'))
    ods['info.shot']=1
    ods['info.run']=0

    ods['equilibrium']['time_slice'][0]['time']=1000.
    ods['equilibrium']['time_slice'][0]['global_quantities']['ip']=1.5

    # issue with x_point structure?
    if False:
        ods['equilibrium']['time_slice'][1]['time']=2000.
        ods['equilibrium']['time_slice'][1]['boundary']['x_point'][0]['z']=0.

    ods2=omas()
    ods2['equilibrium']['time_slice'][2]=ods['equilibrium']['time_slice'][0]

    printd(ods['equilibrium']['time_slice'][0]['global_quantities'].location,topic='sample')
    printd(ods['equilibrium']['time_slice'][2]['global_quantities'].location,topic='sample')

    ods['equilibrium.time_slice.1.time']=2000.
    ods['equilibrium.time_slice.1.global_quantities.ip']=2.
    ods['equilibrium.time_slice[2].time']=3000.
    ods['equilibrium.time_slice[2].global_quantities.ip']=3.

    #check different ways of addressing data
    printd(ods['equilibrium.time_slice']['1.global_quantities.ip'],topic='sample')
    printd(ods[['equilibrium','time_slice',1,'global_quantities','ip']],topic='sample')
    printd(ods[('equilibrium','time_slice','1','global_quantities','ip')],topic='sample')
    printd(ods['equilibrium.time_slice.1.global_quantities.ip'],topic='sample')
    printd(ods['equilibrium.time_slice[1].global_quantities.ip'],topic='sample')

    ods['equilibrium.time_slice.0.profiles_1d.psi']=numpy.linspace(0,1,10)

    #pprint(ods.paths())
    #pprint(ods2.paths())

    #check data slicing is working
    printd(ods['equilibrium.time_slice[:].global_quantities.ip'],topic='sample')

    ckBKP=ods.consistency_check
    tmp=pickle.dumps(ods)
    ods=pickle.loads(tmp)
    if ods.consistency_check!=ckBKP:
        raise(Exception('consistency_check attribute changed'))

    save_omas_pkl(ods,'test.pkl')
    ods=load_omas_pkl('test.pkl')

    tmp=ods.flat()
    #pprint(tmp)

    return ods

def different_ods(ods1, ods2):
    '''
    Checks if two ODSs have any difference and returns the string with the cause of the different

    :param ods1: first ods to check

    :param ods2: second ods to check

    :return: string with reason for difference, or False otherwise
    '''
    ods1=ods1.flat()
    ods2=ods2.flat()

    k1=set(ods1.keys())
    k2=set(ods2.keys())
    for k in k1.difference(k2):
        return 'DIFF: key `%s` missing in 2nd ods'%k
    for k in k2.difference(k1):
        return 'DIFF: key `%s` missing in 1st ods'%k
    for k in k1.intersection(k2):
        if type(ods1[k])!=type(ods2[k]):
            return 'DIFF: `%s` differ in type (%s,%s)'%(k,type(ods1[k]),type(ods2[k]))
        elif isinstance(ods1[k],basestring):
            if ods1[k]!=ods2[k]:
                return 'DIFF: `%s` differ in value'%k
        else:
            if not numpy.allclose(ods1[k],ods2[k]):
                return 'DIFF: `%s` differ in value'%k
    return False

_tests=['pkl','json','nc','s3','imas']
def test_omas_suite(test_type=None):
    '''
    :param test_type: None tests all suite, otherwise choose among %s
    '''

    ods=ods_sample()

    if test_type in _tests:
        os.environ['OMAS_DEBUG_TOPIC']=test_type
        ods1=globals()['test_omas_'+test_type](ods)
        check=different_ods(ods, ods1)
        if not check:
            print('OMAS data got saved and loaded correctly')
        else:
            print(check)

    else:
        print('='*20)

        os.environ['OMAS_DEBUG_TOPIC']='*'

        results=numpy.zeros((len(_tests),len(_tests)))

        for k1,t1 in enumerate(_tests):
            failed1=False
            try:
                ods1=globals()['test_omas_'+t1](ods)
            except Exception as _excp:
                failed1=True
            for k2,t2 in enumerate(_tests):
                try:
                    if failed1:
                        raise
                    ods2=globals()['test_omas_'+t2](ods1)

                    different=different_ods(ods1,ods2)
                    if not different:
                        print('FROM %s TO %s : OK'%(t1.center(5),t2.center(5)))
                        results[k1,k2]=1.0
                    else:
                        print('FROM %s TO %s : NO --> %s'%(t1.center(5),t2.center(5),different))
                        results[k1,k2]=-1.0

                except Exception as _excp:
                    print('FROM %s TO %s : NO --> %s'%(t1.center(5),t2.center(5),repr(_excp)))

        print('='*20)
        print(results.astype(int))
        print('='*20)
test_omas_suite.__doc__=test_omas_suite.__doc__%_tests

#--------------------------------------------
# save and load OMAS with default saving method
#--------------------------------------------
def save_omas(ods, filename):
    '''
    Save omas data to filename. The file extension defines format to use.

    :param ods: OMAS data set

    :param filename: filename to save to
    '''
    if os.path.splitext(filename)[1].lower()=='.json':
        return save_omas_json(ods,filename)
    elif os.path.splitext(filename)[1].lower()=='.nc':
        return save_omas_nc(ods,filename)
    else:
        return save_omas_pkl(ods,filename)

def load_omas(filename):
    '''
    Load omas data from filename. The file extension defines format to use.

    :param filename: filename to load from

    :returns: ods OMAS data set
    '''
    if os.path.splitext(filename)[1].lower()=='.json':
        return load_omas_json(filename)
    elif os.path.splitext(filename)[1].lower()=='.nc':
        return load_omas_nc(filename)
    else:
        return load_omas_pkl(filename)

#--------------------------------------------
# import other omas tools and methods in this namespace
#--------------------------------------------
from .omas_imas import *
from .omas_s3 import *
from .omas_nc import *
from .omas_json import *
from .omas_structure import *

#--------------------------------------------
if __name__ == '__main__':
    test_omas_suite()
