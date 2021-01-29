import re


def p2l(key):
    """
    Converts the many different ways of addressing an ODS path to a list of keys ['bla',0,'bla']

    :param key: ods location in some format

    :return: list of keys that make the ods path
    """
    if isinstance(key, list):
        return key

    if isinstance(key, tuple):
        return list(key)

    if isinstance(key, int):
        return [int(key)]

    if isinstance(key, str) and not ('.' in key or '[' in key):
        if len(key):
            return [key]
        else:
            return []

    if key is None:
        raise TypeError('OMAS key cannot be None')

    if isinstance(key, dict):
        raise TypeError('OMAS key cannot be of type dictionary')

    if not isinstance(key, (list, tuple)):
        key = str(key).replace('[', '.').replace(']', '').split('.')

    key = [k for k in key if k]
    for k, item in enumerate(key):
        try:
            key[k] = int(item)
        except ValueError:
            pass

    return key


def l2i(path):
    """
    Formats a list ['bla',0,'bla'] into a IMAS path 'bla[0].bla'

    :param path: ODS path format

    :return: IMAS path format
    """
    ipath = ''
    for kstep, step in enumerate(path):
        if isinstance(step, int) or step == ':':
            ipath += "[%s]" % step
        elif kstep == 0:
            ipath += '%s' % step
        else:
            ipath += '.%s' % step
    return ipath


def l2o(path):
    """
    Formats a list ['bla',0,'bla'] into an ODS path format 'bla.0.bla'

    :param path: list of strings and integers

    :return: ODS path format
    """
    return '.'.join(filter(None, map(str, path)))


_o2u_pattern = re.compile(r'\.[0-9:]+')
_o2u_pattern_no_split = re.compile(r'^[0-9:]+')
_o2i_pattern = re.compile(r'\.([:0-9]+)')


def o2u(path):
    """
    Converts an ODS path 'bla.0.bla' into a universal path 'bla.:.bla'

    :param path: ODS path format

    :return: universal ODS path format
    """
    if '.' in path:
        return re.sub(_o2u_pattern, '.:', path)
    else:
        return re.sub(_o2u_pattern_no_split, ':', path)


_o2i_pattern = re.compile(r'\.([:0-9]+)')


def i2o(path):
    """
    Formats a IMAS path 'bla[0].bla' into an ODS path 'bla.0.bla'

    :param path: IMAS path format

    :return: ODS path format
    """
    return path.replace(']', '').replace('[', '.')


def o2i(path):
    """
    Formats a ODS path 'bla.0.bla' into an IMAS path 'bla[0].bla'

    :param path: ODS path format

    :return: IMAS path format
    """
    return re.sub(_o2i_pattern, r'[\1]', path)


def u2o(upath, path):
    """
    Replaces `:` and integers in `upath` with ':' and integers from in `path`
    e.g. uo2('a.:.b.:.c.1.d.1.e','f.:.g.1.h.1.i.:.k')) becomes 'bla.1.hello.2.bla'

    :param upath: universal ODS path

    :param path: ODS path

    :return: filled in ODS path
    """
    if upath.startswith('1...'):
        return upath
    ul = p2l(upath)
    ol = p2l(path)
    for k in range(min([len(ul), len(ol)])):
        if (ul[k] == ':' or isinstance(ul[k], int)) and (ol[k] == ':' or isinstance(ol[k], int)):
            ul[k] = ol[k]
        elif ul[k] == ol[k]:
            continue
        else:
            break
    return l2o(ul)


def u2n(upath, list_of_n):
    """
    Replaces `:` and integers in `upath` with integers provided
    e.g. uo2('a.:.b.:.c.1.d.1.e',[2,3]) becomes 'a.2.b.3.c.1.d.1.e'

    :param upath: universal ODS path

    :param list_of_n: list of numbers

    :return: filled in ODS path
    """
    ul = p2l(upath)
    i = 0
    for k in range(len(ul)):
        if ul[k] == ':' or isinstance(ul[k], int):
            if ul[k] == ':' and i < len(list_of_n):
                ul[k] = list_of_n[i]
            i += 1
    return l2o(ul)


def l2u(path):
    """
    Formats a list ['bla',0,'bla'] into a universal path 'bla.:.bla'
    NOTE: a universal ODS path substitutes lists indices with :

    :param path: list of strings and integers

    :return: universal ODS path format
    """
    return o2u(l2o(path))


def trim_common_path(p1, p2):
    """
    return paths in lists format trimmed of the common first path between paths p1 and p2

    :param p1: ODS path

    :param p2: ODS path

    :return: paths in list format trimmed of common part
    """
    p1 = p2l(p1)
    p2 = p2l(p2)
    both = [x if x[0] == x[1] else None for x in zip(p1, p2)] + [None]
    return p1[both.index(None) :], p2[both.index(None) :]
