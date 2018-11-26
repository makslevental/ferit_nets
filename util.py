import cProfile
import pstats
import random, string
from operator import itemgetter

__all__ = [
    'map', 'filter', 'zip',
    'profile',
    'DEBUG'
]

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


old_filter = filter
filter = lambda x, y: list(old_filter(x, y))
old_zip = zip
zip = lambda x, y: list(old_zip(x, y))
old_map = map
map = lambda x, y: list(old_map(x, y))

DEBUG = False


def profile(fn, args, kwargs):
    # FullArgSpec(args=[], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={})
    # fn_args = inspect.getfullargspec(fn)
    pos_args = [(randomword(5), arg) for arg in args]
    kwargs = list(kwargs.items())

    globals_dict = {param: arg for param, arg in pos_args + kwargs}
    globals_dict[fn.__name__] = fn

    cmd = f'{fn.__name__}({",".join(map(itemgetter(0), pos_args + kwargs))})'
    cProfile.runctx(cmd, globals=globals_dict, locals={}, filename=f"{__file__}.profile")

    s = pstats.Stats("{}.profile".format(__file__))
    s.strip_dirs()
    s.sort_stats('module', 'cumtime').print_stats()
