import cProfile
import inspect
import pstats
from operator import itemgetter

__all__ = [
    'map', 'filter', 'zip',
    'old_map', 'old_filter', 'old_zip',
    'profile',
]

old_filter = filter
filter = lambda x, y: list(old_filter(x, y))
old_zip = zip
zip = lambda x, y: list(old_zip(x, y))
old_map = map
map = lambda x, y: list(old_map(x, y))


def profile(fn, params, kwparams={}, sort=[]):
    # FullArgSpec(args=[], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={})
    fn_args = inspect.getfullargspec(fn)
    pos_args = [(fn_args.args[i], param) for i, param in enumerate(params)]
    kwparams = list(kwparams.items())

    globals_dict = {param: arg for param, arg in pos_args + kwparams}
    globals_dict[fn.__name__] = fn

    cmd = f'{fn.__name__}({",".join(map(itemgetter(0), pos_args + kwparams))})'
    cProfile.runctx(cmd, globals=globals_dict, locals={}, filename=f"{__file__}.profile")

    s = pstats.Stats("{}.profile".format(__file__))
    s.strip_dirs()
    s.sort_stats(*sort).print_stats()
