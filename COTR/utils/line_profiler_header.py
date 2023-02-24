import builtins

def profile(func):
    return func

try:
    profile = builtins.profile
except AttributeError:
    pass
