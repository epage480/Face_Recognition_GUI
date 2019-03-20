# Working example for my blog post at:
# https://danijar.com/structuring-your-tensorflow-models/
# https://danijar.github.io/structuring-your-tensorflow-models
import functools

"""
A decorator for functions that define TensorFlow operations. The wrapped
function will only be executed once. Subsequent calls to it will directly
return the result so that operations are added to the graph only once.
"""
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator