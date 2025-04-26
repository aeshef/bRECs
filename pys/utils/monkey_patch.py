# monkey_patch.py
import inspect
import collections

def apply_patch():
    if not hasattr(inspect, 'getargspec'):
        ArgSpec = collections.namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])
        inspect.ArgSpec = ArgSpec
        
        def getargspec(func):
            sig = inspect.signature(func)
            parameters = sig.parameters
            args = []
            varargs = None
            keywords = None
            defaults = []
            
            for name, param in parameters.items():
                if param.kind == inspect.Parameter.POSITIONAL_ONLY or param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    args.append(name)
                    if param.default != inspect.Parameter.empty:
                        defaults.append(param.default)
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    varargs = name
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    keywords = name
                    
            if not defaults:
                defaults = None
                
            return ArgSpec(args=args, varargs=varargs, keywords=keywords, defaults=defaults)
        
        inspect.getargspec = getargspec
        print("Monkey patch applied: inspect.getargspec added")
