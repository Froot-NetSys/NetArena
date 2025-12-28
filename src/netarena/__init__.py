import importlib
import pkgutil


# Import all immediate submodules.
for mod_info in pkgutil.iter_modules(__path__, __name__ + '.'):
    mod = importlib.import_module(mod_info.name)
