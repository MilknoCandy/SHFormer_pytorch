"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-11-30 19:47:12
❤LastEditTime: 2022-11-30 19:47:12
❤Github: https://github.com/MilknoCandy
"""
import importlib
import os
import pkgutil


def recursive_find_python_class(folder, trainer_name, current_module):
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            m = importlib.import_module(f'{current_module}.{modname}')
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break
    return tr