"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-11-03 21:52:56
❤LastEditTime: 2022-11-04 09:20:57
❤Github: https://github.com/MilknoCandy
"""
from typing import Optional


def combine_deepest_dict(d: dict, mod: list, s: Optional[str]=None):
    for k, v in d.items():
        if isinstance(v, dict):
            co = '.'.join((s,k)) if s is not None else k
            mod = combine_deepest_dict(v, mod, co)
        else:
            m = '.'.join((s,k,v)) if s is not None else '.'.join((k,v))
            mod.append(m)
    return mod

if __name__ == "__main__":
    nested_dict = {'nest1': {'nest2': {'nest3': {'val': 'tt', 'val2':'ts'}}, 'unknown_key': 'val', 'unknown_key2': 'val'}}

    print(combine_deepest_dict(nested_dict, []))