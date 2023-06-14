from __future__ import annotations

import collections.abc
import doctest
import pathlib
from typing import Any, Iterator, Literal

from Analysis.DynamicRoutingAnalysisUtils import DynRoutData


class PropertyDict(collections.abc.Mapping):
    """Dict type, where the keys are the class's properties (regular attributes
    and property getters) which don't have a leading underscore, and values are
    corresponding property values.
    
    Methods can be added and used as normal, they're just not visible in the
    dict.
    """
    
    @property
    def _properties(self) -> tuple[str, ...]:
        """Names of properties without leading underscores. No methods."""
        dict_attrs = dir(collections.abc.Mapping)
        no_dict_attrs = (attr for attr in dir(self) if attr not in dict_attrs)
        no_leading_underscore = (attr for attr in no_dict_attrs if attr[0] != '_')
        no_functions = (attr for attr in no_leading_underscore if not hasattr(getattr(self, attr), '__call__'))
        return tuple(no_functions)
    
    @property
    def _dict(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in self._properties}
    
    def __getitem__(self, key) -> Any:
        return self._dict[key]
    
    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)
         
    def __len__(self) -> int:
        return len(self._dict)
    
    def __repr__(self) -> str:
        return self._dict.__repr__()

    @property
    def _docstrings(self) -> dict[str, str]:
        """Docstrings of property getter methods (without leading underscore).
        
        Getting docstrings of regular properties/attributes isn't easy.
        """
        cls_attr = lambda attr: getattr(self.__class__, attr)
        return {
            attr: cls_attr(attr).__doc__ 
            for attr in self._properties
            if isinstance(cls_attr(attr), property)
            }
            

class TestPropertyDict(PropertyDict):
    """
    >>> TestPropertyDict()
    {'visible_property': True, 'visible_property_getter': True}
    >>> TestPropertyDict().invisible_method()
    True
    >>> TestPropertyDict()._docstrings
    {'visible_property': 'Docstring available'}
    """
    
    visible_property = True
    
    @property
    def visible_property_getter(self): 
        """Docstring available"""
        return True
    
    _invisible_property = None
    
    def invisible_method(self): 
        """Docstring not available"""
        return True
    
    
class DRTaskTrials(PropertyDict):
    
    _obj: DynRoutData
    _hdf5_file: pathlib.Path
    
    def __init__(self, hdf5_file_or_obj: DynRoutData | str | pathlib.Path, *args, **kwargs):
        """Provide path to hdf5 file from session, or an instance of `DynRoutData`"""
        if isinstance(hdf5_file_or_obj, DynRoutData):
            self._obj = hdf5_file_or_obj
            if not self._is_data_loaded:
                raise ValueError(f'{self._obj} has not been loaded, and no filepath is available to load')
            self._hdf5_file = pathlib.Path(self._obj.behavDataPath)
        else:
            self._obj = DynRoutData()
            self._hdf5_file = pathlib.Path(hdf5_file_or_obj)
        self.ensure_data()
    
    @property
    def _is_data_loaded(self) -> bool:
        return hasattr(self._obj, 'behavDataPath')
    
    def ensure_data(self) -> None:
        """Load from hdf5, if not yet loaded"""
        if not self._is_data_loaded:
            self._obj.loadBehavData(self._hdf5_file)

    @property
    def start_time(self) -> float:
        return self._obj.startTime # TODO

if __name__ == "__main__":
    doctest.testmod()
    
    x = DRTaskTrials(
        "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_644864_20230201/DynamicRouting1_644864_20230201_124332.hdf5"
        )
    x._docstrings