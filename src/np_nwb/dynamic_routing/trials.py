from __future__ import annotations
import contextlib

import datetime
import doctest
import functools
import pathlib
import collections.abc
import reprlib
from typing import Any, Generator, Iterable, Iterator, Literal, NamedTuple, Optional, Sequence
import warnings

import numpy as np
import pandas as pd
import np_session
import np_logging
import pynwb
import allensdk.brain_observatory.sync_dataset as sync_dataset
from Analysis.DynamicRoutingAnalysisUtils import DynRoutData

import np_nwb.utils as utils

logger = np_logging.getLogger(__name__)

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
        return reprlib.repr(self._dict)

    def to_add_trial_columns(
        self
    ) -> Generator[dict[Literal['name', 'description'], str], None, None]:
        """Name and description for each trial column.
        
        Iterate over result and unpack each dict:
        
        >>> for column in obj.to_add_trial_columns(): # doctest: +SKIP
        ...    nwb_file.add_trial_column(**column)

        """
        attrs = self._properties
        descriptions = self._docstrings
        missing = tuple(column for column in attrs if column not in descriptions)
        if any(missing):
           raise UserWarning(f'These properties do not have descriptions (add docstrings to their property getters): {missing}')
        descriptions.update(dict(zip(missing, ('' for _ in missing))))
        return ({'name': name, 'description': description} for name, description in descriptions.items())
    
    def to_add_trials(self) -> Generator[dict[str, int | float | str | datetime.datetime], None, None]:
        """Column name and value for each trial.
            
        Iterate over result and unpack each dict:
        
        >>> for trial in obj.to_add_trials(): # doctest: +SKIP
        ...    nwb_file.add_trials(**trial)
        
        """
        # for trial in self.to_dataframe().itertuples(index=False):
        for trial in self.to_dataframe().iterrows():
            yield dict(trial[1])
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(data=dict(self))
    
    @property
    def _docstrings(self) -> dict[str, str]:
        """Docstrings of property getter methods that have no leading
        underscore in their name.
        
        - getting the docstring of a regular property/attribute isn't easy:
        if we query its docstring in the same way as a property getter/method,
        we'll just receive the docstring for its value's type.
        """
        cls_attr = lambda attr: getattr(self.__class__, attr)
        regular_properties = {attr: "" for attr in self._properties if not isinstance(cls_attr(attr), property)}
        property_getters = {attr: cls_attr(attr).__doc__ or "" for attr in self._properties if isinstance(cls_attr(attr), property)}
        return {
            attr: cls_attr(attr).__doc__ or "" # if no docstring present, __doc__ is None
            for attr in property_getters
        }
            

class TestPropertyDict(PropertyDict):
    """
    >>> TestPropertyDict()
    {'no_docstring': True, 'visible_property': True, 'visible_property_getter': True}
    
    >>> TestPropertyDict().invisible_method()
    True
    
    >>> TestPropertyDict()._docstrings
    {'no_docstring': '', 'visible_property_getter': 'Docstring available'}
    """
    
    visible_property = True
    
    @property
    def visible_property_getter(self): 
        """Docstring available"""
        return True

    @property
    def no_docstring(self): 
        return True
    
    _invisible_property = None
    
    def invisible_method(self): 
        """Docstring not available"""
        return True

class TestPropertyDictExports(PropertyDict):
    """
    >>> TestPropertyDictExports()
    {'start_time': [1.0, 2.0], 'stop_time': [1.5, 2.5]}
    
    >>> for kwargs in TestPropertyDictExports().to_add_trials():
    ...     print(kwargs)
    {'start_time': 1.0, 'stop_time': 1.5}
    {'start_time': 2.0, 'stop_time': 2.5}
    
    >>> for kwargs in TestPropertyDictExports().to_add_trial_columns():
    ...     print(kwargs)
    {'name': 'start_time', 'description': 'Start of trial'}
    {'name': 'stop_time', 'description': 'End of trial'}
    """
    @property
    def start_time(self) -> Sequence[float]:
        "Start of trial"
        return [1.0, 2.0]

    @property
    def stop_time(self) -> Sequence[float]:
        "End of trial"
        return [start + 0.5 for start in self.start_time]
    
class DRDataLoader:
    """Class for finding required raw data from a DRpilot session.
    """
    
    session: np_session.DRPilotSession
    
    def __init__(self, session: str | pathlib.Path | np_session.Session):
        """Provide a folder path or `np_session.Session` input arg"""
        self.session = np_session.DRPilotSession(str(session))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.task_hdf5})'
    
    @property
    def sam(self) -> DynRoutData | None:
        """Sam's analysis object using data from an hdf5 file.
        
        Loading behav data may fail due to missing keys if not an ecephys experiment.
        """
        with contextlib.suppress(AttributeError):
            return self._sam
               
        obj = DynRoutData()
        self._sam = None
        with contextlib.suppress(Exception):
            obj.loadBehavData(self.task_hdf5)
            self._sam = obj
            
        return self.sam
    
    @property
    def task_hdf5(self) -> pathlib.Path:
        return next(file for file in self.session.hdf5s if 'DynamicRouting' in file.stem)
    
    @property
    def rf_hdf5(self) -> pathlib.Path | None:
        return next((file for file in self.session.hdf5s if 'RFMapping' in file.stem), None)
    
    @property
    def sync(self) -> sync_dataset.Dataset | None:
        with contextlib.suppress(AttributeError):
            return self._sync
        if self.session.sync:
            self._sync = sync_dataset.Dataset(self.session.sync)
        else:
            self._sync = None
        return self.sync
    
    class Frametimes(NamedTuple):
        main: Sequence[float]
        rf: Sequence[float] | None
        
    @property
    def frametime_blocks(self) -> Frametimes:
        """Blocks of adjusted diode times from sync: one block per stimulus.
        
        Not available (or needed) for experiments without sync.
        
        Assumes task was the longest block of diode
        stim events (RFmapping / opto are shorter)
        """
        if self.sync is None:
            raise AttributeError(f'Cannot get sync file for {self}')
        
        with contextlib.suppress(AttributeError):
            return self._frametime_blocks
        
        vsync_rising_edges: Sequence[float] = self.sync.get_rising_edges('vsync_stim', units = 'seconds')
        vsync_falling_edges: Sequence[float] = self.sync.get_falling_edges('vsync_stim', units = 'seconds')
        all_vsync_times = vsync_falling_edges if vsync_rising_edges[0] < vsync_falling_edges[0] else vsync_falling_edges[1:]
        
        vsync_times_in_blocks = [] 
        for vsyncs in utils.reshape_timestamps_into_blocks(all_vsync_times):
            long_interval_threshold = np.median(np.diff(vsyncs)) + 3 * np.std(np.diff(vsyncs))
            while np.diff(vsyncs)[0] > long_interval_threshold:
                vsyncs = vsyncs[1:]
                
            while np.diff(vsyncs)[-1] > long_interval_threshold:
                vsyncs = vsyncs[:-1]
            vsync_times_in_blocks.append(vsyncs)
            
        block_lengths = tuple(len(block) for block in vsync_times_in_blocks)
        main_stim_block_idx = block_lengths.index(max(block_lengths))
        rf_block_idx = None
        if len(vsync_times_in_blocks) <= 2:
            rf_block_idx = int(not main_stim_block_idx)
        else:
            logger.warning("More than 2 vsync blocks found: cannot determine which is which. Assumptions made may be incorrect.")
        
        stim_running_rising_edges: Sequence[float] = self.sync.get_rising_edges('stim_running', units = 'seconds')
        stim_running_falling_edges: Sequence[float] = self.sync.get_falling_edges('stim_running', units = 'seconds')

        
        if len(stim_running_rising_edges) and len(stim_running_falling_edges):
            if stim_running_rising_edges[0] > stim_running_falling_edges[0]:
                stim_running_falling_edges[1:]
            assert len(stim_running_rising_edges) == len(vsync_times_in_blocks)
            # TODO filter vsync blocks on stim running
            
        diode_rising_edges: Sequence[float] = self.sync.get_rising_edges('stim_photodiode', units = 'seconds')
        diode_falling_edges: Sequence[float] = self.sync.get_falling_edges('stim_photodiode', units = 'seconds')
        assert abs(len(diode_rising_edges) - len(diode_falling_edges)) < 2
        
        diode_rising_edges_in_blocks = utils.reshape_timestamps_into_blocks(diode_rising_edges)
        diode_falling_edges_in_blocks = utils.reshape_timestamps_into_blocks(diode_falling_edges)
        
        diode_times_in_blocks = []
        for idx, (vsyncs, rising, falling) in enumerate(zip(vsync_times_in_blocks, diode_rising_edges_in_blocks, diode_falling_edges_in_blocks)):
            
            # diodeBox in Sam's TaskControl script is initially set to 1, but
            # drawing the first frame inverts to -1, so assuming the pre-stim
            # frames are grey then the first frame is a grey-to-black (falling)
            # transition
            falling = falling[falling > vsyncs[0]]
            rising = rising[rising > falling[0]]
            
            diode_flips = np.sort(np.concatenate((rising, falling)))
            
            short_interval_threshold = 0.1 * 1/60
            long_interval_threshold = np.mean(np.diff(diode_flips)) + 3 * np.std(np.diff(diode_flips))
            # long threshold should only be applied at start or end of timestamps
            
            while np.diff(diode_flips)[0] > long_interval_threshold:
                diode_flips = diode_flips[1:]
                
            while np.diff(diode_flips)[-1] > long_interval_threshold:
                diode_flips = diode_flips[:-1]
                
            def short_interval_indices(frametimes):
                intervals = np.diff(frametimes)
                return np.where(intervals < short_interval_threshold)[0]
            
            while any(short_interval_indices(diode_flips)):
                indices = short_interval_indices(diode_flips)
                diode_flips = np.delete(diode_flips, slice(indices[0], indices[0] + 2))
            
            if len(diode_flips) - len(vsyncs) == 1:
                # likely transition from last frame to grey
                diode_flips = diode_flips[:-1]
            
            if round(np.mean(np.diff(diode_flips)), 1) == round(np.mean(np.diff(vsyncs)), 1):
                # diode flip every vsync
                
                if len(diode_flips) != len(vsyncs):
                    logger.warning(f'Mismatch in stim block {idx = }: {len(diode_flips) = }, {len(vsyncs) = }')
                
                    if len(diode_flips) > len(vsyncs):
                        logger.warning('Cutting excess diode flips at number of vsync')
                        diode_flips = diode_flips[:len(vsyncs)]
                    else:
                        raise IndexError('Fewer diode flips than vsyncs: needs investigating')
            else:
                pass
                # TODO adjust frametimes with diode data when flip is every 1 s

            # intervals are in a bimodal distribution due to asymmetry of
            # photodiode thresholding: adjust intervals to get a closer
            # estimate of actual transition time 
            intervals = np.diff(diode_flips)
            mean_flip_interval = np.mean(intervals)
            shift = 0.5 * np.median(abs(intervals - mean_flip_interval))
            
            for idx in range(0, len(diode_flips) - 1, 2):
                # alternate on every short/long interval and shift accordingly
                if idx == 0:
                    if intervals[0] > mean_flip_interval:
                        shift = -shift 
                diode_flips[idx] -= shift
                diode_flips[idx + 1] += shift
                
            AVERAGE_SCREEN_REFRESH_TIME = 0.008
            """Screen refreshes in stages top-to-bottom, total 16 ms measured by
            Corbett, use center as average"""
            frametimes = diode_flips + AVERAGE_SCREEN_REFRESH_TIME
            diode_times_in_blocks.append(frametimes)
        
        self._frametime_blocks = self.Frametimes(main=diode_times_in_blocks[main_stim_block_idx], rf=diode_times_in_blocks[rf_block_idx]if rf_block_idx else None)
        return self.frametime_blocks
    
    @property   
    def task_frametimes(self) -> Sequence[float]:
        """If sync file present, """
        if self.sync:
            return self.frametime_blocks.main
        else:
            assert self.sam is not None
            monitor_latency = 0.027 # s
            # 0.020 s avg to start refreshing (top line of monitor)
            # 0.007 s (after the top line) for center of monitor to refresh
            return self.sam.trialStartTimes + monitor_latency

    #TODO get RF frametimes - cannot load sam's obj currently
    
import functools

def convert_frame_times(method):
    @functools.wraps(method)
    def _impl(self):
        frame_numbers = method(self)
        return self._data.task_frametimes[frame_numbers]
    return _impl

class DRTaskTrials(PropertyDict):
    """All property getters without a leading underscore will be
    considered nwb trials columns. Their docstrings will become the column
    `description`.
    
    To add trials to a pynwb.NWBFile:
    
    >>> obj = DRTaskTrials("DRpilot_626791_20220817") # doctest: +SKIP
    
    >>> for column in obj.to_add_trial_columns(): # doctest: +SKIP
    ...    nwb_file.add_trial_column(**column)
        
    >>> for trial in obj.to_add_trials(): # doctest: +SKIP
    ...    nwb_file.add_trials(**trial)
        
    """
    
    def __init__(self, session: str | pathlib.Path | np_session.Session, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._data = DRDataLoader(session)
        assert self._data.sam is not None
    
    @convert_frame_times
    @property
    def start_time(self) -> Sequence[float]:
        "Earliest time in trial, before any events occur."
        return self._data.sam.trialStartFrame

    @property
    def stop_time(self) -> Sequence[float]:
        return self._data.sam.trialEndFrame


if __name__ == "__main__":
    doctest.testmod()
    
    # x = DRTaskTrials(
    #     "DRpilot_626791_20220817"
    #     )
    # x._data.frametime_blocks
    # x.to_add_trial_columns()
    # x.to_add_trials()