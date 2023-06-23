from __future__ import annotations

import collections.abc
import contextlib
import datetime
import doctest
import functools
import pathlib
import reprlib
import warnings
from typing import (Any, Generator, Iterable, Iterator, Literal, NamedTuple,
                    Optional, Sequence)

import allensdk.brain_observatory.sync_dataset as sync_dataset
import h5py
import np_logging
import np_session
import np_tools
import numpy as np
import pandas as pd
import pynwb
from Analysis.DynamicRoutingAnalysisUtils import DynRoutData

import np_nwb.interfaces as interfaces
import np_nwb.utils as utils
from np_nwb.trials.property_dict import PropertyDict

logger = np_logging.getLogger(__name__)   

class RFTrials(PropertyDict):
    """All property getters without a leading underscore will be
    considered nwb trials columns. Their docstrings will become the column
    `description`.
    
    To add trials to a pynwb.NWBFile:
    
    >>> obj = RFTrials("DRpilot_626791_20220817") # doctest: +SKIP
    
    >>> for column in obj.to_add_trial_column(): # doctest: +SKIP
    ...    nwb_file.add_trial_column(**column)
        
    >>> for trial in obj.to_add_trial(): # doctest: +SKIP
    ...    nwb_file.add_trial(**trial)
        
    """
    
    _data: utils.DRDataLoader
            
    def __init__(self, session: str | pathlib.Path | np_session.Session, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._data = utils.DRDataLoader(session) # session is stored in obj
        if not self._data.rf_hdf5:
            raise FileNotFoundError(f'No RF mapping hdf5 found in {session.npexp_path}')
        self._frame_times = np.concatenate(([0], np.cumsum(self._h5['frameIntervals'][:])))
        
    def get_vis_presentation_times(self, frame_idx: int | Sequence[float]):
        frame_idx = utils.check_array_indices(frame_idx)
        return np.array(self._data.rf_frametimes)[frame_idx]
    
        
    def get_script_times(self, frame_idx: Sequence[float]):
        """Times of psychopy script 'frames' relative to start of sync"""
        frame_idx = utils.check_array_indices(frame_idx)
        return np.array(
            [
            self._first_vsync_time + self._frame_times[int(idx)]
            if not np.isnan(idx) 
            else np.nan
            for idx in frame_idx
            ]
        )    
        
    @property
    def _first_vsync_time(self) -> float:
        return self._data.vsync_time_blocks.rf[0]
    
    @property
    def _h5(self) -> h5py.File:
        return self._data.rf
    
    @property
    def _len(self) -> int:
        """Number of trials, cached"""
        with contextlib.suppress(AttributeError):
            return self._length
        self._length = len(self.start_time)
        return self._len
    
    @property
    def start_time(self) -> Sequence[float]:
        return self.get_script_times(
            self._h5['stimStartFrame'][()]
        )
        
    @property
    def stim_start_time(self) -> Sequence[float]:
        """Onset of visual or auditory stimulus."""
        starts = np.nan * np.ones(self._len)
        for idx in range(self._len):
            if self.is_vis_stim[idx]:
                starts[idx] = self.get_vis_presentation_times(self._h5['stimStartFrame'][idx])
            if self.is_aud_stim[idx]:
                starts[idx] = self._data.rf_sound_on_off[idx][0]
        return starts

    @property
    def stim_stop_time(self) -> Sequence[float]:
        """Onset of visual or auditory stimulus."""
        ends = np.nan * np.ones(self._len)
        frames_per_stim = self._h5['stimFrames'][()]
        for idx in range(self._len):
            if self.is_vis_stim[idx]:
                ends[idx] = self.get_vis_presentation_times(
                    self._h5['stimStartFrame'][idx] + frames_per_stim
                    )
            if self.is_aud_stim[idx]:
                ends[idx] = self._data.rf_sound_on_off[idx][1]

        return ends
    
    @property
    def stop_time(self) -> Sequence[float]:
        """Latest time in each trial, after all events have occurred."""
        return self.get_script_times(
            self._h5['stimStartFrame'][()] 
            + self._h5['stimFrames'][()]
            + self._h5['interStimFrames'][()]
        )

    @property
    def grating_x_pos(self) -> Sequence[float]:
        return np.array([xy[0] for xy in self._h5['trialVisXY'][()]])
    
    @property
    def grating_y_pos(self) -> Sequence[float]:
        return np.array([xy[1] for xy in self._h5['trialVisXY'][()]])
    
    @property
    def full_field_contrast(self) -> Sequence[float]:
        return self._h5['trialFullFieldContrast'][()] if 'trialFullFieldContrast' in self._h5 else np.nan * np.ones(self._len)
    
    @property
    def grating_orientation(self) -> Sequence[float]:
        return self._h5['trialGratingOri'][()] 
    
    @property
    def tone_freq(self) -> Sequence[float]:
        return self._h5['trialToneFreq'][()] if 'trialToneFreq' in self._h5 else np.nan * np.ones(self._len)
    
    @property
    def am_noise_freq(self) -> Sequence[float]:
        return self._h5['trialAMNoiseFreq'][()] if 'trialAMNoiseFreq' in self._h5 else np.nan * np.ones(self._len)
    
    @property
    def is_vis_stim(self) -> Sequence[bool]:
        return ~np.isnan(self.grating_orientation)
    
    @property
    def is_aud_stim(self) -> Sequence[bool]:
        """Includes AM noise and pure tones."""
        return np.isnan(self.grating_orientation)
    
    @property
    def is_aud_noise_stim(self) -> Sequence[bool]:
        return ~np.isnan(self.am_noise_freq)
    
    @property
    def is_aud_tone_stim(self) -> Sequence[bool]:
        return ~np.isnan(self.tone_freq)
    
    
def main(
    session: interfaces.SessionFolder,
    nwb_file: interfaces.OptionalInputFile = None,
    output_file: interfaces.OptionalOutputFile = None,
) -> pynwb.NWBFile:
    """Add trials table to nwb_file."""

    session, nwb_file, output_file = utils.parse_session_nwb_args(
        session, nwb_file, output_file
    )
    
    if hasattr(session, 'hdf5s') and any('rfmapping' in f.name.lower() for f in session.hdf5s):
        
        rf_mapping = pynwb.epoch.TimeIntervals(
            name="rf_mapping",
            description="Intervals for each receptive-field mapping trial",
        )
        
        obj = RFTrials(session)
        
        for column in obj.to_add_trial_column():
            rf_mapping.add_column(**column)
        
        for trial in obj.to_add_trial():
            rf_mapping.add_row(**trial)
        
        nwb_file.add_time_intervals(rf_mapping)
    # TODO add timeseries to link?
    return nwb_file

if __name__ == "__main__":
    doctest.testmod()
    # x = DRTaskTrials("DRpilot_644864_20230201")
    # tuple(x.keys())
    
    # nwb_file = main('DRpilot_644864_20230131')
    
    # x = RFTrials("DRpilot_644864_20230201")
    # df = x.to_dataframe()
    # x = DRTaskTrials("DRpilot_626791_20220817")
    # x.repeat_number
    
    # import np_tools
    # nwb = np_tools.init_nwb(x._data.session)
    # for column in x.to_add_trial_column():
    #     nwb.add_trial_column(**column)
    # for trial in x.to_add_trial():
    #     nwb.add_trial(**trial)
        
    # import yaml
    # pathlib.Path('test.yaml').write_text(
    #     yaml.dump(x._docstrings, line_break='\n')
    #     )
    