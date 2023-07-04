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
        return np.array(
            [
            np.nan if np.isnan(idx)
            else self._data.rf_frametimes[int(idx)]
            for idx in frame_idx
            ]
        )    
        
    def get_script_times(self, frame_idx: Sequence[float]):
        """Times of psychopy script 'frames' relative to start of sync"""
        frame_idx = utils.check_array_indices(frame_idx)
        return np.array(
            [
            np.nan if np.isnan(idx)
            else self._first_vsync_time + self._frame_times[int(idx)]
            for idx in frame_idx
            ]
        )    
        
    @functools.cached_property
    def _first_vsync_time(self) -> float:
        return self._data.vsync_time_blocks.rf[0]
    
    @property
    def _h5(self) -> h5py.File:
        return self._data.rf
    
    @functools.cached_property
    def _len_all_trials(self) -> int:
        return len(self._h5['stimStartFrame'][()])
    
    def find(self, key: str) -> Sequence[bool] | None:
        if key in self._h5:
            return ~np.isnan(self._h5[key][()])
        return None
    
    
    @functools.cached_property
    def _all_aud_freq(self) -> Sequence[float]:
        freq = np.full(self._len_all_trials, np.nan)
        for key in ('trialToneFreq', 'trialSoundFreq', 'trialAMNoiseFreq'):
            if key in self._h5:
                array = self._h5[key][()]
                freq[~np.isnan(array)] = array[~np.isnan(array)]
        return freq
    
    @functools.cached_property
    def _all_aud_idx(self) -> Sequence[float]:
        return np.where(~np.isnan(self._all_aud_freq), np.arange(self._len_all_trials) , np.nan)
    
    @functools.cached_property
    def _all_vis_idx(self) -> Sequence[float]:
        flashes = self.find('trialFullFieldContrast')
        if flashes is None:
            flashes = np.full(self._len_all_trials, False)
        gratings = self.find('trialGratingOri')
        if gratings is None:
            gratings = np.full(self._len_all_trials, False)
        return np.where(gratings ^ flashes, np.arange(self._len_all_trials), np.nan)
    
    @functools.cached_property
    def start_time(self) -> Sequence[float]:
        return self.get_vis_presentation_times(
            self._h5['stimStartFrame'][self._idx]
        )
    
    @functools.cached_property
    def stop_time(self) -> Sequence[float]:
        """Latest time in each trial, after all events have occurred."""
        return self.get_vis_presentation_times(
            self._h5['stimStartFrame'][self._idx] 
            + self._h5['stimFrames'][()]
            + self._h5['interStimFrames'][()]
        )
    
    @functools.cached_property
    def stim_start_time(self) -> Sequence[float]:
        """Onset of mapping stimulus."""
        starts = np.full(self._len, np.nan)
        for idx in range(self._len):
            if self._is_vis_stim[idx]:
                starts[idx] = self.get_vis_presentation_times(self._h5['stimStartFrame'][self._idx[idx]])
            if self._is_aud_stim[idx]:
                starts[idx] = self._data.rf_sound_on_off[self._idx[idx]][0]
        return starts

    @functools.cached_property
    def stim_stop_time(self) -> Sequence[float]:
        """Offset of mapping stimulus."""
        ends = np.full(self._len, np.nan)
        frames_per_stim = self._h5['stimFrames'][()]
        for idx in range(self._len):
            if self._is_vis_stim[idx]:
                ends[idx] = self.get_vis_presentation_times(
                    self._h5['stimStartFrame'][self._idx[idx]] + frames_per_stim
                    )
            if self._is_aud_stim[idx]:
                ends[idx] = self._data.rf_sound_on_off[self._idx[idx]][1]
        return ends
    
    @functools.cached_property
    def index(self) -> Sequence[int]:
        return np.arange(self._len_all_trials)[self._idx]
    
    @functools.cached_property
    def _idx(self) -> Sequence[int]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.arange(self._len_all_trials)
        
    @functools.cached_property
    def _len(self) -> int:
        return len(self._idx)
    
    @functools.cached_property
    def _tone_freq(self) -> Sequence[float]:
        for key in ('trialToneFreq', 'trialSoundFreq'):
            if key in self._h5:
                return self._h5[key][self._idx]
        return np.full(self._len, np.nan)
    
    @functools.cached_property
    def _AM_noise_freq(self) -> Sequence[float]:
        return self._h5['trialAMNoiseFreq'][self._idx] if 'trialAMNoiseFreq' in self._h5 else np.full(self._len, np.nan)
    
    @functools.cached_property
    def _is_aud_stim(self) -> Sequence[bool]:
        """Includes AM noise and pure tones."""
        return np.where(np.isnan(self._all_aud_idx[self._idx]), False, True)
    
    @functools.cached_property
    def _is_vis_stim(self) -> Sequence[bool]:
        return np.where(np.isnan(self._all_vis_idx[self._idx]), False, True)
    
    @functools.cached_property
    def _full_field_contrast(self) -> Sequence[float]:
        return self._h5['trialFullFieldContrast'][self._idx] if 'trialFullFieldContrast' in self._h5 else np.full(self._len, np.nan)
    
class VisMappingTrials(RFTrials):
    
    @functools.cached_property
    def _idx(self) -> Sequence[int]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.array(self._all_vis_idx[~np.isnan(self._all_vis_idx)], dtype=np.int32)
    
    @functools.cached_property
    def is_small_field_grating(self) -> Sequence[bool]:
        return ~np.isnan(self.grating_orientation)
    
    @functools.cached_property
    def grating_orientation(self) -> Sequence[float]:
        return self._h5['trialGratingOri'][self._idx] 

    @functools.cached_property
    def grating_x(self) -> Sequence[float]:
        """Position of grating patch center, in pixels from screen center"""
        return np.array([xy[0] for xy in self._h5['trialVisXY'][self._idx]])
    
    @functools.cached_property
    def grating_y(self) -> Sequence[float]:
        """Position of grating patch center, in pixels from screen center"""
        return np.array([xy[1] for xy in self._h5['trialVisXY'][self._idx]])
    
    @functools.cached_property
    def is_full_field_flash(self) -> Sequence[bool]:
        return ~np.isnan(self._full_field_contrast)
    
    @functools.cached_property
    def flash_contrast(self) -> Sequence[float]:
        return self._full_field_contrast
    

        
class AudMappingTrials(RFTrials):
    
    @functools.cached_property
    def _idx(self) -> Sequence[int]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.array(self._all_aud_idx[~np.isnan(self._all_aud_idx)], dtype=np.int32)
    
    @functools.cached_property
    def is_AM_noise(self) -> Sequence[bool]:
        return ~np.isnan(self._AM_noise_freq)
    
    @functools.cached_property
    def is_pure_tone(self) -> Sequence[bool]:
        return ~np.isnan(self._tone_freq)
    
    @functools.cached_property
    def freq(self) -> Sequence[float]:
        """Frequency of pure tone or frequency of modulation for AM noise, in Hz"""
        return self._all_aud_freq[self._idx]
    
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
        
        for modality, obj in zip(
            ('visual', 'auditory'), 
            (VisMappingTrials(session), AudMappingTrials(session)),
            ):
            module = pynwb.epoch.TimeIntervals(
                name=f"{modality}_mapping_trials",
                description=f"{modality} receptive-field mapping trials",
            )
            
            for column in obj.to_add_trial_column():
                module.add_column(**column)
            
            for trial in obj.to_add_trial():
                module.add_row(**trial)
            
            nwb_file.add_time_intervals(module)
    # TODO add timeseries to link?
    return nwb_file

if __name__ == "__main__":
    doctest.testmod()
    # x = VisMappingTrials("DRpilot_644864_20230201").stim_start_time
    # x = AudMappingTrials("DRpilot_644864_20230201").stim_start_time
    # tuple(x.keys())
    
    nwb_file = main('DRpilot_644864_20230131')
    
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
    