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
import np_tools
import h5py
import pynwb
import allensdk.brain_observatory.sync_dataset as sync_dataset
from Analysis.DynamicRoutingAnalysisUtils import DynRoutData

import np_nwb.utils as utils
import np_nwb.interfaces as interfaces
from np_nwb.trials.property_dict import PropertyDict

logger = np_logging.getLogger(__name__)


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
    def task(self) -> h5py.File:
        with contextlib.suppress(AttributeError):
            return self._task
        self._task = h5py.File(self.task_hdf5)
        return self.task
    
    @property
    def rf_hdf5(self) -> pathlib.Path | None:
        return next((file for file in self.session.hdf5s if 'RFMapping' in file.stem), None)
    
    @property
    def rf(self) -> h5py.File:
        with contextlib.suppress(AttributeError):
            return self._rf
        self._rf = h5py.File(self.rf_hdf5)
        return self.rf

    @property
    def sync(self) -> sync_dataset.Dataset | None:
        with contextlib.suppress(AttributeError):
            return self._sync
        if self.session.sync:
            self._sync = sync_dataset.Dataset(self.session.sync)
        else:
            self._sync = None
        return self.sync
    
    class Times(NamedTuple):
        main: Sequence[float]
        rf: Sequence[float] | None
    
    
    
    @property
    def vsync_time_blocks(self) -> Times:
        """Blocks of vsync falling edge times from sync: one block per stimulus.
        
        Not available (or needed) for experiments without sync.
        """
        if self.sync is None:
            raise AttributeError(f'Cannot get sync file for {self}')
        
        with contextlib.suppress(AttributeError):
            return self._vsync_time_blocks
        
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
        self.main_stim_block_idx = block_lengths.index(max(block_lengths))
        self.rf_block_idx = None
        if len(vsync_times_in_blocks) <= 2:
            self.rf_block_idx = int(not self.main_stim_block_idx)
        else:
            logger.warning("More than 2 vsync blocks found: cannot determine which is which. Assumptions made may be incorrect.")
        
        stim_running_rising_edges: Sequence[float] = self.sync.get_rising_edges('stim_running', units = 'seconds')
        stim_running_falling_edges: Sequence[float] = self.sync.get_falling_edges('stim_running', units = 'seconds')
        
        if len(stim_running_rising_edges) and len(stim_running_falling_edges):
            if stim_running_rising_edges[0] > stim_running_falling_edges[0]:
                stim_running_falling_edges[1:]
            assert len(stim_running_rising_edges) == len(vsync_times_in_blocks)
            # TODO filter vsync blocks on stim running
        
        self._vsync_time_blocks = self.Times(main=vsync_times_in_blocks[self.main_stim_block_idx], rf=vsync_times_in_blocks[self.rf_block_idx]if self.rf_block_idx else None)
        return self.vsync_time_blocks
    
    @property
    def frame_display_time_blocks(self) -> Times:
        """Blocks of adjusted diode times from sync: one block per stimulus.
        
        Not available (or needed) for experiments without sync.
        
        Assumes task was the longest block of diode
        stim events (RFmapping / opto are shorter)
        """
        if self.sync is None:
            raise AttributeError(f'Cannot get sync file for {self}')
        
        with contextlib.suppress(AttributeError):
            return self._frame_display_time_blocks
            
        diode_rising_edges: Sequence[float] = self.sync.get_rising_edges('stim_photodiode', units = 'seconds')
        diode_falling_edges: Sequence[float] = self.sync.get_falling_edges('stim_photodiode', units = 'seconds')
        assert abs(len(diode_rising_edges) - len(diode_falling_edges)) < 2
        
        diode_rising_edges_in_blocks = utils.reshape_timestamps_into_blocks(diode_rising_edges)
        diode_falling_edges_in_blocks = utils.reshape_timestamps_into_blocks(diode_falling_edges)
        
        diode_times_in_blocks = []
        for idx, (vsyncs, rising, falling) in enumerate(zip(self.vsync_time_blocks, diode_rising_edges_in_blocks, diode_falling_edges_in_blocks)):
            
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
                        logger.warning('Cutting excess diode flips at length of vsyncs')
                        diode_flips = diode_flips[:len(vsyncs)]
                    else:
                        raise IndexError('Fewer diode flips than vsyncs: needs investigating')
            else:
                pass
                # TODO adjust frametimes with diode data when flip is every 1 s

            # intervals have bimodal distribution due to asymmetry of
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
        
        self._frame_display_time_blocks = self.Times(main=diode_times_in_blocks[self.main_stim_block_idx], rf=diode_times_in_blocks[self.rf_block_idx] if self.rf_block_idx else None)
        return self.frame_display_time_blocks
    
    @property   
    def task_frametimes(self) -> Sequence[float]:
        """If sync file present, """
        if self.sync:
            return self.frame_display_time_blocks.main
        else:
            assert self.sam is not None
            monitor_latency = 0.028 # s
            # 0.020 s avg to start refreshing (top line of monitor)
            # 0.008 s (after the top line) for center of monitor to refresh
            return self.sam.trialStartTimes + monitor_latency

    @property
    def rf_frametimes(self) -> Sequence[float]:
        if not self.sync:
            raise AttributeError(f'No sync file: rf times only available for ecephys experiments with sync: {self.session}')
        
        if not self.frame_display_time_blocks.rf:
            raise AttributeError(f'A block of frametimes corresponding to rf mapping was not found for this experiment: {self.session}')
       
        return self.frame_display_time_blocks.rf
    

class DRTaskTrials(PropertyDict):
    """All property getters without a leading underscore will be
    considered nwb trials columns. Their docstrings will become the column
    `description`.
    
    To add trials to a pynwb.NWBFile:
    
    >>> obj = DRTaskTrials("DRpilot_626791_20220817") # doctest: +SKIP
    
    >>> for column in obj.to_add_trial_column(): # doctest: +SKIP
    ...    nwb_file.add_trial_column(**column)
        
    >>> for trial in obj.to_add_trial(): # doctest: +SKIP
    ...    nwb_file.add_trial(**trial)
        
    """
    
    _data: DRDataLoader
    
    def __init__(self, session: str | pathlib.Path | np_session.Session, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._data = DRDataLoader(session) # session is stored in obj
        
    def get_display_times(self, frame_idx: Sequence[float]):
        return np.array(self._data.task_frametimes)[frame_idx]

    def get_task_times(self, frame_idx: Sequence[float]):
        """Times of task event 'frames' relative to start of sync"""
        return self._first_vsync_time + self._sam.frameTimes[frame_idx]
    # ---------------------------------------------------------------------- #
    # helper-properties that won't become columns:
    
    @property
    def _has_opto(self) -> bool:
        return hasattr(self._sam, 'trialOptoOnsetFrame')

    @property
    def _first_vsync_time(self) -> float:
        # TODO update to use rising edge, or subtract avg (falling - rising)
        return min(block[0] for block in self._data.vsync_time_blocks)
    
    @property
    def _sam(self) -> DynRoutData:
        assert self._data.sam is not None
        return self._data.sam
    
    @property
    def _h5(self) -> h5py.File:
        return self._data.task
    
    
    @property
    def _aud_stims(self) -> Sequence[str]:
        return np.unique([stim for stim in self.stim_name if 'sound' in stim.lower()])
    
    @property
    def _vis_stims(self) -> Sequence[str]:
        return np.unique([stim for stim in self._sam.trialStim if 'vis' in stim.lower()])
    
    @property
    def _targets(self) -> Sequence[str]:
        return np.unique([stim for stim in self._sam.blockStimRewarded])
    
    @property
    def _aud_targets(self) -> Sequence[str]:
        return np.unique([stim for stim in self._aud_stims if stim in self._targets])
    
    @property
    def _vis_targets(self) -> Sequence[str]:
        return np.unique([stim for stim in self._vis_stims if stim in self._targets])
    
    @property
    def _aud_nontargets(self) -> Sequence[str]:
        return np.unique([stim for stim in self._aud_stims if stim not in self._targets])
    
    @property
    def _vis_nontargets(self) -> Sequence[str]:
        return np.unique([stim for stim in self._vis_stims if stim not in self._targets])
    
    @property
    def _trial_rewarded_stim_name(self) -> Sequence[str]:
        return self._sam.blockStimRewarded[self.block_index]
    
    # ---------------------------------------------------------------------- #
    # times:
        
    @property
    def start_time(self) -> Sequence[float]:
        """Earliest time in each trial, before any events occur.
        
        - currently discards inter-trial period and any quiescent violations
        """
        return self.get_task_times(
            self._sam.stimStartFrame - self._h5['preStimFramesFixed'][()]  - self._h5['quiescentFrames'][()]
        )

    @property
    def quiescent_start_time(self) -> Sequence[float]:
        """Start of period in which the subject should not lick, otherwise the
        trial will be aborted and start over.
        
        - currently just the last quiescent period which was not violated
        - not tracking quiescent violations
        """
        return self.get_task_times(
            self._sam.stimStartFrame - self._h5['quiescentFrames'][()]
        )
    
    @property
    def quiescent_stop_time(self) -> Sequence[float]:
        """End of period in which the subject should not lick, otherwise the
        trial will be aborted and start over."""
        return self.get_task_times(
            self._sam.stimStartFrame
        )
        
    @property
    def response_window_start_time(self) -> Sequence[float]:
        """"""
        return self.get_task_times(
            self._sam.stimStartFrame + self._h5['responseWindow'][()][0]
        )
        
    @property
    def opto_start_time(self) -> Sequence[float]:
        """Onset of optogenetic inactivation."""
        if not self._has_opto:
            return np.nan * np.ones_like(self.start_time)
        return self.get_task_times(
            self._sam.stimStartFrame + self._sam.trialOptoOnsetFrame
        ) 

    @property
    def opto_stop_time(self) -> Sequence[float]:
        """Offset of optogenetic inactivation."""
        if not self._has_opto:
            return np.nan * np.ones_like(self.start_time)
        return self.opto_start_time + self._sam.trialOptoDur
    
    @property
    def stim_start_time(self) -> Sequence[float]:
        """Onset of visual or auditory stimulus."""
        starts = np.nan * np.ones_like(self.start_time)
        for idx in range(len(self.start_time)):
            if self.is_vis_stim[idx]:
                starts[idx] = self.get_display_times(self._sam.stimStartFrame[idx])
            if self.is_catch[idx]:
                starts[idx] = self.get_task_times(self._sam.stimStartFrame[idx])
            if self.is_aud_stim[idx]:
                # TODO get corrected aud offset
                starts[idx] = self.get_task_times(self._sam.stimStartFrame[idx])
        return starts

    @property
    def stim_stop_time(self) -> Sequence[float]:
        """TODO"""
        ends = np.nan * np.ones_like(self.start_time)
        for idx in range(len(self.start_time)):
            if self.is_vis_stim[idx]:
                ends[idx] = self.get_display_times(self._sam.stimStartFrame[idx] + self._h5['visStimFrames'][()] + 1)
            if self.is_catch[idx]:
                ends[idx] = self.get_task_times(self._sam.stimStartFrame[idx] + self._h5['visStimFrames'][()] + 1)
            if self.is_aud_stim[idx]:
                # TODO get corrected aud offset
                ends[idx] = self.get_task_times(self._sam.stimStartFrame[idx]) + self._h5['soundDur'][()]
        return ends
        
    @property
    def response_window_stop_time(self) -> Sequence[float]:
        """"""
        return self.get_task_times(
            self._sam.stimStartFrame + self._h5['responseWindow'][()][1]
        )
        
    @property
    def post_response_window_start_time(self) -> Sequence[float]:
        """"""
        return self.response_window_stop_time
    
    @property
    def post_response_window_stop_time(self) -> Sequence[float]:
        """"""
        return self.get_task_times(
            self._sam.stimStartFrame + self._h5['postResponseWindowFrames'][()]
        )
    
    @property
    def timeout_start_time(self) -> Sequence[float]:
        """"""
        starts = np.nan * np.ones_like((self.start_time))
        for idx in range(0, len(self.start_time) - 1):
            if self.is_repeat[idx + 1]:
                starts[idx] = self.get_display_times(
                    self._sam.stimStartFrame[idx] + self._h5['postResponseWindowFrames'][()] + 1
                )
        return starts
    
    @property
    def timeout_stop_time(self) -> Sequence[float]:
        """"""
        ends = np.nan * np.ones_like((self.start_time))
        for idx in range(0, len(self.start_time) - 1):
            if self.is_repeat[idx + 1]:
                ends[idx] = self.get_display_times(
                    self._sam.stimStartFrame[idx] + self._h5['postResponseWindowFrames'][()] + self._sam.incorrectTimeoutFrames + 1
                )
        return ends

    '''
    @property
    def _time(self) -> Sequence[float]:
        """TODO"""
        return np.nan * np.zeros_like(self.start_time)
    '''

    @property
    def stop_time(self) -> Sequence[float]:
        """Latest time in each trial, after all events have occurred."""
        return self.get_display_times(self._sam.trialEndFrame)

    
    # ---------------------------------------------------------------------- #
    # parameters:
        
    @property
    def stim_name(self) -> Sequence[str]:
        """TODO"""
        return self._sam.trialStim
    
    @property
    def stim_index(self) -> Sequence[float]:
        """0-indexed stim number, randomized over trials.
        
        - refers to `nwb.stimulus.templates` table
        - nan for aborted trials
        - nan for catch trials
        """
        # TODO link to stimulus table 
        return np.unique(self.stim_name).searchsorted(self.stim_name)
    
    
    @property
    def block_index(self) -> Sequence[int]:
        """0-indexed block number, increments with each block."""
        assert min(self._sam.trialBlock) == 1
        return self._sam.trialBlock - 1
    
    @property
    def context_name(self) -> Sequence[str]:
        """TODO"""
        return np.array([name[:-1] for name in self._trial_rewarded_stim_name])
    
    @property
    def index_all(self) -> Sequence[int]:
        """0-indexed trial number, increments over time.
        
        - includes aborted trials
        - includes catch trials
        """
        return np.arange(len(self.start_time))
    
    @property
    def index(self) -> Sequence[int]:
        """0-indexed trial number for regular trials (with stimuli), increments over
        time.
        
        - nan for catch trials
        - nan for aborted trials
        """
        regular_trial_index = np.nan * np.zeros_like(self.start_time)
        counter = 0
        for idx in range(len(self.start_time)):
            if not (self.is_catch[idx] or self.is_aborted[idx]):
                regular_trial_index[idx] = int(counter)
                counter += 1
        return regular_trial_index
    
    @property
    def index_within_block(self) -> Sequence[int]:
        """0-indexed trial number within a block, increments over the block.
        
        # TODO aborted trials not tracked
        """
        return np.concatenate([np.arange(count) for count in np.unique(self.block_index, return_counts=True)[1]])
    
    @property
    def scheduled_reward_index_within_block(self) -> Sequence[float]:
        """
        # TODO check w/Sam
         - changed from 'noncontingent_reward_index_within_block'
         - autoreward scheduled, not necessarily given
        """
        return np.where(self.is_reward_scheduled == True, self.index_within_block, np.nan * np.ones_like(self.index_within_block))

    @property
    def opto_area_name(self) -> Sequence[str]:
        """Target location for optogenetic inactivation during the trial."""
        # TODO
        return np.nan * np.ones_like(self.is_opto)
    
    @property
    def opto_area_index(self) -> Sequence[int | float]:
        """Target location for optogenetic inactivation during the trial.
        
        - nan if no opto applied
        """
        # TODO
        return np.nan * np.ones_like(self.is_opto)
    
    @property
    def repeat_number(self) -> Sequence[float]:
        """Number of times the trial has already been presented in immediately
        preceding trials (repeated due to false alarm).
        
        - nan for aborted trials
        - nan for catch trials
        """
        zeros = np.zeros_like(self.index)
        repeats = np.where(np.isnan(self.index), zeros, np.nan * zeros)
        counter = 0
        for idx in np.where(repeats == 0)[0]:
            if self.is_repeat[idx]:
                counter += 1
            else:
                counter = 0
            repeats[idx] = int(counter)
        return repeats

    # ---------------------------------------------------------------------- #
    # bools:
    
    @property
    def is_response(self) -> Sequence[bool]:
        """The subject licked one or more times during the response window."""
        return self._sam.trialResponse

    @property
    def is_correct(self) -> Sequence[bool]:
        """The subject acted correctly in the response window, according to its
        training.
        
        - includes catch trials
        - excludes aborted trials
        """
        return self.is_hit | self.is_correct_reject | (self.is_catch & ~self.is_response)
    
    @property
    def is_incorrect(self) -> Sequence[bool]:
        """The subject acted incorrectly in the response window, according to its
        training.
        
        - includes catch trials
        - excludes aborted trials
        """
        return self.is_miss | self.is_false_alarm | (self.is_catch & self.is_response)
    
    @property
    def is_hit(self) -> Sequence[bool]:
        """The subject responded in a GO trial."""
        return self.is_response & self.is_go
    
    @property
    def is_false_alarm(self) -> Sequence[bool]:
        """The subject responded in a NOGO trial."""
        return self.is_response & self.is_nogo
    
    @property
    def is_correct_reject(self) -> Sequence[bool]:
        """The subject did not respond in a NOGO trial.
        
        - excludes catch trials
        """
        return ~self.is_response & self.is_nogo
    
    @property
    def is_miss(self) -> Sequence[bool]:
        """The subject did not respond in a GO trial."""
        return ~self.is_response & self.is_go
    
    @property
    def is_go(self) -> Sequence[bool]:
        """Condition in which the subject should respond.
    
        - target stim presented in rewarded context block
        """
        return self._sam.goTrials
    
    @property
    def is_nogo(self) -> Sequence[bool]:
        """Condition in which the subject should not respond.
        
        - non-target stim presented in any context block
        - target stim presented in non-rewarded context block
        - excludes catch trials
        """
        return self._sam.nogoTrials
    
    @property
    def is_rewarded(self) -> Sequence[bool]:
        """The subject received a reward.
        
        - includes non-contingent rewards
        """
        return self._sam.trialRewarded
    
    @property
    def is_noncontingent_reward(self) -> Sequence[bool]:
        """The subject received a reward that did not depend on its response."""        
        return self._sam.autoRewarded

    @property
    def is_contingent_reward(self) -> Sequence[bool]:
        """The subject received a reward for a correct response in a GO trial."""     
        return self.is_rewarded & self.is_hit
    
    @property
    def is_reward_scheduled(self) -> Sequence[bool]:
        """A non-contingent reward was scheduled to occur.
        
        - subject may have responded correctly and received contingent reward
          instead
        """
        return self._sam.autoRewardScheduled
    
    @property
    def is_aud_stim(self) -> Sequence[bool]:
        """An auditory stimulus was presented.
        
        - target and non-target stimuli
        - rewarded and unrewarded contexts
        """
        return np.isin(self._sam.trialStim, self._aud_stims)
    
    @property
    def is_vis_stim(self) -> Sequence[bool]:
        """A visual stimulus was presented.
        
        - target and non-target stimuli
        - rewarded and unrewarded contexts
        """
        return np.isin(self._sam.trialStim, self._vis_stims)
    
    @property
    def is_catch(self) -> Sequence[bool]:
        """No stimulus was presented."""
        return np.isin(self._sam.trialStim, 'catch')
    
    @property
    def is_aud_target(self) -> Sequence[bool]:
        """An auditory stimulus was presented that the subject should respond to in a certain context."""
        return np.isin(self._sam.trialStim, self._aud_targets)
    
    @property
    def is_vis_target(self) -> Sequence[bool]:
        """A visual stimulus was presented that the subject should respond to in a certain context."""
        return np.isin(self._sam.trialStim, self._vis_targets)
    
    @property
    def is_aud_nontarget(self) -> Sequence[bool]:
        """An auditory stimulus was presented that the subject should never respond to."""
        return np.isin(self._sam.trialStim, self._aud_nontargets)
    
    @property
    def is_vis_nontarget(self) -> Sequence[bool]:
        """A visual stimulus was presented that the subject should never respond to."""
        return np.isin(self._sam.trialStim, self._vis_nontargets)
    
    @property
    def is_vis_context(self) -> Sequence[bool]:
        """Trial occurs within a block in which the subject should respond to
        a visual target."""
        return np.isin(self._trial_rewarded_stim_name, self._vis_stims)
    
    @property
    def is_aud_context(self) -> Sequence[bool]:
        """Trial occurs within a block in which the subject should respond to
        a auditory target."""
        return np.isin(self._trial_rewarded_stim_name, self._aud_stims)
    
    @property
    def is_repeat(self) -> Sequence[bool]:
        """The trial is a repetition of the previous trial, which resulted in a
        miss.
        
        - always False on first full trial after context block switch
        - False for all aborted trials
        - False for catch trials
        """
        return self._sam.trialRepeat
    
    @property
    def is_aborted(self) -> Sequence[bool]:
        """The subject licked one or more times during the quiescent period.
        
        - trial ends at quiescent period stop time
        # TODO
        """
        return np.zeros_like(self._sam.trialStartTimes, dtype=bool)
    
    @property
    def is_opto(self) -> Sequence[bool]:
        """Optogenetic inactivation was applied during the trial."""
        return np.isnan(self.opto_start_time)
    
    @property
    def is_context_switch(self) -> Sequence[bool]:
        """The first trial with a stimulus after a change in context.
        
        - excludes aborted trials
        """
        return np.isin(self.index_within_block, 1)
    
    """
    @property
    def is_(self) -> Sequence[bool]:
        """"""
        return 
    """
def main(
    session: interfaces.SessionFolder,
    nwb_file: interfaces.OptionalInputFile = None,
    output_file: interfaces.OptionalOutputFile = None,
) -> pynwb.NWBFile:
    """Add trials table to nwb_file."""

    session, nwb_file, output_file = utils.parse_session_nwb_args(
        session, nwb_file, output_file
    )
    
    obj = DRTaskTrials(session)
    
    for column in obj.to_add_trial_column():
        nwb_file.add_trial_column(**column)
    
    for trial in obj.to_add_trial():
        nwb_file.add_trial(**trial)
    
    return nwb_file

if __name__ == "__main__":
    doctest.testmod()
    # nwb_file = main('DRpilot_626791_20220817')

    
    x = DRTaskTrials("DRpilot_626791_20220817")
    x.repeat_number
    
    import np_tools
    nwb = np_tools.init_nwb(x._data.session)
    for column in x.to_add_trial_column():
        nwb.add_trial_column(**column)
    for trial in x.to_add_trial():
        nwb.add_trial(**trial)
        
    import yaml
    pathlib.Path('test.yaml').write_text(
        yaml.dump(x._docstrings, line_break='\n')
        )
    