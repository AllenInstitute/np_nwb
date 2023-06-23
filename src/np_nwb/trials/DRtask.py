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
    
    _data: utils.DRDataLoader
    
    def __init__(self, session: str | pathlib.Path | np_session.Session, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._data = utils.DRDataLoader(session) # session is stored in obj
        
    def get_display_times(self, frame_idx: int | Sequence[float]):
        frame_idx = utils.check_array_indices(frame_idx)
        return np.array(self._data.task_frametimes)[frame_idx]


    def get_script_times(self, frame_idx: int | float | Sequence[int | float]):
        """Times of psychopy script 'frames' relative to start of sync"""
        frame_idx = utils.check_array_indices(frame_idx)
        return np.array(
            [
            self._first_vsync_time + self._sam.frameTimes[int(idx)]
            if not np.isnan(idx)
            else np.nan
            for idx in frame_idx
            ]
        )

    
    # ---------------------------------------------------------------------- #
    # helper-properties that won't become columns:
    
    @property
    def _has_opto(self) -> bool:
        return hasattr(self._sam, 'optoVoltage') and any(self._sam.optoVoltage)

    @property
    def _first_vsync_time(self) -> float:
        return self._data.vsync_time_blocks.task[0]
    
    @property
    def _sam(self) -> DynRoutData:
        assert self._data.sam is not None
        return self._data.sam
    
    @property
    def _h5(self) -> h5py.File:
        return self._data.task
    
    @property
    def _len(self) -> int:
        """Number of trials, cached"""
        with contextlib.suppress(AttributeError):
            return self._length
        self._length = len(self.start_time)
        return self._len
    
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
        
        - currently discards inter-trial period
        - extensions due to quiescent violations are discarded: only the final
          `preStimFramesFixed` before a stim are included 
        """
        return self.get_script_times(
            self._sam.stimStartFrame - self._h5['preStimFramesFixed'][()]
        )

    @property
    def quiescent_start_time(self) -> Sequence[float]:
        """Start of period in which the subject should not lick, otherwise the
        trial will be aborted and start over.
        
        - currently just the last quiescent period which was not violated
        - not tracking quiescent violations
        """
        return self.get_script_times(
            self._sam.stimStartFrame - self._h5['quiescentFrames'][()]
        )
    
    @property
    def quiescent_stop_time(self) -> Sequence[float]:
        """End of period in which the subject should not lick, otherwise the
        trial will be aborted and start over."""
        return self.get_script_times(
            self._sam.stimStartFrame
        )
        
    @property
    def response_window_start_time(self) -> Sequence[float]:
        """"""
        return self.get_script_times(
            self._sam.stimStartFrame + self._h5['responseWindow'][()][0]
        )
        
    @property
    def response_time(self) -> Sequence[float]:
        """Time of first lick in the response window, or NaN if no lick occurred."""
        return self.get_script_times(
            self._h5['trialResponseFrame'][()]
        )
        
    
    @property
    def opto_start_time(self) -> Sequence[float]:
        """Onset of optogenetic inactivation."""
        if not self._has_opto:
            return np.nan * np.ones(self._len)
        return np.where(
            ~np.isnan(self._sam.trialOptoOnsetFrame), 
            self.get_script_times(
                self._sam.stimStartFrame + self._sam.trialOptoOnsetFrame
            ),
            np.nan * np.ones(self._len),
        )

    @property
    def opto_stop_time(self) -> Sequence[float]:
        """Offset of optogenetic inactivation."""
        if not self._has_opto:
            return np.nan * np.ones(self._len)
        return self.opto_start_time + self._sam.trialOptoDur
    
    @property
    def stim_start_time(self) -> Sequence[float]:
        """Onset of visual or auditory stimulus."""
        starts = np.nan * np.ones(self._len)
        for idx in range(self._len):
            if self.is_vis_stim[idx]:
                starts[idx] = self.get_display_times(self._sam.stimStartFrame[idx])
            if self.is_catch[idx]:
                starts[idx] = self.get_script_times(self._sam.stimStartFrame[idx])
            if self.is_aud_stim[idx]:
                starts[idx] = self._data.task_sound_on_off[idx][0]
        return starts

    @property
    def stim_stop_time(self) -> Sequence[float]:
        """TODO"""
        ends = np.nan * np.ones(self._len)
        for idx in range(self._len):
            if self.is_vis_stim[idx]:
                ends[idx] = self.get_display_times(self._sam.stimStartFrame[idx] + self._h5['visStimFrames'][()])
            if self.is_catch[idx]:
                ends[idx] = self.get_script_times(self._sam.stimStartFrame[idx] + self._h5['visStimFrames'][()])
            if self.is_aud_stim[idx]:
                ends[idx] = self._data.task_sound_on_off[idx][1]
        return ends
        
    @property
    def response_window_stop_time(self) -> Sequence[float]:
        """"""
        return self.get_script_times(
            self._sam.stimStartFrame + self._h5['responseWindow'][()][1]
        )
        
    @property
    def post_response_window_start_time(self) -> Sequence[float]:
        """"""
        return self.response_window_stop_time
    
    @property
    def post_response_window_stop_time(self) -> Sequence[float]:
        """"""
        return self.get_script_times(
            self._sam.stimStartFrame + self._h5['postResponseWindowFrames'][()]
        )
    
    @property
    def timeout_start_time(self) -> Sequence[float]:
        """"""
        starts = np.nan * np.ones_like((self.start_time))
        for idx in range(0, self._len - 1):
            if self.is_repeat[idx + 1]:
                starts[idx] = self.get_display_times(
                    self._sam.stimStartFrame[idx] + self._h5['postResponseWindowFrames'][()]
                    # TODO + 1?
                )
        return starts
    
    @property
    def timeout_stop_time(self) -> Sequence[float]:
        """"""
        ends = np.nan * np.ones_like((self.start_time))
        for idx in range(0, self._len - 1):
            if self.is_repeat[idx + 1]:
                ends[idx] = self.get_display_times(
                    self._sam.stimStartFrame[idx] + self._h5['postResponseWindowFrames'][()] + self._sam.incorrectTimeoutFrames
                )
        return ends

    '''
    @property
    def _time(self) -> Sequence[float]:
        """TODO"""
        return np.nan * np.zeros(self._len)
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
    def index(self) -> Sequence[int]:
        """0-indexed trial number for regular trials (with stimuli), increments over
        time.
        
        - nan for catch trials
        """
        regular_trial_index = np.nan * np.zeros(self._len)
        counter = 0
        for idx in range(self._len):
            if not self.is_catch[idx]:
                regular_trial_index[idx] = int(counter)
                counter += 1
        return regular_trial_index
    
    @property
    def index_within_block(self) -> Sequence[int]:
        """0-indexed trial number within a block, increments over the block."""
        return np.concatenate([np.arange(count) for count in np.unique(self.block_index, return_counts=True)[1]])
    
    @property
    def scheduled_reward_index_within_block(self) -> Sequence[float]:
        """
        # TODO check w/Sam
         - changed from 'noncontingent_reward_index_within_block'
         - autoreward scheduled, not necessarily given
        """
        return np.where(self.is_reward_scheduled == True, self.index_within_block, np.nan * np.ones(self._len))

    @property
    def opto_area_name(self) -> Sequence[str]:
        """Target location for optogenetic inactivation during the trial."""
        # TODO
        return self._sam.trial
    
    @property
    def opto_area_index(self) -> Sequence[int | float]:
        """Target location for optogenetic inactivation during the trial.
        
        - nan if no opto applied
        """
        # TODO
        return np.nan * np.ones(self._len)
    
    @property
    def opto_voltage(self):
        if not any(self.is_opto):
            return np.full(self._len, np.nan)
        return self._sam.trialOptoVoltage
    
    @property
    def galvo_voltage_x(self):
        if not any(self.is_opto):
            return np.full(self._len, np.nan)
        return np.array([voltage[0] for voltage in self._sam.trialGalvoVoltage])
    
    @property
    def galvo_voltage_y(self):
        if not any(self.is_opto):
            return np.full(self._len, np.nan)
        return np.array([voltage[1] for voltage in self._sam.trialGalvoVoltage])
    
    @property
    def repeat_number(self) -> Sequence[float]:
        """Number of times the trial has already been presented in immediately
        preceding trials (repeated due to false alarm).
        
        - nan for catch trials
        """
        repeats = np.where(~np.isnan(self.index), np.zeros(self._len), np.full(self._len, np.nan))
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
        """
        return self.is_hit | self.is_correct_reject | (self.is_catch & ~self.is_response)
    
    @property
    def is_incorrect(self) -> Sequence[bool]:
        """The subject acted incorrectly in the response window, according to its
        training.
        
        - includes catch trials
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
        - False for catch trials
        """
        return self._sam.trialRepeat
    
    @property
    def is_opto(self) -> Sequence[bool]:
        """Optogenetic inactivation was applied during the trial."""
        return np.isnan(self.opto_start_time)
    
    @property
    def is_context_switch(self) -> Sequence[bool]:
        """The first trial with a stimulus after a change in context."""
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
    
    # TODO add timeseries to link?
    return nwb_file

if __name__ == "__main__":
    doctest.testmod()
    # x = DRTaskTrials("DRpilot_644864_20230201")
    # tuple(x.keys())
    
    nwb_file = main('DRpilot_626791_20220817')
    nwb_file
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
    