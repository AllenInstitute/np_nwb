from __future__ import annotations

import argparse
import contextlib
import doctest
import functools
import itertools
import json
import logging
import pathlib
import pickle
import sys
import tempfile
from typing import Any, Generator, Iterable, Iterator, Literal, NamedTuple, Optional, Sequence

import np_logging
import np_session
import np_tools
import numpy as np
import numpy.typing as npt
import pynwb
import pandas as pd
import h5py
import allensdk.brain_observatory.sync_dataset as sync_dataset
from Analysis.DynamicRoutingAnalysisUtils import DynRoutData

import np_nwb.interfaces as interfaces
import np_nwb.ephys_utils as ephys_utils


logger = np_logging.getLogger(__name__)

def check_array_indices(frame_idx: int | float | Sequence[int | float]) -> Sequence[int | float]:
    """Check indices can be safely converted from float to int. Make
    a single int/float index iterable."""
    try:
        _ = len(frame_idx)
    except TypeError:
        frame_idx = (frame_idx,)
        
    for idx in frame_idx:
        if (
            isinstance(idx, (float, np.floating))
            and not np.isnan(idx)
            and int(idx) != idx
        ):
            raise TypeError('Non-integer `float` used as an index')
    return frame_idx


def get_behavior(nwb_file: pynwb.NWBFile) -> pynwb.ProcessingModule:
    """Get or create `nwb_file['behavior']`"""
    return nwb_file.processing.get('behavior') or nwb_file.create_processing_module(
    name="behavior", description="Processed behavioral data",
    )
def get_ecephys(nwb_file: pynwb.NWBFile) -> pynwb.ProcessingModule:
    """Get or create `nwb_file['ecephys']`"""
    return nwb_file.processing.get('ecephys') or nwb_file.create_processing_module(
    name="ecephys", description="Processed ecephys data",
    )
    
class Info(NamedTuple):
    """
    Equivalent to:
    ```
    tuple[np_session.Session, pynwb.NWBFile, pathlib.Path | None]
    ```
    """
    session: np_session.Session
    nwb: pynwb.NWBFile
    output: pathlib.Path | None

def parse_cli_args() -> Info:
    """
    Get args from the command line, process and return.

    For use in modules that add to an .nwb file.

    Passes args to `parse_session_nwb_args` and returns its results.
    """
    args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'session',
        type=np_session.Session,
        help='A path to a session folder, or an appropriate input argument to `np_session.Session()`, e.g. lims session id',
    )
    parser.add_argument(
        'nwb_filepath',
        nargs='?',
        default=None,
        type=pathlib.Path,
        help='A path to an existing .nwb file to append.',
    )
    parser.add_argument(
        'output_filepath',
        nargs='?',
        default=None,
        type=pathlib.Path,
        help='A path for saving the appended .nwb file, if different to the input path.',
    )
    opts = parser.parse_args(args)
    np_logging.getLogger()
    return parse_session_nwb_args(*vars(opts).values())


def parse_session_nwb_args(
    session_folder: str | pathlib.Path | np_session.Session,
    nwb_file: Optional[str | pathlib.Path | pynwb.NWBFile] = None,
    output_file: Optional[str | pathlib.Path] = None,
) -> Info:
    """Parse the args we need for appending data to an nwb file.

    Ensures that arguments can be provided from the command, in which case the
    results should be saved to disk.

    - only `session_folder` is required

    - if `nwb_file` is provided as a `pynwb.NWBFile`, `output_file` will not be
      modified

    - if `nwb_file` is provided as a path, it will be loaded if it exists
        - if `output_file` is not provided, it will be set to overwrite
          `nwb_file`

    - if neither `nwb_file` and `output_file` are provided, a
      `pynwb.NWBFile` will be initialized from `session_folder` and an output
      in a tempdir will be assigned

    - if the third returned value is a path, it signals that the appended
      nwb file should be written to disk
    """
    if isinstance(session_folder, np_session.Session):
        session = session_folder
    else:
        session = np_session.Session(session_folder)

    if output_file is None:
        output = None
    else:
        output = pathlib.Path(output_file)

    if isinstance(nwb_file, pynwb.NWBFile):
        # this could not have been passed from command line
        nwb = nwb_file
    else:
        if nwb_file is not None and pathlib.Path(nwb_file).exists():
            nwb = np_tools.load_nwb(nwb_file)
        else:
            logger.info('Generating new `pynwb.NWBFile`')
            nwb = np_tools.init_nwb(session)
            nwb_file = (
                output or pathlib.Path(tempfile.mkdtemp()) / f'{session}.nwb'
            )
        if output is None:
            # set output path to overwrite input path
            output = pathlib.Path(nwb_file)

    logger.info(f'Using {session!r}')
    logger.info(f'Using {nwb!r}')
    if output:
        logger.info(f'Writing appended nwb to {output}')

    return Info(session, nwb, output)


def get_sync_file(
    session: np_session.Session,
) -> pathlib.Path:
    sync_file = tuple(
        itertools.chain(
            session.npexp_path.glob('*.sync'),
            session.npexp_path.glob('*T*.h5'),
        ),
    )
    if len(sync_file) != 1:
        raise FileNotFoundError(
            f'Could not find a single sync file: {sync_file}'
        )
    return sync_file[0]


@functools.cache
def get_sync_dataset(
    session: np_session.Session,
) -> sync_dataset.Dataset:
    return sync_dataset.Dataset(get_sync_file(session))


def get_frame_timestamps(
    session: np_session.Session,
) -> npt.NDArray[np.float64]:
    return get_sync_dataset(session).get_rising_edges(
        'vsync_stim', units='seconds'
    )   # type: ignore


def reshape_into_blocks(
    timestamps: Sequence[float],
    min_gap: Optional[int | float] = None,
) -> tuple[Sequence[float], ...]:
    """
    Find the large gaps in timestamps and split at each gap.

    For example, if two blocks of stimuli were recorded in a single sync
    file, there will be one larger-than normal gap in timestamps.

    default min gap threshold: median + 3 * std (won't work well for short seqs)

    >>> reshape_into_blocks([0, 1, 2, 103, 104, 105], min_gap=100)
    ([0, 1, 2], [103, 104, 105])

    >>> reshape_into_blocks([0, 1, 2, 3])
    ([0, 1, 2, 3],)
    """
    intervals = np.diff(timestamps)
    long_interval_threshold = (
        min_gap
        if min_gap is not None
        else (np.median(intervals) + 3 * np.std(intervals))
    )

    ends_of_blocks = []
    for interval_index, interval in zip(intervals.argsort()[::-1], sorted(intervals)[::-1]):
        if interval > long_interval_threshold:
            # large interval found
            ends_of_blocks.append(interval_index + 1)
        else:
            break

    if not ends_of_blocks: 
        # a single block of timestamps
        return (timestamps,)
    
    # create blocks as intervals [start:end]
    ends_of_blocks.sort()
    blocks = []
    start = 0
    for end in ends_of_blocks:
        blocks.append(timestamps[start:end])
        start = end
    blocks.append(timestamps[start:])
    
    # filter out blocks with a single sample (not a block)
    blocks = tuple(block for block in blocks if len(block) > 1)
    
    # filter out blocks with long avg timstamp interval (a few, widely-spaced timestamps)
    blocks = tuple(block for block in blocks if np.median(np.diff(block)) < long_interval_threshold)
    
    return tuple(blocks)


@functools.cache
def get_blocks_of_frame_timestamps(
    session: np_session.Session,
) -> tuple[npt.NDArray[np.float64], ...]:
    frame_times = get_frame_timestamps(session)
    return reshape_into_blocks(frame_times)


def get_stim_epochs(
    session: np_session.Session,
) -> tuple[tuple[float, float], ...]:
    """`(start_sec, end_sec)` for each stimulus block - constructed from
    vsyncs"""
    return tuple(
        (block[0], block[-1])
        for block in get_blocks_of_frame_timestamps(session)
    )

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
    
    def get_nidaq_device_and_data(self) -> None:
        self._nidaq_mic, self._nidaq_timing = get_pxi_nidaq_mic_data(self.session)
        
    @property
    def nidaq_timing(self) -> EphysTimingOnSync:
        if not self.sync:
            raise AttributeError(f'No sync file/PXI NI-DAQmx available for {self.session = }')
        with contextlib.suppress(AttributeError):
            return self._nidaq_timing
        self.get_nidaq_device_and_data()
        return self.nidaq_timing
    
    @property
    def nidaq_mic(self) -> npt.NDArray:
        if not self.sync:
            raise AttributeError(f'No sync file/PXI NI-DAQmx available for {self.session = }')
        with contextlib.suppress(AttributeError):
            return self._nidaq_mic
        self.get_nidaq_device_and_data()
        return self.nidaq_mic
    
    @property
    def nidaq_timestamps(self):
        return (self.nidaq_timing.start_time + np.arange(self.nidaq_mic.size)) / self.nidaq_timing.sampling_rate
    
    class Times(NamedTuple):
        task: Sequence[float]
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
        for vsyncs in reshape_into_blocks(all_vsync_times):
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
        
        self._vsync_time_blocks = self.Times(task=vsync_times_in_blocks[self.main_stim_block_idx], rf=vsync_times_in_blocks[self.rf_block_idx] if self.rf_block_idx is not None else None)
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
        
        diode_rising_edges_in_blocks = reshape_into_blocks(diode_rising_edges)
        diode_falling_edges_in_blocks = reshape_into_blocks(diode_falling_edges)
        
        diode_rising_edges_in_blocks = self.Times(
            task=diode_rising_edges_in_blocks[self.main_stim_block_idx],
            rf=diode_rising_edges_in_blocks[self.rf_block_idx] if self.rf_block_idx is not None else None,
        )
        
        diode_falling_edges_in_blocks = self.Times(
            task=diode_falling_edges_in_blocks[self.main_stim_block_idx],
            rf=diode_falling_edges_in_blocks[self.rf_block_idx] if self.rf_block_idx is not None else None,
        )
        
        diode_times_in_blocks = {}
        for block, vsyncs, rising, falling in zip(self.Times._fields, self.vsync_time_blocks, diode_rising_edges_in_blocks, diode_falling_edges_in_blocks):
            
            if vsyncs is None:
                continue
            
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
                    logger.warning(f'Mismatch in stim {block = }: {len(diode_flips) = }, {len(vsyncs) = }')
                
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
            diode_times_in_blocks[block] = frametimes
        
        self._frame_display_time_blocks = self.Times(task=diode_times_in_blocks['task'], rf=diode_times_in_blocks.get('rf', None))
        return self.frame_display_time_blocks
    
    @property   
    def task_frametimes(self) -> Sequence[float]:
        """Best estimate of monitor update times, using sync+photodiode info if
        available. Adjusted to approximate the moment the center of the screen is updated.
        
        - one timepoint per psychopy 'frame'
        
        - if sync file present:
            - returns monitor update times estimated from photodiode
            - time is relative to first sync sample
            
        - otherwise:
            - returns frametime based on interval between 'frames' of the
              psychopy script
            - adjusted with an empirical estimate of monitor latency
            - time is relative to start of psychopy script
        """
        if self.sync:
            return self.frame_display_time_blocks.task
        else:
            assert self.sam is not None
            stim_latency = 0.028 # s
            # 0.020 s avg to start refreshing (top line of monitor)
            # 0.008 s (after the top line) for center of monitor to refresh
            return self.sam.trialStartTimes + stim_latency

    @property
    def rf_frametimes(self) -> Sequence[float]:
        """Monitor update times, estimated from photodiode, adjusted to
        approximate the moment the center of the screen is updated.
        
        - one timepoint per psychopy 'frame' / corresponding vsync
        - time is relative to first sync sample
        - assumes rfs only run in ecephys experiments, which have a sync file 
        """
        if not self.sync:
            raise AttributeError(f'No sync file: rf times only available for ecephys experiments with sync: {self.session}')
        
        if self.frame_display_time_blocks.rf is None:
            raise AttributeError(f'A block of frametimes corresponding to rf mapping was not found for this experiment: {self.session}')
       
        return self.frame_display_time_blocks.rf
    
    @property
    def cache(self) -> pathlib.Path:
        root = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting/ben/nwb_cache')
        cache = root / str(self.session)
        cache.mkdir(parents=True, exist_ok=True)
        return cache
    
    def get_trial_sound_onsets_offsets_lags(self) -> tuple[Times, Times, Times]:
        cache = self.cache / 'trial_sound_onsets_offsets_lags.pkl'
        if cache.exists():
            return pickle.loads(cache.read_bytes())
        results: list[tuple] = []
        for block in self.Times._fields:
            
            vsyncs = getattr(self.vsync_time_blocks, block)
            h5 = getattr(self, block)
            num_trials = len((h5.get('trialEndFrame') or h5.get('trialSoundArray'))[:])
            
            onsets = np.full(num_trials, np.nan)
            offsets = np.full(num_trials, np.nan)
            lags = np.full(num_trials, np.nan)
            
            trigger_frames = (h5.get('trialStimStartFrame') or h5.get('stimStartFrame'))[:num_trials]
            waveform_rate = h5['soundSampleRate'][()]
            waveforms = h5['trialSoundArray'][:num_trials]
            
            padding = .15 # sec, either side of expected sound window to make sure entire signal is captured
            for idx, waveform in enumerate(waveforms):
                print(f'{idx}/{num_trials}\r', flush=True) 
                if not any(waveform):
                    continue
                
                trigger_time = vsyncs[trigger_frames[idx]]
                trigger_time_on_pxi_nidaq = trigger_time - self.nidaq_timing.start_time
                duration = len(waveform) / waveform_rate
                onset_sample_on_pxi_nidaq = int((trigger_time_on_pxi_nidaq - padding) * self.nidaq_timing.sampling_rate)
                offset_sample_on_pxi_nidaq = int((trigger_time_on_pxi_nidaq + duration + padding) * self.nidaq_timing.sampling_rate)
                
                times = np.arange(offset_sample_on_pxi_nidaq - onset_sample_on_pxi_nidaq) / self.nidaq_timing.sampling_rate - padding
                values = self.nidaq_mic[onset_sample_on_pxi_nidaq:offset_sample_on_pxi_nidaq]
                interp_times = np.arange(-padding, duration + padding, 1 / waveform_rate)
                interp_values = np.interp(interp_times, times, values)
                
                c = np.correlate(interp_values, waveform)
                
                # idx, lag = xcorr_waveform_recording()
                lags[idx] = interp_times[np.argmax(c)]
                onsets[idx] = trigger_time + lags[idx]
                offsets[idx] = onsets[idx] + duration
                
                padding = 2 * lags[idx]
                
                # to verify:
                """
                import matplotlib.pyplot as plt
                norm_values = (interp_values - np.mean(interp_values))/max(interp_values)
                waveform_times = np.arange(0, duration, 1 / waveform_rate)
                plt.plot(waveform_times + lags[idx], waveform)
                plt.plot(interp_times, norm_values)
                """
                
            results.append((onsets, offsets, lags))
            
        onsets = self.Times(*[result[0] for result in results])
        offsets = self.Times(*[result[1] for result in results])
        lags = self.Times(*[result[2] for result in results])
        cache.write_bytes(pickle.dumps((onsets, offsets, lags)))
        return onsets, offsets, lags
    
    def get_trial_sound_stim_times(self) -> None:
        self.sound_onset, self.sound_offset, self.sound_lags = self.get_trial_sound_onsets_offsets_lags()
        
    @property
    def task_sound_on_off(self) -> Sequence[tuple[float, float]]:
        """Best estimate of onset/offset of sound stim using openephys+microphone signals, if available.
        
        - tuple of two timepoints per trialStimStartFrame in h5 file
        - [onset, offset] of sound stim for each trial
        - if session had pxi nidaq/mic recording of sound:
            - times are relative to start of first sync sample
        - otherwise:
            - times are relative to start of psychopy script
            - uses empirical estimate of sound stim latency
        """
        if self.sync:
            with contextlib.suppress(AttributeError):
                return self._task_sound_on_off
            self.get_trial_sound_stim_times()
            self._task_sound_on_off = tuple(zip(self.sound_onset.task, self.sound_offset.task))
            return self.task_sound_on_off
        else:
            assert self.sam is not None
            # TODO get correct value for sound stim latency in beh boxes
            aud_stim_latency_estimate = 0 # s
            return self.sam.stimStartTimes + aud_stim_latency_estimate
    
    
    @property
    def rf_sound_on_off(self) -> Sequence[tuple[float, float]]:
        """Best estimate of offset of sound stim using openephys+microphone signals, if available.
        
        see onset for more info
        """
        if not self.sync:
            raise AttributeError(f'No sync file: rf times only available for ecephys experiments with sync: {self.session}')
        
        with contextlib.suppress(AttributeError):
            return self._rf_sound_on_off
        self.get_trial_sound_stim_times()
        self._rf_sound_on_off = tuple(zip(self.sound_onset.rf, self.sound_offset.rf))
        return self.rf_sound_on_off
    


def get_video_files(session: np_session.Session) -> dict[str, pathlib.Path]:
    """
    >>> session = np_session.Session('DRpilot_644864_20230201')
    >>> files = get_video_files(session)
    >>> len(files.keys())
    6
    """
    lims_name_to_path = {}
    for path in session.npexp_path.glob(
        '*[eye|face|side|behavior]*[.mp4|.json]'
    ):
        if path.is_dir():
            continue
        key = ''
        if path.suffix.lower() == '.mp4':
            if 'eye' in path.name.lower():
                key = 'eye_tracking'
            if 'face' in path.name.lower():
                key = 'face_tracking'
            if 'side' in path.name.lower() or 'behavior' in path.name.lower():
                key = 'behavior_tracking'
        if path.suffix.lower() == '.json':
            if 'eye' in path.name.lower():
                key = 'eye_cam_json'
            if 'face' in path.name.lower():
                key = 'face_cam_json'
            if 'side' in path.name.lower() or 'behavior' in path.name.lower():
                key = 'beh_cam_json'
        if not key:
            logger.debug(f'Skipping - not an expected raw video data mp4 or json: {path}')
            continue
        if key in lims_name_to_path:
            logger.debug(f'Multiple files found for {session} {key} - using largest')
            if path.stat().st_size < lims_name_to_path[key].stat().st_size:
                continue
        # assign new value
        lims_name_to_path[key] = path
    assert lims_name_to_path, f'No raw video data found: {session}'
    return lims_name_to_path


def get_sync_file_frame_times(
    session: np_session.Session,
) -> dict[Literal['beh', 'eye', 'face'], npt.NDArray[np.float64]]:

    labels = ('beh', 'eye', 'face')
    output = {}
    
    sync = get_sync_dataset(session)
    
    for cam in labels:
    
        cam_json = json.loads(
            get_video_files(session)[f'{cam}_cam_json'].read_bytes()
        )

        def extract_lost_frames_from_json(cam_json):
            lost_count = cam_json['RecordingReport']['FramesLostCount']
            if lost_count == 0:
                return []

            lost_string = cam_json['RecordingReport']['LostFrames'][0]
            lost_spans = lost_string.split(',')

            lost_frames = []
            for span in lost_spans:

                start_end = span.split('-')
                if len(start_end) == 1:
                    lost_frames.append(int(start_end[0]))
                else:
                    lost_frames.extend(
                        np.arange(int(start_end[0]), int(start_end[1]) + 1)
                    )

            return (
                np.array(lost_frames) - 1
            )   # you have to subtract one since the json starts indexing at 1 according to Totte

        exposure_sync_line_label_dict = {
            'Eye': 'eye_cam_exposing',
            'Face': 'face_cam_exposing',
            'Behavior': 'beh_cam_exposing',
        }

        cam_label = cam_json['RecordingReport']['CameraLabel']
        sync_line = exposure_sync_line_label_dict[cam_label]

        exposure_times = sync.get_rising_edges(
            sync_line, units='seconds'
        )

        lost_frames = extract_lost_frames_from_json(cam_json)

        frame_times = [
            e for ie, e in enumerate(exposure_times) if ie not in lost_frames
        ]
        output[cam] = np.array(frame_times)
    
    assert all(label in output for label in labels)
    return output

def get_sync_messages_text(session):
    for raw_folder in np_tools.get_raw_ephys_subfolders(session.npexp_path):
        for record_node_folder in raw_folder.glob('Record Node*'):
            largest_recording_folder = np_tools.get_single_oebin_path(record_node_folder).parent
            sync_messages_text = largest_recording_folder / 'sync_messages.txt'
            if sync_messages_text.exists():
                return sync_messages_text
    else:
        raise FileNotFoundError(f'No sync_messages.txt found in ephys raw data: {session = }')


def get_ephys_timing_info(session) -> dict[str, dict[str, int]]:
    """
    Start Time for Neuropix-PXI (107) - ProbeA-AP @ 30000 Hz: 210069564
    Start Time for Neuropix-PXI (107) - ProbeA-LFP @ 2500 Hz: 17505797
    Start Time for NI-DAQmx (109) - PXI-6133 @ 30000 Hz: 210265001
    
    >>> dirname_to_sample = get_ephys_timing_info(np_session.Session('DRpilot_626791_20220817_probeABCF'))
    >>> dirname_to_sample['NI-DAQmx-107.PXI-6133']
    {'start': 210265001, 'rate': 30000}
    """
    label = lambda line: ''.join(line.split('Start Time for ')[-1].split(' @')[0].replace(') - ', '.').replace(' (', '-'))
    sample = lambda line: int(line.strip(' ').split('Hz:')[-1])
    rate = lambda line: int(line.split('@ ')[-1].split(' Hz')[0])
    
    return {
        label(line): {
            'start': sample(line),
            'rate': rate(line),
        }
        for line in get_sync_messages_text(session).read_text().splitlines()[1:]
    } 
    
class EphysTimingOnPXI(NamedTuple):
    continuous: pathlib.Path
    """Abs path to device's folder within raw data/continuous/"""
    events: pathlib.Path
    """Abs path to device's folder within raw data/events/"""
    ttl: pathlib.Path
    """Abs path to device's folder within events/"""
    sampling_rate: float
    """Nominal sample rate reported in sync_messages.txt"""
    ttl_sample_numbers: npt.NDArray
    """Sample numbers on open ephys clock, after subtracting first sample reported in
    sync_messages.txt"""
    ttl_states: npt.NDArray
    """Contents of ttl/states.npy"""
    
def get_ephys_timing_on_pxi(session, only_dirs_including: str = '') -> Generator[EphysTimingOnPXI, None, None]:
    dirname_to_first_sample_number = get_ephys_timing_info(session) # includes name of each input device used (probe, nidaq)
    for raw_folder in np_tools.get_raw_ephys_subfolders(session.npexp_path):
        for record_node_folder in raw_folder.glob('Record Node*'):
            largest_recording_folder = np_tools.get_single_oebin_path(record_node_folder).parent
            for dirname in dirname_to_first_sample_number:
                if only_dirs_including not in dirname:
                    continue
                continuous = largest_recording_folder / 'continuous' / dirname
                if not continuous.exists():
                    continue
                events = largest_recording_folder / 'events' / dirname
                ttl = next(events.glob('TTL*'))
                first_sample_on_ephys_clock = dirname_to_first_sample_number[dirname]['start']
                sampling_rate = dirname_to_first_sample_number[dirname]['rate']
                ttl_sample_numbers = np.load(ttl / 'sample_numbers.npy') - first_sample_on_ephys_clock
                ttl_states = np.load(ttl / 'states.npy')
                yield EphysTimingOnPXI(
                    continuous, events, ttl, sampling_rate, ttl_sample_numbers, ttl_states
                )
                
def get_pxi_nidaq_mic_data(session) -> tuple[npt.NDArray[np.int16], EphysTimingOnSync]:
    """
    >>> mic_data, nidaq = get_pxi_nidaq_mic_data(np_session.Session('DRpilot_626791_20220817_probeABCF'))
    >>> mic_data.size
    (1, 138586329)
    >>> nidaq.sampling_rate
    30000
    """
    device = get_pxi_nidaq_device(session)
    if device.continuous.name.endswith('6133'):
        num_channels = 8
        speaker_channel, mic_channel = 1, 3
    else:
        raise IndexError(f'Unknown channel configuration for {device.continuous.name = }')
    dat = np.memmap(device.continuous / 'continuous.dat', dtype='int16', mode='r')
    data = np.reshape(dat, (int(dat.size / num_channels), -1)).T
    return data[mic_channel], next(get_ephys_timing_on_sync(session, device))
    
    
def get_pxi_nidaq_device(session: np_session.Session) -> EphysTimingOnPXI:
    """NI-DAQmx device info
    
    >>> device = get_ephys_nidaq_dirs(np_session.Session('DRpilot_626791_20220817_probeABCF'))
    >>> device.ttl.name
    'NI-DAQmx-107.PXI-6133'
    """
    device = tuple(get_ephys_timing_on_pxi(session, only_dirs_including='NI-DAQmx-'))
    if not device:
        raise FileNotFoundError(f'No */continuous/NI-DAQmx-*/ dir found in ephys raw data: {session = }')
    if device and len(device) != 1:
        raise FileNotFoundError(f'Expected a single NI-DAQmx folder to exist, but found: {[d.continuous for d in device]}')
    return device[0]

class EphysTimingOnSync(NamedTuple):
    device: EphysTimingOnPXI
    """Info with paths"""
    sampling_rate: float
    """Sample rate assessed on the sync clock"""
    start_time: float
    """First sample time (sec) relative to the start of the sync clock"""
    
def get_ephys_timing_on_sync(
    session, 
    devices: Optional[EphysTimingOnPXI | Iterable[EphysTimingOnPXI]] = None
) -> Generator[EphysTimingOnSync, None, None]:
    
    sync = get_sync_dataset(session)
    
    sync_barcode_times, sync_barcode_ids = ephys_utils.extract_barcodes_from_times(
        on_times=sync.get_rising_edges('barcode_ephys', units='seconds'),
        off_times=sync.get_falling_edges('barcode_ephys', units='seconds'),
    )
    if isinstance(devices, EphysTimingOnPXI):
        devices = (devices,)
        
    for device in devices or get_ephys_timing_on_pxi(session):
        
        ephys_barcode_times, ephys_barcode_ids = ephys_utils.extract_barcodes_from_times(
            on_times=device.ttl_sample_numbers[device.ttl_states > 0] / device.sampling_rate,
            off_times=device.ttl_sample_numbers[device.ttl_states < 0] / device.sampling_rate,
            )
        
        timeshift, sampling_rate, _ = ephys_utils.get_probe_time_offset(
            master_times=sync_barcode_times,
            master_barcodes=sync_barcode_ids,
            probe_times=ephys_barcode_times,
            probe_barcodes=ephys_barcode_ids,
            acq_start_index=0,
            local_probe_rate=device.sampling_rate,
            )
        start_time = -timeshift
        sampling_rate = sampling_rate
        if (np.isnan(sampling_rate)) | (~np.isfinite(sampling_rate)):
            sampling_rate = device.sampling_rate
            
        yield EphysTimingOnSync(
            device, sampling_rate, start_time
        )
# def xcorr_waveform_recording(*args):
    
    
if __name__=="__main__":
    x = DRDataLoader('DRpilot_644864_20230201')
    x.get_trial_sound_stim_times()
    x.sound_lags
    # doctest.testmod()