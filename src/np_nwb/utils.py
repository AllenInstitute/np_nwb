from __future__ import annotations

import argparse
import contextlib
import functools
import itertools
import json
import logging
import pathlib
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
from np_nwb.trials.property_dict import PropertyDict


logger = np_logging.getLogger(__name__)


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


def reshape_timestamps_into_blocks(
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
            ends_of_blocks.append(interval_index)
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
    return reshape_timestamps_into_blocks(frame_times)


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
        for vsyncs in reshape_timestamps_into_blocks(all_vsync_times):
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
        
        self._vsync_time_blocks = self.Times(main=vsync_times_in_blocks[self.main_stim_block_idx], rf=vsync_times_in_blocks[self.rf_block_idx] if self.rf_block_idx is not None else None)
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
        
        diode_rising_edges_in_blocks = reshape_timestamps_into_blocks(diode_rising_edges)
        diode_falling_edges_in_blocks = reshape_timestamps_into_blocks(diode_falling_edges)
        
        diode_rising_edges_in_blocks = self.Times(
            main=diode_rising_edges_in_blocks[self.main_stim_block_idx],
            rf=diode_rising_edges_in_blocks[self.rf_block_idx] if self.rf_block_idx is not None else None,
        )
        
        diode_falling_edges_in_blocks = self.Times(
            main=diode_falling_edges_in_blocks[self.main_stim_block_idx],
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
        
        self._frame_display_time_blocks = self.Times(main=diode_times_in_blocks['main'], rf=diode_times_in_blocks.get('rf', None))
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
        
        if self.frame_display_time_blocks.rf is None:
            raise AttributeError(f'A block of frametimes corresponding to rf mapping was not found for this experiment: {self.session}')
       
        return self.frame_display_time_blocks.rf


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
