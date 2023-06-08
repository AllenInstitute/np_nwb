"""
Use `np_session.Session` to initialize & add metadata to a `pynwb.NWBFile`.
"""
from __future__ import annotations

import datetime
import doctest
import itertools
import logging
import pathlib
import sys
import np_tools
from typing import Optional, Sequence
import uuid

import h5py
import np_session
import np_tools
import pynwb
import numpy as np
import numpy.typing as npt
import scipy.signal

import np_nwb.utils as utils
import np_nwb.interfaces as interfaces

logger = logging.getLogger(__name__)

FRAMERATE = 60
"""Visual stim f.p.s - assumed to equal running wheel sampling rate. i.e. one
running wheel sample per camstim vsync"""


def main(
    session: interfaces.SessionFolder,
    nwb_file: interfaces.OptionalInputFile = None,
    output_file: interfaces.OptionalOutputFile = None,
) -> pynwb.NWBFile:
    """Add running wheel data from h5 file."""

    session, nwb_file, output_file = utils.parse_session_nwb_args(
        session, nwb_file, output_file
    )

    # running wheel data is recorded with camstim output, so if we have
    # multiple pkl/h5 files, we need to pool their data:
    h5_files = []
    for h5_prefix in ('DynamicRouting', 'RFMapping'):
        h5_file = next(session.npexp_path.glob(f'{h5_prefix}*.hdf5'), None)
        if not h5_file:
            logger.warning(
                f'No h5 file matching {h5_prefix} in {session.npexp_path}'
            )
            continue
        h5_files.append(h5_file)
        
    running_speed, timestamps = get_filtered_running_speed_across_multiple_files(session, *h5_files)

    nwb_file = add_to_nwb(running_speed, timestamps, nwb_file)

    return nwb_file



def get_filtered_running_speed_across_multiple_files(
    session: np_session.Session, *h5_files: pathlib.Path
) -> tuple[npt.NDArray, npt.NDArray]:
    """Pools running speeds across files. Returns arrays of running speed and
    corresponding timestamps."""
    
    # we need timestamps for each frame (encoder reads for every vsync)
    # sync file has 'vsyncs', but there may be multiple h5 files with encoder
    # data per sync file (ie vsyncs are in blocks with a separating gap)
    timestamp_blocks = utils.get_blocks_of_frame_timestamps(session)

    running_speed = np.array([])
    timestamps = np.array([])

    for h5_file in h5_files:
        h5_data = get_running_speed(h5_file)
        if h5_data is None:
            continue
        for timestamp_block in timestamp_blocks:

            if len(timestamp_block) + 1 == len(h5_data):
                h5_data = h5_data[1:]

            if len(timestamp_block) == len(h5_data):
                timestamp_block = timestamp_block[1:]
                h5_data = h5_data[1:]
                timestamp_block = timestamp_block + 0.5 * np.median(
                    np.diff(timestamp_block)
                )
                break
        else:
            raise LookupError(
                f'No matching block of timestamps found for {h5_file} running data: length {len(h5_data)}, timestamp block lengths {[len(_) for _ in timestamp_blocks]}'
            )

        # we need to filter before pooling discontiguous blocks of samples
        running_speed = np.concatenate((running_speed, filter_running_speed(h5_data)))
        timestamps = np.concatenate((timestamps, timestamp_block))

    assert len(running_speed) == len(timestamps)
    return running_speed, timestamps


def get_running_speed(h5_file: pathlib.Path) -> npt.NDArray | None:
    """
    Running speed in cm/s.


    To align with timestamps, remove timestamp[0] and sample[0] and shift
    timestamps by half a frame (speed is estimated from difference between
    timestamps)

    See https://github.com/samgale/DynamicRoutingTask/blob/main/Analysis/DynamicRoutingAnalysisUtils.py
    """
    d = h5py.File(h5_file, 'r')
    if (
        'rotaryEncoder' in d
        and isinstance(d['rotaryEncoder'][()], bytes)
        and d['rotaryEncoder'].asstr()[()] == 'digital'
    ):
        assert d['frameRate'][()] == FRAMERATE
        wheel_revolutions = (
            d['rotaryEncoderCount'][:] / d['rotaryEncoderCountsPerRev'][()]
        )
        wheel_radius_cm = d['wheelRadius'][()]
        speed = np.diff(
            wheel_revolutions * 2 * np.pi * wheel_radius_cm * FRAMERATE
        )
        # we lost one sample due to diff: pad with nan to keep same number of samples
        return np.concatenate(([np.nan], speed))   # type: ignore


def filter_running_speed(running_speed: npt.NDArray):
    """
    Careful not to filter discontiguous blocks of samples.
    See
    https://github.com/AllenInstitute/AllenSDK/blob/36e784d007aed079e3cad2b255ca83cdbbeb1330/allensdk/brain_observatory/behavior/data_objects/running_speed/running_processing.py
    """
    b, a = scipy.signal.butter(3, Wn=4, fs=FRAMERATE, btype='lowpass')
    return scipy.signal.filtfilt(b, a, np.nan_to_num(running_speed))


def add_to_nwb(
    running_speed: npt.NDArray,
    timestamps: npt.NDArray,
    nwb_file: pynwb.NWBFile,
) -> pynwb.NWBFile:
    """Add filtered data to a 'nwb.processing['behavior']"""   
    
    behavior_module = nwb_file.processing.get('behavior') or nwb_file.create_processing_module(
        name="behavior", description="Processed behavioral data",
        )
    
    unit = 'cm/s'
    time_series = pynwb.TimeSeries(
            name='running',
            description='Linear forward running speed on a rotating disk. Low-pass filtered with a 3rd order Butterworth filter at 4 Hz.',
            data=running_speed,
            timestamps=timestamps,
            unit=unit,
            conversion=0.01 if unit == 'm/s' else 1.,
        )  # type: ignore

    behavior_module.add(time_series)
    
    return nwb_file


if __name__ == '__main__':
    doctest.testmod()
    main(np_session.Session('DRpilot_644864_20230201'))
    main(*utils.parse_cli_args())
