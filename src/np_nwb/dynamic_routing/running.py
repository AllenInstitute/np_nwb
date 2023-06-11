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
from typing import Callable, Literal, Optional, Sequence, TypeVar
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
import np_nwb.dynamic_routing.utils as DR_utils

logger = logging.getLogger(__name__)

FRAMERATE = 60
"""Visual stim f.p.s - assumed to equal running wheel sampling rate. i.e. one
running wheel sample per camstim vsync"""

UNITS: Literal['cm/s', 'm/s'] = 'cm/s'
"""Running speed unit of measurement - NWB expects SI, but allows non-SI + a
`conversion` scalar"""

WHEEL_RADIUS: int | float
"""Value currently assigned from hdf5 file - units correspond to `UNITS`"""

LOWPASS_HZ = 4
"""Frequency for filtering running speed - filtered data stored in NWB `processing`, unfiltered
in `acquisition`"""

def main(
    session: interfaces.SessionFolder,
    nwb_file: interfaces.OptionalInputFile = None,
    output_file: interfaces.OptionalOutputFile = None,
) -> pynwb.NWBFile:
    """Add running wheel data from h5 file."""

    session, nwb_file, output_file = utils.parse_session_nwb_args(
        session, nwb_file, output_file
    )

    obj = DR_utils.data_from_session(session)
    """Sam's DR analysis - not needed currently"""

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
        
    running_speed, timestamps = get_running_speed_across_files(session, *h5_files, filt=None)
    filtered_running_speed, timestamps = get_running_speed_across_files(session, *h5_files, filt=lowpass_filter)
    # note: we can't simply filter the raw speed because it's not necessarily continuous 
    # - it was [potentially] recorded across multiple files, with a time gap between them
    
    raw = pynwb.TimeSeries(
            name='running',
            description='Linear forward running speed on a rotating disk.',
            data=running_speed,
            timestamps=timestamps,
            unit=UNITS,
            conversion=0.01 if UNITS == 'm/s' else 1.,
            comments=f'Assumes mouse runs at `radius = {WHEEL_RADIUS} {UNITS.split("/")[0]}` on disk.',
        )  # type: ignore

    nwb_file.add_acquisition(raw)
    
    filtered = pynwb.TimeSeries(
            name=raw.name,
            description=f'{raw.description} Low-pass filtered at {LOWPASS_HZ} Hz with a 3rd order Butterworth filter.',
            data=filtered_running_speed,
            timestamps=raw.timestamps,
            unit=UNITS,
            conversion=0.01 if UNITS == 'm/s' else 1.,
        )  # type: ignore
    
    module = nwb_file.processing.get('behavior') or nwb_file.create_processing_module(
        name="behavior", description="Processed behavioral data",
        )
    # module = nwb_file.processing.get('running') or nwb_file.create_processing_module(
    #     name="running", description="Processed running speed data",
    #     )
    module.add(filtered)
    
    return nwb_file


T = TypeVar('T')

def get_running_speed_across_files(
    session: np_session.Session, *h5_files: pathlib.Path,
    filt: Optional[Callable[[Sequence[T]], Sequence[T]]] = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Pools running speeds across files. Returns arrays of running speed and
    corresponding timestamps."""
    
    # we need timestamps for each frame (wheel encoder is read at every vsync)
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
        running_speed = np.concatenate((running_speed, filt(h5_data) if filt else h5_data))
        timestamps = np.concatenate((timestamps, timestamp_block))

    assert len(running_speed) == len(timestamps)
    return running_speed, timestamps


def get_running_speed(h5_file: pathlib.Path) -> npt.NDArray | None:
    """
    Running speed in m/s or cm/s (see `UNITS`).


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
        global WHEEL_RADIUS
        if UNITS == 'm/s':
            WHEEL_RADIUS = wheel_radius_cm / 100
        elif UNITS == 'cm/s':
            WHEEL_RADIUS = wheel_radius_cm
        else:
            raise ValueError(f'Unexpected units for running speed: {UNITS}')   
        speed = np.diff(
            wheel_revolutions * 2 * np.pi * WHEEL_RADIUS * FRAMERATE
        )
        # we lost one sample due to diff: pad with nan to keep same number of samples
        return np.concatenate(([np.nan], speed))   # type: ignore


def lowpass_filter(running_speed: npt.NDArray) -> npt.NDArray:
    """
    Careful not to filter discontiguous blocks of samples.
    See
    https://github.com/AllenInstitute/AllenSDK/blob/36e784d007aed079e3cad2b255ca83cdbbeb1330/allensdk/brain_observatory/behavior/data_objects/running_speed/running_processing.py
    """
    b, a = scipy.signal.butter(3, Wn=LOWPASS_HZ, fs=FRAMERATE, btype='lowpass')
    return scipy.signal.filtfilt(b, a, np.nan_to_num(running_speed))


if __name__ == '__main__':
    doctest.testmod()
    main(np_session.Session('DRpilot_644864_20230201'))
    main(*utils.parse_cli_args())
