from __future__ import annotations

import argparse
import functools
import itertools
import logging
import pathlib
import sys
import tempfile
from typing import Optional, NamedTuple, Sequence

import allensdk.brain_observatory.sync_dataset as sync_dataset
import np_logging
import np_session
import np_tools
import numpy as np
import numpy.typing as npt
import pynwb


logger = logging.getLogger(__name__)

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
    timestamps: Sequence[int | float],
    min_gap: Optional[int | float] = None,
) -> tuple[Sequence[int | float], ...]:
    """
    Find the large gaps in timestamps and split at each gap.

    For example, if two blocks of stimuli were recorded in a single sync
    file, there will be one larger-than normal gap in timestamps.

    default min gap threshold: median + 6 * std (won't work well for short seqs)

    >>> reshape_into_blocks([0, 1, 2, 103, 104, 105], min_gap=100)
    ([0, 1, 2], [103, 104, 105])

    >>> reshape_into_blocks([0, 1, 2, 3])
    ([0, 1, 2, 3],)
    """
    intervals = np.diff(timestamps)
    threshold = (
        min_gap
        if min_gap is not None
        else (np.median(intervals) + 6 * np.std(intervals))
    )

    ends_of_blocks = []
    for interval in sorted(intervals, reverse=True):
        if interval > threshold:
            # large interval found
            ends_of_blocks.append(tuple(intervals).index(interval) + 1)
        else:
            break

    if not ends_of_blocks:
        return (timestamps,)

    blocks = []
    start = 0
    for end in ends_of_blocks:
        blocks.append(timestamps[start:end])
        start = end
    blocks.append(timestamps[start:])
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
