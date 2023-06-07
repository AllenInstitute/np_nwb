from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import tempfile
from typing import Optional, NamedTuple

import np_logging
import np_session
import np_tools
import pynwb

logger = logging.getLogger(__name__)


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
    parser.add_argument('session', type=np_session.Session, help='A path to a session folder, or an appropriate input argument to `np_session.Session()`, e.g. lims session id')
    parser.add_argument('nwb_filepath', nargs='?', default=None, type=pathlib.Path, help='A path to an existing .nwb file to append.')
    parser.add_argument('output_filepath', nargs='?', default=None, type=pathlib.Path, help='A path for saving the appended .nwb file, if different to the input path.')
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
            nwb_file = output or pathlib.Path(tempfile.mkdtemp()) / f'{session}.nwb'
        if output is None:
            # set output path to overwrite input path
            output = pathlib.Path(nwb_file)
        
    logger.info(f'Using {session!r}')
    logger.info(f'Using {nwb!r}')
    if output:
        logger.info(f'Writing appended nwb to {output}')
        
    return Info(session, nwb, output)
    