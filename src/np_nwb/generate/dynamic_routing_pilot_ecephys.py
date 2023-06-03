from __future__ import annotations

import doctest
import pathlib
import sys
from typing import Optional

import np_eyetracking.dlc_lims.session_to_nwb as eye_tracking
import np_logging
import np_session
import pynwb

import np_nwb.init_with_metadata.from_np_session as init
import np_nwb.utils as utils

def main(
    session_folder: str | pathlib.Path | np_session.Session,
    output_file: Optional[str | pathlib.Path] = None,
    ) -> pynwb.NWBFile:
    """
    Initialize a `pynwb.NWBFile`, add experiment & subject metadata.
    
    `output_file` must be specified if called from command line, in order to
    save to disk.
    
    >>> nwb_file = main('DRpilot_626791_20220817')
    >>> isinstance(nwb_file, pynwb.NWBFile)
    True
    """
    nwb_file = init.main(session_folder, output_file)
    nwb_file = eye_tracking.add_to_nwb(np_session.Session(session_folder), nwb_file)
    
    utils.write_nwb_to_disk(nwb_file, output_file)
    
    return nwb_file


if __name__ == "__main__":
    np_logging.getLogger()
    doctest.testmod(raise_on_error=True)
    main(*sys.argv[1:])