from __future__ import annotations

import doctest
import pathlib
import sys
from typing import Optional

import np_eyetracking.dlc_lims.session_to_nwb as eye_tracking
import np_logging
import np_tools
import np_session
import pynwb

import np_nwb.metadata.from_np_session as metadata
import np_nwb.utils as utils

def main(
    session_folder: str | pathlib.Path | np_session.Session,
    output_file: Optional[str | pathlib.Path] = None,
    ) -> pynwb.NWBFile:
    """
    Initialize a `pynwb.NWBFile`, add experiment & subject metadata.
    
    >>> nwb_file = main('DRpilot_626791_20220817')
    >>> isinstance(nwb_file, pynwb.NWBFile)
    True
    """
    nwb_file = np_tools.init_nwb(
        np_session.Session(session_folder),
        description='Data and metadata for a Neuropixels ecephys experiment',
        )
    nwb_file = metadata.main(session_folder, output_file)
    nwb_file = eye_tracking.add_to_nwb(np_session.Session(session_folder), nwb_file)
    
    np_tools.save_nwb(nwb_file, output_file)
    
    return nwb_file


if __name__ == "__main__":
    main(*utils.parse_cli_args()[::2])