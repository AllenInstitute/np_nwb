from __future__ import annotations

import doctest
import pathlib
import sys
from typing import Optional

import np_eyetracking.dlc_lims.session_to_nwb as eye_tracking
from np_nwb_trials import processing as trials_processing
from np_nwb_trials import nwb as trials
import np_logging
import np_tools
import np_session
import pynwb
from np_probes.probes_to_nwb import add_to_nwb as probes_and_units
import numpy as np

import np_nwb.metadata.from_np_session as metadata
import np_nwb.dynamic_routing.utils as DR_utils
import np_nwb.dynamic_routing.running as running
import np_nwb.utils as utils
import ndx_events


logger = np_logging.getLogger(__name__)

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
    session = np_session.Session(session_folder)
    obj = DR_utils.data_from_session(session)
    
    nwb_file = np_tools.init_nwb(
        np_session.Session(session_folder),
        description='Data and metadata for a Neuropixels ecephys experiment',
    )
    nwb_file = metadata.main(session_folder, nwb_file)
    nwb_file = eye_tracking.add_to_nwb(
        np_session.Session(session_folder), nwb_file
    )
    
    nwb_file = running.main(session, nwb_file)
    
    lick_times = utils.get_sync_dataset(session).get_rising_edges('lick_sensor', units='seconds')

    lick_nwb_data = ndx_events.Events(
        name="licks",
        timestamps=lick_times,
        description='Times at which the subject interacted with a water spout.',
    )
    nwb_file.add_acquisition(lick_nwb_data)
    
    nwb_file = probes_and_units(session.npexp_path, nwb_file)
    
    import np_nwb.trials.DRtask as task_trials
    
    nwb_file = task_trials.main(nwb_file)
    
    np_tools.save_nwb(nwb_file, output_file)

    return nwb_file


if __name__ == '__main__':
    sys.argv.append('DRpilot_626791_20220817')
    main(*utils.parse_cli_args()[::2])
