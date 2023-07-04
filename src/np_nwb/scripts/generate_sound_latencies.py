from __future__ import annotations

import doctest
import itertools
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
import np_probes.probes_to_nwb as probes_and_units_and_lfp
import numpy as np
from hdmf.backends.hdf5 import h5_utils
import ndx_events

import np_nwb.metadata.from_np_session as metadata
import np_nwb.dynamic_routing.utils as DR_utils
import np_nwb.dynamic_routing.running as running
import np_nwb.trials.DRtask as task_trials
import np_nwb.trials.RFtrials as rf_trials
import np_nwb.utils as utils


if __name__ == '__main__':
    logger = np_logging.getLogger()
    
    skip_existing = True
    # if skip_existing is False:
    #     utils.clear_cache()
        
    sessions = itertools.chain(*(np_session.sessions(root=dir) for dir in np_session.DRPilotSession.storage_dirs))
    for session in sessions:
        # 'DRpilot_644867_20230221', # 2 extra running datapoints
        
        if 'allen' not in session.npexp_path.as_posix():
            continue # no lfp for synology sessions - they are copied to workgroups/
        
        if isinstance(session, np_session.TempletonPilotSession):
            continue
        utils.USE_MIC_SIGNAL = True
        utils.DRDataLoader(session).get_trial_sound_onsets_offsets_lags()
        utils.USE_MIC_SIGNAL = False
        utils.DRDataLoader(session).get_trial_sound_onsets_offsets_lags()
        continue
        cache = utils.get_cache(session)
        output_file = cache / (f'{session}.nwb')
        if skip_existing and output_file.exists():
            print(f'skipping: {session}')
            continue
        main(session, output_file)
        
    # sys.argv.append('DRpilot_626791_20220817')
    # main(*utils.parse_cli_args()[::2])
