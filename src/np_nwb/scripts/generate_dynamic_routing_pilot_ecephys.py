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
    
    nwb_file = np_tools.init_nwb(
        np_session.Session(session_folder),
        description='Data and metadata for a Neuropixels ecephys experiment',
    )
    nwb_file = metadata.main(session_folder, nwb_file)
    
    
    
    ## Remove eye-tracking for now: need to correct sync files/cam frametimes for some sessions 
    # nwb_file = eye_tracking.add_to_nwb(
    #     np_session.Session(session_folder), nwb_file
    # )
    
    nwb_file = running.main(session, nwb_file)
    
    lick_times = utils.get_sync_dataset(session).get_rising_edges('lick_sensor', units='seconds')

    lick_nwb_data = ndx_events.Events(
        name="licks",
        timestamps=lick_times,
        description='times at which the subject interacted with a water spout',
    )
    nwb_file.add_acquisition(lick_nwb_data)
    
    # !trials/intervals should be added last in case timeseries need to be linked
    nwb_file = task_trials.main(session, nwb_file)
    nwb_file = rf_trials.main(session, nwb_file)
    
    with_lfp = 'allen' in session.npexp_path.as_posix() # arjun hasn't run LFP sub-sampling yet for sessions on synology nas 
    try:
        nwb_file, lfp_files = probes_and_units_and_lfp.add_to_nwb(session.npexp_path, nwb_file, with_lfp=with_lfp)
    except (IndexError, FileNotFoundError):
        print(f'skipping: no sorted data found: {session}')
        return
    
    ecephys_module = nwb_file.create_processing_module(
        name="ecephys", description="processed extracellular electrophysiology data"
    )
    lfp_container = pynwb.ecephys.LFP()
    ecephys_module.add(lfp_container)

    for probe in lfp_files:
        lfp_file = lfp_files[probe]
        linked_lfp_name = next((key for key in itertools.chain(lfp_file.acquisition, lfp_file.processing) if "lfp" in key), None)
        linked_lfp = lfp_file.acquisition.get(linked_lfp_name) or lfp_file.processing.get(linked_lfp_name)
        linked_electrical_series = linked_lfp[linked_lfp_name].get_electrical_series()
        existing_electrode_ids = tuple(e.index[0] for e in nwb_file.electrodes)
        df = nwb_file.electrodes.to_dataframe()
        electrodes = []
        for e in linked_electrical_series.electrodes:
            match = (
                df[(df['probe_channel_number']==int(e.probe_channel_number)) & (df['group_name']==e.group_name.values[0])].iloc[0]
                )
            electrodes.append(existing_electrode_ids.index(match.name))
        
        electrode_table_region = nwb_file.create_electrode_table_region(
            region=electrodes,
            name='electrodes',
            description=f"LFP channels on {probe}"
        )
        lfp_container.create_electrical_series(
            name=probe, 
            data=linked_electrical_series.data,
            electrodes=electrode_table_region,
            timestamps=linked_electrical_series.timestamps,
            channel_conversion=None,
            filtering='TODO add details of temporal, spatial sub-sampling plus any filtering',
            conversion=1.0, # use this if we store in uV
            comments='', 
            description=f'sub-sampled local field potential data from Neuropixels {probe}',
        )
    
    np_tools.save_nwb(nwb_file, output_file)
    return nwb_file


if __name__ == '__main__':
    
    skip_existing = True
    # if skip_existing is False:
    #     utils.clear_cache()
        
    sessions = itertools.chain(*(np_session.sessions(root=dir) for dir in np_session.DRPilotSession.storage_dirs))
    for session in sessions:
        
        if session.folder in (
            'DRpilot_644864_20230202', # no sorted data
            'DRpilot_644867_20230221', # 2 extra running datapoints
            'DRpilot_644866_20230207', # problem getting sound offsets from pxi nidaq
        ):
            continue
        
        if 'allen' not in session.npexp_path.as_posix():
            continue # no lfp for synology sessions - they are copied to workgroups/
        
        if isinstance(session, np_session.TempletonPilotSession):
            continue
        cache = utils.get_cache(session)
        output_file = cache / (f'{session}.nwb')
        if skip_existing and output_file.exists():
            print(f'skipping: {session}')
            continue
        main(session, output_file)
        
    # sys.argv.append('DRpilot_626791_20220817')
    # main(*utils.parse_cli_args()[::2])
