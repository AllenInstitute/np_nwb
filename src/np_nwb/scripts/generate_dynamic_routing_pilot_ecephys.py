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
    obj = DR_utils.data_from_session(session)
    
    nwb_file = np_tools.init_nwb(
        np_session.Session(session_folder),
        description='Data and metadata for a Neuropixels ecephys experiment',
    )
    nwb_file = metadata.main(session_folder, nwb_file)
    nwb_file = eye_tracking.add_to_nwb(
        np_session.Session(session_folder), nwb_file
    )
    
    # !trials/intervals should be added last in case timeseries need to be linked
    nwb_file = task_trials.main(session, nwb_file)
    nwb_file = rf_trials.main(session, nwb_file)
    
    nwb_file = running.main(session, nwb_file)
    
    lick_times = utils.get_sync_dataset(session).get_rising_edges('lick_sensor', units='seconds')

    lick_nwb_data = ndx_events.Events(
        name="licks",
        timestamps=lick_times,
        description='Times at which the subject interacted with a water spout.',
    )
    nwb_file.add_acquisition(lick_nwb_data)
    
    ecephys_module = nwb_file.create_processing_module(
        name="ecephys", description="processed extracellular electrophysiology data"
    )
    
    link_lfp = False
    nwb_file, lfp_files = probes_and_units_and_lfp.add_to_nwb(session.npexp_path, nwb_file)
    
    open_files = {}
    # manager = pynwb.get_manager()
    for probe in lfp_files:
        lfp_file = output_file.parent / f'{session}_LFP_{probe}.nwb'
        # np_tools.save_nwb(lfp_files[probe], lfp_file)
        

        if link_lfp:
            open_files[probe] = pynwb.NWBHDF5IO(lfp_file, "r")
            io = open_files[probe].read()
            
            linked_lfp_name = next((key for key in itertools.chain(io.acquisition, io.processing) if "lfp" in key), None)
            
            # if linked_lfp_name:
            #     ecephys_module.add(io.acquisition.get(linked_lfp_name) or io.processing.get(linked_lfp_name))
            
            linked_lfp = io.acquisition.get(linked_lfp_name) or io.processing.get(linked_lfp_name)
            linked_electrical_series = linked_lfp[linked_lfp_name].get_electrical_series()
            lfp_electrical_series = pynwb.ecephys.ElectricalSeries(
                name="ElectricalSeries",
                data=h5_utils.H5DataIO(data=linked_electrical_series.data, link_data=True),
                electrodes=pynwb.core.DynamicTable('electrodes', description=linked_electrical_series.electrodes.description, data=h5_utils.H5DataIO(linked_electrical_series.electrodes.data, link_data=True)), # TODO find corresponding electrodes in main nwb
                timestamps=h5_utils.H5DataIO(data=linked_electrical_series.timestamps, link_data=True),
            )
            lfp = pynwb.ecephys.LFP(electrical_series=lfp_electrical_series, name=probe)
            ecephys_module.add(lfp)
    
    
    # use the same manager instance used to open the linked lfp
    if link_lfp:
        with pynwb.NWBHDF5IO(output_file, "w") as io:
            io.write(nwb_file, link_data=True)
            
        for io in open_files.values():
            io.close()
    else:
        np_tools.save_nwb(nwb_file, output_file)
        

    ####

    return nwb_file


if __name__ == '__main__':
    sys.argv.append('DRpilot_626791_20220817')
    main(*utils.parse_cli_args()[::2])
