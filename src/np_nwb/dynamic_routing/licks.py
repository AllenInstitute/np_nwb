import pynwb
import np_session
import ndx_events

import np_nwb.utils as utils
import np_nwb.dynamic_routing.utils as DR_utils

session = np_session.Session('DRpilot_644864_20230201')
obj = DR_utils.data_from_session(session)


lick_times = utils.get_sync_dataset(session).get_rising_edges('lick_sensor', units='seconds')

time_series = ndx_events.Events(
    name="licks",
    timestamps=lick_times,
    description='Times at which the subject interacted with a water spout.',
)

nwb_file.acquis
nwbfile.add_processing_module(rewards_mod)