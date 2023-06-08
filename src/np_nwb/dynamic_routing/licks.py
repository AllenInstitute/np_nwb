import pynwb
import np_session

import np_nwb.dynamic_routing.utils as DR_utils
import np_nwb.utils as utils

session = np_session.Session('DRpilot_644864_20230201')
obj = DR_utils.data_from_session(session)


lick_times = utils.get_sync_dataset(session).get_rising_edges('lick_sensor', units='seconds')

time_series = pynwb.TimeSeries(
    name="reward",
    data=reward_amount,
    timestamps=obj.lickTimes,
    description="Water volume received by the subject as a reward",
    unit="mL",
)
behavioral_events = pynwb.behavior.BehavioralEvents(time_series=time_series, name="BehavioralEvents")

behavior_module.add(behavioral_events)


        reward_volume_ts = TimeSeries(
            name='volume',
            data=self.value['volume'].values,
            timestamps=self.value['timestamps'].values,
            unit='mL'
        )

        autorewarded_ts = TimeSeries(
            name='autorewarded',
            data=self.value['auto_rewarded'].values,
            timestamps=reward_volume_ts.timestamps,
            unit='mL'
        )

        rewards_mod = ProcessingModule('rewards',
                                       'Licking behavior processing module')
        rewards_mod.add_data_interface(reward_volume_ts)
        rewards_mod.add_data_interface(autorewarded_ts)
        nwbfile.add_processing_module(rewards_mod)