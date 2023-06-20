import pynwb

rf_mapping = pynwb.epoch.TimeIntervals(
    name="rf_mapping",
    description="Intervals for each receptive-field mapping trial",
)

rf_mapping.add_column(name="stage", description="stage of sleep")
rf_mapping.add_column(name="confidence", description="confidence in stage (0-1)")

rf_mapping.add_row(start_time=0.3, stop_time=0.5, stage=1, confidence=0.5)
rf_mapping.add_row(start_time=0.7, stop_time=0.9, stage=2, confidence=0.99)
rf_mapping.add_row(start_time=1.3, stop_time=3.0, stage=3, confidence=0.7)

nwbfile.add_time_intervals(rf_mapping)