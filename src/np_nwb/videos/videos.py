from __future__ import annotations

import collections.abc
import contextlib
import datetime
import doctest
import functools
import pathlib
import reprlib
import warnings
from typing import (Any, Generator, Iterable, Iterator, Literal, NamedTuple,
                    Optional, Sequence)

import allensdk.brain_observatory.sync_dataset as sync_dataset
import h5py
import np_logging
import np_session
import np_tools
import numpy as np
import pandas as pd
import pynwb
import np_eyetracking.dlc_lims.utils
from Analysis.DynamicRoutingAnalysisUtils import DynRoutData

import np_nwb.interfaces as interfaces
import np_nwb.utils as utils
from np_nwb.trials.property_dict import PropertyDict

logger = np_logging.getLogger(__name__)


def main(
    session: interfaces.SessionFolder,
    nwb_file: interfaces.OptionalInputFile = None,
    output_file: interfaces.OptionalOutputFile = None,
) -> pynwb.NWBFile:
    """Add trials table to nwb_file."""

    session, nwb_file, output_file = utils.parse_session_nwb_args(
        session, nwb_file, output_file
    )

    videos = np_eyetracking.dlc_lims.utils.get_video_files(session)
    timestamps = utils.get_sync_file_frame_times(session)
    descriptions = {
        'beh': 'Lateral view of subject\'s left side during session.',
        'eye': 'Macro view of subject\'s right eye during session.',
        'face': 'Frontal view of subject\'s face during session.',
    }
    for name in ['beh', 'eye', 'face']:
        
        external_file = videos[f'{"behavior" if name is "beh" else name}_tracking']#.relative_to(session.npexp_path)
        
        behavior_external_file = pynwb.image.ImageSeries(
            name=F'ExternalFile{"Behavior" if name is "beh" else name.capitalize()}Video',
            description=descriptions[name],
            # unit="n.a.",
            external_file=[str(external_file)],
            format="external",
            starting_frame=[0],
            timestamps=timestamps[name],
        )

        nwb_file.add_acquisition(behavior_external_file)
    np_tools.save_nwb(nwb_file, output_file)
    return nwb_file

if __name__ == "__main__":
    nwb_file = main('DRpilot_644864_20230131')