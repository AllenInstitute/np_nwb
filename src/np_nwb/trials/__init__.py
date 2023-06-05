import logging
import pathlib
import typing
import pynwb
from np_nwb_trials import nwb, processing

from ..utils import load_nwb_from_disk, write_nwb_to_disk

logger = logging.getLogger(__name__)


def append(
    session_folder: str | pathlib.Path,
    nwb_file: str | pathlib.Path | pynwb.NWBFile,
    output_file: typing.Optional[str | pathlib.Path] = None,
) -> pynwb.NWBFile:
    """Appends session related information as trials to nwb file
    """
    # ... process session_folder
    logger.debug("Proccessing storage directory...")
    trials_table = processing.storage_directory_to_trials_table(
        str(session_folder)
    )

    if not isinstance(nwb_file, pynwb.NWBFile):
        nwb_file = load_nwb_from_disk(nwb_file)

    # ... append new components to nwb_file
    logger.debug("Appending to NWB...")
    nwb.append_trials_to_nwb(
        trials_table,
        nwb_file,
    )

    if output_file is not None:
        write_nwb_to_disk(nwb_file, output_file)

    return nwb_file
