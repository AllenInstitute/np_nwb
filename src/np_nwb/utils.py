from __future__ import annotations

import logging
import pathlib
import tempfile
from typing import Optional

import pynwb

logger = logging.getLogger(__name__)

def load_nwb_from_disk(
    nwb_path: str | pathlib.Path,
    ) -> pynwb.NWBFile:
    logger.info(f'Loading nwb file at {nwb_path}')
    with pynwb.NWBHDF5IO(nwb_path, mode='r') as f:
        return f.read()


def write_nwb_to_disk(
    nwb_file: pynwb.NWBFile, output_path: Optional[str | pathlib.Path] = None
    ) -> None:
    if output_path is None:
        output_path = pathlib.Path(tempfile.mkdtemp()) / f'{nwb_file.session_id}.nwb'
    
    nwb_file.set_modified()

    logger.info(f'Writing nwb file `{nwb_file.session_id!r}` to {output_path}')
    with pynwb.NWBHDF5IO(output_path, mode='w') as f:
        f.write(nwb_file, cache_spec=True)
    logger.debug(f'Writing complete for nwb file `{nwb_file.session_id!r}`')