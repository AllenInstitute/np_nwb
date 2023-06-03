"""
Use `np_session.Session` to initialize & add metadata to a `pynwb.NWBFile`.
"""
from __future__ import annotations

import datetime
import doctest
import pathlib
import sys
from typing import Optional
import uuid

import np_session
import pynwb

import np_nwb.utils as utils

def initialize(
    session: str | pathlib.Path | np_session.Session,
    description: str = 'Data and metadata for a Neuropixels session',
) -> pynwb.NWBFile:
    """Init `NWBFile` with minimum required arguments."""
    
    session = np_session.Session(session)

    return pynwb.NWBFile(
        session_description=description,
        identifier=str(uuid.uuid4()),  # globally unique for this nwb - not human-readable
        session_start_time=session.start,
    )
    
    
def add_metadata(
    session: str | pathlib.Path | np_session.Session,
    nwb_file: pynwb.NWBFile,
    **kwargs,
) -> pynwb.NWBFile:
    """Add metadata from `np_session`, plus any kwargs provided.
    
    See
    https://pynwb.readthedocs.io/en/stable/pynwb.file.html#pynwb.file.NWBFile
    for possible kwargs.
    """
    session = np_session.Session(session)
    
    nwb_file.session_id = session.folder
    if session.user:
        nwb_file.experimenter = tuple(session.user)
    nwb_file.institution = 'Allen Institute'
    nwb_file.lab = 'Mindscope, Neural Circuits and Behavior'
    
    for field, value in kwargs:
        setattr(nwb_file, field, value)
    
    return nwb_file


def add_subject(
    session: str | pathlib.Path | np_session.Session,
    nwb_file: pynwb.NWBFile,
) -> pynwb.NWBFile:
    """Add subject metadata from `np_session`"""
    session = np_session.Session(session)
    
    gender_id_to_str = {1: "M", 2: "F", 3: "U"}
    
    nwb_file.subject = pynwb.file.Subject(
        subject_id=str(session.mouse),
        description=session.mouse.lims['name'],
        date_of_birth=datetime.datetime.fromisoformat(session.mouse.lims['date_of_birth']),
        genotype=session.mouse.lims['full_genotype'],
        species='Mus musculus',
        sex=gender_id_to_str[session.mouse.lims['gender_id']],
    )
    return nwb_file

def main(
    session_folder: str | pathlib.Path | np_session.Session,
    output_file: Optional[str | pathlib.Path] = None,
    ) -> pynwb.NWBFile:
    """
    Initialize a `pynwb.NWBFile`, add experiment & subject metadata.
    
    `output_file` must be specified if called from command line, in order to
    save to disk.
    
    >>> nwb_file = main('DRpilot_644864_20230201')
    >>> isinstance(nwb_file, pynwb.NWBFile)
    True
    """
    
    if __name__ == "__main__" and output_file is None:
        raise TypeError('Missing required argument `output_file`')
    
    nwb_file = initialize(session_folder)
    nwb_file = add_metadata(session_folder, nwb_file)
    nwb_file = add_subject(session_folder, nwb_file)
    if output_file is not None:
        utils.write_nwb_to_disk(nwb_file, output_file)
    return nwb_file


if __name__ == "__main__":
    doctest.testmod()
    main(*sys.argv)