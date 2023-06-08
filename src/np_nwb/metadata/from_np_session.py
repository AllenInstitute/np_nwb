"""
Use `np_session.Session` to initialize & add metadata to a `pynwb.NWBFile`.
"""
from __future__ import annotations

import datetime
import doctest
import pathlib
import sys
import np_tools
from typing import Optional
import uuid

import np_session
import np_tools
import pynwb

import np_nwb.utils as utils


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

    gender_id_to_str = {1: 'M', 2: 'F', 3: 'U'}

    nwb_file.subject = pynwb.file.Subject(
        subject_id=str(session.mouse),
        description=session.mouse.lims['name'],
        date_of_birth=datetime.datetime.fromisoformat(
            session.mouse.lims['date_of_birth']
        ),
        genotype=session.mouse.lims['full_genotype'],
        species='Mus musculus',
        sex=gender_id_to_str[session.mouse.lims['gender_id']],
    )
    return nwb_file


def main(
    session_folder: str | pathlib.Path | np_session.Session,
    nwb_file: pynwb.NWBFile,
    output_file: Optional[str | pathlib.Path] = None,
) -> pynwb.NWBFile:
    """
    Add experiment & subject metadata to a `pynwb.NWBFile`.

    `output_file` must be specified if called from command line, in order to
    save to disk.

    >>> nwb_file = main('DRpilot_644864_20230201')
    >>> isinstance(nwb_file, pynwb.NWBFile)
    True
    """

    nwb_file = add_metadata(session_folder, nwb_file)
    nwb_file = add_subject(session_folder, nwb_file)
    if output_file is not None:
        np_tools.save_nwb(nwb_file, output_file)
    return nwb_file


if __name__ == '__main__':
    doctest.testmod()
    main(*sys.argv[1:])
