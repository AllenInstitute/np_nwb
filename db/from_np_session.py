from __future__ import annotations

import abc
import dataclasses
import datetime
import doctest
import itertools
import pathlib
from typing import Any, Literal, Optional, Sequence, Union, Iterable
import typing
import uuid

import np_session

import base
import validated
import json_db
   
class MetadataFromNpSession:
    """Fetch metadata from an `np_session.Session` instance.
    
    >>> MetadataFromNpSession('DRpilot_626791_20220817').session.session_id
    'DRpilot_626791_20220817'
    >>> MetadataFromNpSession('DRpilot_626791_20220817').subject.subject_id
    '626791'
    """
    
    _session: np_session.Session
    
    def __init__(self, session: str | np_session.Session) -> None:
        self._session = session if isinstance(session, np_session.Session) else np_session.Session(session)
    
    @property
    def session(self) -> validated.Session:
        return validated.Session(
            session_id=self._session.folder,
            session_start_time=self._session.start,
            experimenter=tuple(self._session.user) if self._session.user else None,
        )
        
    @property
    def subject(self) -> validated.Subject:
        gender_id_to_str = {1: 'M', 2: 'F', 3: 'U'}
        return validated.Subject(
            subject_id=str(self._session.mouse),
            description=self._session.mouse.lims['name'],
            date_of_birth = datetime.datetime.fromisoformat(
                self._session.mouse.lims['date_of_birth']
            ),
            genotype=self._session.mouse.lims['full_genotype'],
            sex=gender_id_to_str[self._session.mouse.lims['gender_id']],
        )
    
    @property
    def intervals(self) -> Iterable[validated.Intervals] | None:
        return None
        
def add_pilot_sessions() -> None:
    for session in itertools.chain(*(np_session.sessions(root=dir) for dir in np_session.DRPilotSession.storage_dirs)):
        try:
            json_db.JsonNwbMetadataDB().add(MetadataFromNpSession(session))
        except Exception as e:
            print(f'Error adding {session}: {e!r}')

if __name__ == "__main__":
    add_pilot_sessions()