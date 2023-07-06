from __future__ import annotations

import abc
import dataclasses
import datetime
import doctest
import functools
import itertools
import pathlib
import pickle
import tempfile
from typing import Any, Literal, Optional, Sequence, Union, Iterable
import typing
import uuid

import np_session
import pandas as pd
import upath

import np_nwb.database.base as base
import np_nwb.database.validated as validated
import np_nwb.database.json_db as json_db
from np_nwb.trials.DRtask import DRTaskTrials
import np_nwb.trials.RFtrials as RFtrials
from np_nwb.trials.property_dict import PropertyDict
import np_nwb.utils as utils

DB_ROOT = pathlib.Path(__file__).parent / 'json'
 
class MetadataFromNpSession:
    """Fetch metadata from an `np_session.Session` instance.
    
    >>> MetadataFromNpSession('DRpilot_626791_20220817').session.session_id
    'DRpilot_626791_20220817'
    >>> MetadataFromNpSession('DRpilot_626791_20220817').subject.subject_id
    '626791'
    """
    
    _session: np_session.Session
    _cache: pathlib.Path
    
    def __init__(self, session: str | np_session.Session) -> None:
        self._session = session if isinstance(session, np_session.Session) else np_session.Session(session)
        self._cache = utils.get_cache(self._session)
        
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
    
    def get_csv(self, obj, name) -> str:
        csv = pathlib.Path(tempfile.mkdtemp()) / f'{name}.csv'
        obj.to_dataframe().to_csv(csv, index=False)
        return str(csv)
    
    def get_column_names_to_descriptions(self, obj: PropertyDict) -> dict[str, str]:
        return {
            col['name']: col['description']
            for col in obj.to_add_trial_column()
        }
        
    @functools.cached_property
    def task_trials(self) -> validated.Intervals:
        name = 'task_trials'
        obj = DRTaskTrials(self._session)
        return validated.Intervals(
            name=name,
            column_names_to_descriptions=self.get_column_names_to_descriptions(obj),
            csv=self.get_csv(obj, name),
            description=(
                'auditory/visual routing behavior task trials: '
                'divided into sequential `blocks` that switch '
                'the reward contingency between `aud_target` and `vis_target`'
            ),
        )
        
    @functools.cached_property
    def vis_mapping_trials(self) -> validated.Intervals | None:
        name = 'vis_mapping_trials'
        try:
            obj = RFtrials.VisMappingTrials(self._session)
        except FileNotFoundError:
            return None
        return validated.Intervals(
            name=name,
            column_names_to_descriptions=self.get_column_names_to_descriptions(obj),
            csv=self.get_csv(obj, name),
            description="visual receptive-field mapping trials",
        )
        
    @functools.cached_property
    def aud_mapping_trials(self) -> validated.Intervals | None:
        name = 'aud_mapping_trials' 
        try:
            obj = RFtrials.AudMappingTrials(self._session)
        except FileNotFoundError:
            return None
        return validated.Intervals(
            name=name,
            column_names_to_descriptions=self.get_column_names_to_descriptions(obj),
            csv=self.get_csv(obj, name),
            description="auditory receptive-field mapping trials",
        )
    
    @property
    def intervals(self) -> dict[str, validated.Intervals]:
        intervals = tuple(
            i for i in (
            self.task_trials,
            self.vis_mapping_trials,
            self.aud_mapping_trials,
        ) if i is not None)
        return dict(zip((i.name for i in intervals), intervals))
        
    @functools.cached_property
    def raw(self) -> upath.UPath | None:
        date = self._session.start.strftime('%Y-%m-%d')
        mouse = self._session.mouse
        try:
            return next(upath.UPath('s3://aind-data-ecephys').glob(f'*{mouse}_{date}*'), None)
        except Exception:
            return None
        
    @functools.cached_property
    def nwb(self) -> upath.UPath | None:
        path = upath.UPath(f's3://aind-scratch-data/np_nwb/{self._session}/{self._session}.nwb')
        try:
            return path if path.exists() else None
        except Exception:
            return None
        
    @functools.cached_property
    def allen(self) -> pathlib.Path | None:
        return self._session.npexp_path
    
    @property
    def paths(self) -> validated.Paths:
        return validated.Paths(
            raw=self.raw if self.raw else None,
            allen=self.allen if self.allen else None,
            nwb=self.nwb if self.nwb else None,
            # TODO add sorted-s3 and nwb
        )
    
def add_pilot_sessions() -> None:
    sessions = itertools.chain(*(np_session.sessions(root=dir) for dir in np_session.DRPilotSession.storage_dirs))
    for session in sessions:
        if session.folder in (
            'DRpilot_644867_20230221', # 2 extra running datapoints
            'DRpilot_644866_20230207', # problem getting sound offsets from pxi nidaq
        ):
            continue
        
        if 'allen' not in session.npexp_path.as_posix():
            continue # no lfp for synology sessions - they are copied to workgroups/
        
        if isinstance(session, np_session.TempletonPilotSession):
            continue
        
        metadata = MetadataFromNpSession(session)
        json_db.JsonNwbMetadataDB(DB_ROOT).add(
            metadata
        )
            

if __name__ == "__main__":
    add_pilot_sessions()