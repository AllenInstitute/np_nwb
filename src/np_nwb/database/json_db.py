from __future__ import annotations

import abc
import dataclasses
import datetime
import doctest
import functools
import importlib.resources
import json
import os
import pathlib
import shutil
import time
from typing import Any, ClassVar, Iterable, Iterator, Literal, Optional, Sequence, Union
import typing
import uuid

import fsspec
import upath

import np_nwb.database.base as base
import np_nwb.database.validated as validated


GITHUB_ROOT = upath.UPath('https://github.com/AllenInstitute/np_nwb/tree/main/db/json')
GITHUB_RAW_ROOT = upath.UPath('https://raw.githubusercontent.com/AllenInstitute/np_nwb/main/db/json')
# currently (July 2023) UPath's github implementation isn't completely
# implemented - we can just use http for now, with some manual switching
# - to glob/iterate over folders and find files, need to use `github.com`
# - to read file contents, need to switch path to `raw.githubusercontent.com` 

        
class JsonNwbMetadataDB(typing.MutableMapping):
    """Metadata database with dict-like interface, backed by JSON files.
    
    >>> JsonNwbMetadataDB().get('not-a-real-id')
    
    >>> JsonNwbMetadataDB().get('DRpilot_626791_20220817').subject.species
    'Mus musculus'
    
    >>> '626791' in JsonNwbMetadataDB().get_subject('626791')[0].session.session_id
    True
    
    >>> len(JsonNwbMetadataDB().get_all()) > 0
    True
    """
    
    root = pathlib.Path(__file__).parent.parent.parent.parent / 'db' / 'json'
    
    def __init__(self, path: Optional[str | pathlib.Path] = None) -> None:
        super().__init__()
        if path is not None:
            self.root = pathlib.Path(path)
    
    def __getitem__(self, key: str | int) -> base.NwbMetadata:
        path = self.root / str(key)
        if not path.exists():
            raise KeyError(key)
        interval_paths = path.glob('intervals/*.json')
        return base.NwbMetadata(
            subject=validated.Subject.parse_file(path / 'subject.json'),
            session=validated.Session.parse_file(path / 'session.json'),
            intervals=dict(zip(
                (interval.stem for interval in interval_paths),
                tuple(validated.Intervals.parse_file(interval_path) for interval_path in interval_paths),
            ),
            )
        )

    def __setitem__(self, key: str | int, metadata: base.NwbMetadata) -> None:
        path = self.root / str(key)
        path.mkdir(parents=True, exist_ok=True)
        for attr in ('subject', 'session', 'paths'):
            pathlib.Path(path / f'{attr}.json').write_text(getattr(metadata, attr).json(indent=4))
        if metadata.intervals:
            intervals_path = path / 'intervals'
            intervals_path.mkdir(parents=True, exist_ok=True)
            for interval in metadata.intervals.values():
                interval_json = pathlib.Path(intervals_path / f'{interval.name}.json')
                if (csv := interval.csv) and pathlib.Path(csv).exists():
                    csv = pathlib.Path(csv)
                    db_path = intervals_path / f'{interval.name}.csv'
                    if db_path.exists():
                        db_path.unlink(missing_ok=True)
                    shutil.copy(csv, db_path)
                    interval.csv = str(db_path)
                interval_json.write_text(interval.json(exclude={'_df', 'column_names'}, exclude_none=True, indent=4))
                if csv:
                    # now that csv has been validated by assignment, we can update its path in
                    # the json to point to github
                    existing = json.loads(interval_json.read_text())
                    existing['csv'] = str(GITHUB_RAW_ROOT / db_path.relative_to(self.root))
                    interval_json.write_text(
                        json.dumps(existing, indent=4)  
                    )
                
    def __delitem__(self, key: str | int) -> None:
        raise NotImplementedError
    
    def __iter__(self) -> Iterator[str]:
        return iter(path.stem for path in self.root.iterdir())
    
    def __len__(self) -> int:
        return len(list(self.root.iterdir()))
        
    def get_all(self) -> Sequence[base.NwbMetadata]:
        """Get metadata for all experiments."""
        return tuple(self[id] for id in self)
        
    def get_subject(self, subject: str | int) -> Sequence[base.NwbMetadata]:
        """Get metadata for all experiments for a single subject."""
        return tuple(self[id] for id in self if self[id].subject.subject_id == str(subject))
        
    def add(self, metadata: base.NwbMetadata) -> None:
        """Add or overwrite metadata in database."""
        self[metadata.session.session_id] = metadata
    
class GithubJsonMetadataDB(JsonNwbMetadataDB):
    """
    >>> v = GithubJsonMetadataDB()['DRpilot_626791_20220817']
    >>> v.session.session_start_time
    datetime.datetime(2022, 8, 17, 13, 24, 54)
    >>> v.subject.genotype
    'wt/wt'
    """
    fs: ClassVar[fsspec.AbstractFileSystem] # connect once to avoid exceeding github rate limit
    fs_root = 'db/json'
    
    def __hash__(self) -> int:
        return hash(self.fs_root)
    
    def get_jsons(self, session: str | int, filename: str = '*', subfolder: str = '') -> list[str]:
        """JSON files in root folder for the session, or in subfolders
        specified with `extra_path`."""
        self.connect()
        filename = f'{filename}.json' if not filename.endswith('.json') else filename
        if subfolder:
            filename = self.fs.sep.join((subfolder, filename))
        pattern = self.fs.sep.join((self.fs_root, session, filename))
        return self.fs.glob(pattern)
    
    @functools.cached_property
    def sessions(self) -> tuple[str]:
        """All available sessions on github"""
        self.connect()
        return tuple(name.replace(self.fs_root, '').strip('/') for name in self.fs.ls(self.fs_root))
    
    root = GITHUB_ROOT
    raw_root = GITHUB_RAW_ROOT
    
    @classmethod
    def connect(cls) -> None:
        if not hasattr(cls, 'fs'):
            cls.fs = fsspec.filesystem('github', org='AllenInstitute', repo='np_nwb', username=os.environ.get('GITHUB_USERNAME'), token=os.environ.get('GITHUB_TOKEN'))
        
    def __setitem__(self, key: str | int, metadata: base.NwbMetadata) -> None:
        raise NotImplementedError
    
    def __getitem__(self, key: str | int) -> base.NwbMetadata:
        self.connect()
        path = self.root / str(key)
        if not next(path.iterdir(), None):
            raise KeyError(key)
        interval_paths = self.get_jsons(session=key, filename='*', subfolder='intervals')
        intervals = tuple(validated.Intervals.parse_raw(self.fs.read_bytes(interval)) for interval in interval_paths)
        return base.NwbMetadata(
            subject=validated.Subject.parse_raw(self.fs.read_bytes(self.get_jsons(key, 'subject')[0])),
            session=validated.Session.parse_raw(self.fs.read_bytes(self.get_jsons(key, 'session')[0])),
            paths=validated.Paths.parse_raw(self.fs.read_bytes(self.get_jsons(key, 'paths')[0])),
            intervals=dict(zip((interval.name for interval in intervals), intervals)),
        )
        
    def __iter__(self) -> Iterator[str]:
        yield from self.sessions
    
      
      
if __name__ == "__main__":
    # os.environ['GITHUB_USERNAME'] = 'bjhardcastle'
    import h5py, pynwb
    entry = GithubJsonMetadataDB()['DRpilot_626791_20220815']
    h5 = h5py.File(entry.paths.nwb.open('rb'))
    nwb = pynwb.NWBHDF5IO(file=h5, load_namespaces=True).read()
    doctest.testmod()