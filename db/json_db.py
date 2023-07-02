from __future__ import annotations

import abc
import dataclasses
import datetime
import doctest
import pathlib
from typing import Any, Iterable, Iterator, Literal, Optional, Sequence, Union
import typing
import uuid

import upath

import base
import validated

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
    
    root = pathlib.Path(__file__).parent / 'json'
    
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
            intervals=tuple(validated.Intervals.parse_file(interval_path) for interval_path in interval_paths),
        )

    def __setitem__(self, key: str | int, metadata: base.NwbMetadata) -> None:
        path = self.root / str(key)
        path.mkdir(parents=True, exist_ok=True)
        for attr in ('subject', 'session'):
            pathlib.Path(path / f'{attr}.json').write_text(getattr(metadata, attr).json(indent=4))
        if metadata.intervals:
            intervals_path = path / 'intervals'
            intervals_path.mkdir(parents=True, exist_ok=True)
            for interval in metadata.intervals:
                pathlib.Path(intervals_path / f'{interval.name}.json').write_text(interval.json(indent=4))
                
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
    >>> GithubJsonMetadataDB()['DRpilot_626791_20220817'].session.session_start_time
    datetime.datetime(2022, 8, 17, 13, 24, 54)
    >>> GithubJsonMetadataDB().get('DRpilot_626791_20220817').subject.subject_id
    '626791'
    """
    root = upath.UPath('https://github.com/AllenInstitute/np_nwb/tree/main/db/json')
    raw_root = upath.UPath('https://raw.githubusercontent.com/AllenInstitute/np_nwb/main/db/json')
    # currently (July 2023) UPath's github implementation isn't completely
    # implemented - we can just use http for now, with some manual switching
    # - to glob/iterate over folders and find files, need to use `github.com`
    # - to read file contents, need to switch path to `raw.githubusercontent.com` 
        
    def __setitem__(self, key: str | int, metadata: base.NwbMetadata) -> None:
        raise NotImplementedError
    
    def __getitem__(self, key: str | int) -> base.NwbMetadata:
        path = self.root / str(key)
        if not next(path.iterdir(), None):
            raise KeyError(key)
        interval_paths = path.glob('intervals/*.json')
        raw_path = self.raw_root / str(key) # for reading files
        return base.NwbMetadata(
            subject=validated.Subject.parse_raw((raw_path / 'subject.json').read_bytes()),
            session=validated.Session.parse_raw((raw_path / 'session.json').read_bytes()),
            intervals=tuple(validated.Intervals.parse_raw((raw_path / interval_path.relative_to(path)).read_bytes()) for interval_path in interval_paths),
        )
        
    def __iter__(self) -> Iterator[str]:
        return iter(path.stem for path in self.root.iterdir() if path.stem[0] not in '.#_' and not path.suffix)
    
      
      
if __name__ == "__main__":
    doctest.testmod()