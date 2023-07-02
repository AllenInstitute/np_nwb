from __future__ import annotations

import abc
import dataclasses
import datetime
import doctest
import pathlib
from typing import Any, Iterable, Literal, Optional, Sequence, Union
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
    
    def __iter__(self) -> typing.Iterator[str | int]:
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
    >>> GithubJsonMetadataDB().get('DRpilot_626791_20220817').subject.species
    """
    # root = upath.UPath('http://github.com/alleninstitute/np_nwb/blob/main/db/json')
    root = upath.UPath('github://alleninstitute:np_nwb@main/') / 'db' / 'json'
    
if __name__ == "__main__":
    doctest.testmod()