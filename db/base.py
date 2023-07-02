from __future__ import annotations

import abc
import dataclasses
import datetime
import doctest
import pathlib
from typing import Any, Iterable, Literal, Optional, Sequence, Union
import typing
import uuid

import validated
   

@dataclasses.dataclass
class NwbMetadata:
    subject: validated.Subject
    session: validated.Session
    intervals: Iterable[validated.Intervals]
     
        
class NwbMetadataDB(typing.Protocol):
    
    @abc.abstractmethod
    def get(self, id: str | int) -> NwbMetadata | None:
        """Get metadata related to one experiment based on some unique identifier."""
        
    @abc.abstractmethod
    def get_all(self) -> Sequence[NwbMetadata]:
        """Get metadata for all experiments."""
        
    @abc.abstractmethod
    def get_subject(self, subject: str | int) -> Sequence[NwbMetadata]:
        """Get metadata for all experiments for a single subject."""
    
    @abc.abstractmethod
    def add(self, metadata: NwbMetadata) -> None:
        """Add or overwrite metadata in database."""
    
    
if __name__ == "__main__":
    doctest.testmod()