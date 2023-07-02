import abc
import datetime
from typing import Literal, Optional, Sequence, Union
import typing
import uuid

import pydantic


class Subject(pydantic.BaseModel):
    
    subject_id: str
    """Labtracks mouse ID, e.g. `366122`"""
    description: str
    """Abbreviated genotype and donor ID from lims.
    >>> np_session.Mouse(366122).lims['name']
    'C57BL/6J-366122'
    """
    date_of_birth: datetime.datetime
    """Converted by pynwb into age based on session date."""
    genotype: str
    """The genotype of the subject.
    >>> np_session.Mouse(366122).lims['full_genotype']
    'wt/wt'
    """
    sex: Literal['M', 'F', 'U']
    strain:  Optional[str] = None
    """The strain of the subject, e.g., `C57BL/6J`"""
    weight: Optional[float] = None
    """Weight of the subject in kg"""
    species: str = 'Mus musculus'
    
    
class Session(pydantic.BaseModel):
    
    session_start_time: datetime.datetime 
    session_id: str
    """np_session.Session.folder, e.g. `DRpilot_626791_20220817`"""
    
    session_description: str = "data and metadata for a Neuropixels ecephys session"
    lab: str = "Mindscope, Neural Circuits & Behavior"
    institution: str = "Allen Institute"
    experimenter: Optional[Sequence[
        Literal[
            "Bennett, Corbett",
            "Cabasco, Hannah",
            "Gale, Sam",
            "Kuyat, Jackie",
            "McBride, Ethan",
        ]
    ]] = None
        # plus anyone running behav boxes?
        # some files instead put all authors on related publication 
    experiment_description: Optional[str] = "dynamic routing aud/vis pilot"
    keywords: Sequence[str] = (
        "go-nogo",
        "context",
        "routing",
        "behavior",
        "neuropixels",
        "auditory",
        "visual",
    )
    notes: Optional[str] = None
    pharmacology: Optional[str] = None
    protocol: Optional[str] = None
    related_publications: Optional[str] = None
    source_script: Optional[str] = None
    source_script_file_name: Optional[str] = None
    data_collection: Optional[str] = None
    surgery: Optional[str] = None
    virus: Optional[str] = None
    stimulus_notes: Optional[str] = None

    
class Intervals(pydantic.BaseModel):
    name: str
    column_names: Sequence[str]
    column_descriptions: Sequence[str]
    column_data: Sequence[Sequence[Union[int, float, str, bool]]]
    
    @pydantic.validator('column_names')
    def check_start_time_and_end_time(cls, v):
        if 'start_time' not in v or 'end_time' not in v:
            raise ValueError('"column_names" must contain "start_time" and "end_time"')
        return v
    
    @pydantic.validator('column_names', 'column_descriptions', 'column_data')
    def check_column_lengths(cls, v, field, values):
        if not all(
            len(v) == len(other_field_values) 
            for other_field_name, other_field_values in values.items() 
            if (other_field_name != field.name) and other_field_name.startswith('column_')
        ):
            raise ValueError(f'{tuple(values.keys())} must all have the same length')
        return v
    
    @pydantic.validator('column_data')
    def check_row_lengths(cls, v):
        if not all(
            len(row) == len(v[0]) 
            for row in v
        ):
            raise ValueError(f'All rows must have the same length')
        return v
 