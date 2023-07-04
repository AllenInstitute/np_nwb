import abc
import datetime
import functools
import pathlib
from typing import Literal, Optional, Sequence, Union
import typing
import uuid

import pandas as pd
import pydantic
from pydantic import BaseModel as PydanticBaseModel
import upath
import upath.implementations.cloud

class BaseModel(PydanticBaseModel):
    class Config:
        validate_assignment = True
        json_encoders = {
            pathlib.Path: lambda v: v.as_posix(),
        }
        
class Subject(BaseModel):
    
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
    
    
class Session(BaseModel):
    
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


    
class Intervals(BaseModel):
    """Metadata for a set of intervals, e.g. task trials, session epochs, behavior
    events (licking), RF mapping trials, optotagging trials.
    
    Can be specified in a csv (where there are many intervals - think trials, licks) or as a list of
    `Interval`s (where there are few intervals, each with `tags` - think session epochs).
    """
    class Interval(BaseModel):
        """Metadata for a single interval, e.g. a single trial or epoch in the
        session.
        
        - `start_time` and `stop_time` are mandatory, all other columns are optional.
        - `_time` columns are for marking events within the interval [start_time:start_time], in seconds
        - `idx` columns are for indexing into arrays: should contain integers or NaNs
        - `is_` columns are booleans for filtering arrays where True: should contain True/False
        """ 
        start_time: float
        stop_time: float
        tags: Optional[Sequence[str]] = None
        
    name: str
    """Name of intervals module in nwb, e.g. 'licks' or 'trials'."""
    intervals: Optional[Sequence[Interval]] = None
    csv: Optional[str] = None
    column_names_to_descriptions: Optional[dict[str, str]] = None
    description: Optional[str] = None
    """Description of the intervals, reported in nwb."""
    
    @classmethod
    def get_column_names(cls, intervals: Union[Sequence[Interval], pathlib.Path, pd.DataFrame]) -> Sequence[str]:
        if isinstance(intervals, (str, pathlib.Path, upath.UPath)):
            names = tuple(pd.read_csv(str(intervals)).columns)
        elif isinstance(intervals, pd.DataFrame):
            names = tuple(intervals.columns)
        elif isinstance(intervals[0], cls.Interval): 
            names = tuple(name for name in intervals[0].dict().keys() if name != 'tags')
        else:
            raise TypeError(f'Unexpected {type(intervals) = }, expected path to csv, pd.DataFrame, or Sequence[Interval]')
        return tuple(name for name in names if name not in ('start_time', 'stop_time'))
    
    @property
    def _column_names(self) -> Sequence[str]:
        """Excl `start_time` and `stop_time`, which will be added automatically"""
        return self.get_column_names(self.intervals or self._df)
    
    @functools.cached_property
    def _df(self) -> Union[pd.DataFrame, None]:
        return pd.read_csv(str(self.csv)) if self.csv else None
    
    @pydantic.validator('intervals', allow_reuse=True)
    def check_interval_columns(cls, v):
        column_names = set(v[0].dict().keys())
        for interval in v:
            if set(interval.dict().keys()) != column_names:
                raise ValueError(f'All intervals must have the same columns')
        return v
    
    @pydantic.validator('intervals', 'csv', allow_reuse=True)
    def check_intervals_xor_csv(cls, v, field, values):
        other = 'csv' if field.name == 'intervals' else 'intervals'
        if not v and (other not in values or not values[other]):
            raise ValueError(f'Either "intervals" or "csv" must be specified')
        if v and (other in values and values[other]):
            raise ValueError(f'Only one of "intervals" or "csv" can be specified')
        return v
    
    @pydantic.validator('column_names_to_descriptions', pre=True, allow_reuse=True)
    def rm_start_stop_descriptions(cls, v):
        for col in ('start_time', 'stop_time'):
            if col in v:
                del v[col]
        return v
    
    @pydantic.validator('intervals', 'csv', allow_reuse=True)
    def check_column_descriptions_len(cls, v, values):
        column_names = cls.get_column_names(v)
        if 'column_names_to_descriptions' not in values:
            # hasn't been added yet
            return v
        current_columns_with_descriptions = values['column_names_to_descriptions'].keys()
        if set(column_names) != set(current_columns_with_descriptions):
            raise ValueError(f'missing descriptions for {set(column_names) ^ set(current_columns_with_descriptions)}')
        return v
    
    @pydantic.validator('csv', allow_reuse=True)
    def check_csv_readable(cls, v):
        try:
            pd.read_csv(str(v))
        except Exception as e:
            raise ValueError(f'Could not read {v}') from e
        return v
    
    @pydantic.validator('csv', allow_reuse=True)
    def check_mandatory_csv_columns(cls, v):
        df = pd.read_csv(str(v))
        if 'start_time' not in df.columns or 'stop_time' not in df.columns:
            raise ValueError(f'{v} must contain "start_time" and "stop_time" columns')
        return v
    
    @pydantic.validator('column_names_to_descriptions', allow_reuse=True)
    def check_descriptions_consistent_with_columns(cls, v, values):
        source = values.get('csv') or values.get('intervals')
        if source is None:
            # haven't been added yet, can't validate
            return v
        current_column_names = cls.get_column_names(source)
        if set(v.keys()) != set(current_column_names):
            raise ValueError(f'missing descriptions for {set(v.keys()) ^ set(current_column_names)}')
        return v
    
    # @pydantic.validator('column_names')
    # def check_start_time_and_stop_time(cls, v):
    #     if 'start_time' not in v or 'stop_time' not in v:
    #         raise ValueError('"column_names" must contain "start_time" and "stop_time"')
    #     return v
    
    # @pydantic.validator('column_names', 'column_descriptions', 'column_data')
    # def check_column_lengths(cls, v, field, values):
    #     if not all(
    #         len(v) == len(other_field_values) 
    #         for other_field_name, other_field_values in values.items() 
    #         if (other_field_name != field.name) and other_field_name.startswith('column_')
    #     ):
    #         raise ValueError(f'{tuple(values.keys())} must all have the same length')
    #     return v
    
    # @pydantic.validator('column_data')
    # def check_row_lengths(cls, v):
    #     if not all(
    #         len(row) == len(v[0]) 
    #         for row in v
    #     ):
    #         raise ValueError(f'All rows must have the same length')
    #     return v
    
class Paths(BaseModel):
    
    allen: Optional[pathlib.Path] = None
    """Path to raw data folder on /allen"""
    raw: Optional[upath.implementations.cloud.S3Path] = None
    """Path to folder with raw data, e.g. on S3 after upload with `aind-data-transfer`"""
    sorted: Optional[upath.implementations.cloud.S3Path] = None
    """Path to folder with spike-sorting output, e.g. data asset on CodeOcean"""
    nwb: Optional[upath.implementations.cloud.S3Path] = None
    """Current nwb file"""
    
    @pydantic.validator('raw', 'sorted', 'nwb', allow_reuse=True, pre=True)
    def check_path(cls, v):
        if v is None:
            return None
        v = upath.implementations.cloud.S3Path(v)
        if v is not None and not v.exists():
            return None
        return v