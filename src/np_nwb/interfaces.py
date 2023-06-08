from __future__ import annotations

import pynwb
import pathlib
import np_session
from typing import Callable, Optional, Union

SessionFolder = Union[str, pathlib.Path, np_session.Session]
"""Path to session raw data, or an instance of `np_session.Session`"""
InputFile = Union[str, pathlib.Path, pynwb.NWBFile]
"""Path to a .nwb file, or an instance of `pynwb.NWBFile`"""
OutputFile = Union[str, pathlib.Path]
OptionalInputFile = Optional[InputFile]
OptionalOutputFile = Optional[OutputFile]

AppendInterface = Callable[
    [SessionFolder, InputFile, OptionalOutputFile], pynwb.NWBFile
]
