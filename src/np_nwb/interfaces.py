import typing
import pynwb
import pathlib
import np_session

SessionFolder = str | pathlib.Path | np_session.Session
OutputFile = str | pathlib.Path
OptionalOutputFile = typing.Optional[OutputFile] = None
AppendInterface = typing.Callable[[
    SessionFolder, OptionalOutputFile], pynwb.NWBFile]
