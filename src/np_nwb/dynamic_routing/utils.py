from __future__ import annotations
import doctest
import itertools
import pathlib

from Analysis.DynamicRoutingAnalysisUtils import DynRoutData, baseDir
import np_session

def data_from_session(session_folder: str | pathlib.Path | np_session.Session) -> DynRoutData:
    """
    >>> obj = data_from_session('DRpilot_644864_20230201')
    >>> obj.taskVersion
    'stage 5 ori AMN moving'
    """
    session = np_session.Session(session_folder)
    glob = f'DynamicRouting*_{session.mouse}_{session.date:%Y%m%d}_*.hdf5'
    paths = (session.npexp_path, pathlib.Path(baseDir) / 'Data' / str(session.mouse))
    file = next(itertools.chain(*(path.glob(glob) for path in paths)), None)
    assert file, f'No file matching {glob} in {paths}'
    obj = DynRoutData()
    obj.loadBehavData(file)
    return obj


if __name__ == "__main__":
    doctest.testmod()