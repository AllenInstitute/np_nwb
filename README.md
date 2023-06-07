# np_nwb
Tools for quickly generating `.nwb` files from non-standard Mindscope Neuropixels experiments.

Generating an `.nwb` file will entail:
- inputting a path to a folder of raw data from a single experiment (`session_folder`)
- creating an instance of `pynwb.NWBFile` and writing to disk (`nwb_file`)
- passing `nwb_file` + `session_folder`, to various **modules** that add nwb components
- each module should:
  - accept or load an instance of `pynwb.NWBFile` from `nwb_file`
  - recognize the type of experiment contained in `session_folder` 
  - process raw data in `session_folder` accordingly
  - append tables to the `pynwb.NWBFile` instance
  - optionally write to disk
  - return the `pynwb.NWBFile` instance

Each module should therefore provide **a single function or method** which
implements the following `append()` interface:

```python
import logging
import pathlib
import tempfile
from typing import Optional

import np_tools
import pynwb

logger = logging.getLogger(__name__)


def append(
    session_folder: str | pathlib.Path,
    nwb_file: str | pathlib.Path | pynwb.NWBFile,
    output_file: Optional[str | pathlib.Path] = None,
    ) -> pynwb.NWBFile:
    """Append one or more new components to an `.nwb` file.

    - callable from within a Python process, by accepting & returning instances of `pynwb.NWBFile` 
    - callable from the command line, in which case all three input arguments are required, with `nwb_file` specified as a path
    """
    session_folder = pathlib.Path(session_folder)

    # ... process session_folder

    if not isinstance(nwb_file, pynwb.NWBFile):
        nwb_file = np_tools.load_nwb(nwb_file) 
    
    # ... append new components to nwb_file

    if output_file is not None:
        np_tools.save_nwb(nwb_file, output_file)
    
    return nwb_file
```
