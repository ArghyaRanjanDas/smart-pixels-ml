"""Normalize a VCD time origin using integer timestamps; reject invalid activity."""
from pathlib import Path
import os
import sys


def normalize(path):
    path=Path(path);temporary=path.with_name(path.name+'.t0.tmp')
    first=None;previous=None
    try:
        with path.open() as source,temporary.open('w') as output:
            for line in source:
                if line.startswith('#'):
                    stamp=int(line[1:].strip())
                    if stamp<0 or (previous is not None and stamp<previous):raise ValueError('Invalid decreasing VCD timestamps')
                    if first is None:first=stamp
                    previous=stamp;line=f'#{stamp-first}\n'
                output.write(line)
        if first is None or previous==first:raise ValueError('VCD has no nonzero activity time window')
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    return first


if __name__=='__main__':print(f'VCD origin shifted by {normalize(sys.argv[1])}')
