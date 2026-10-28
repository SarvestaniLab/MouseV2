from pathlib import Path
import shutil
import numpy as np


def format_model_weight(value):
    """Return the shortest lossless positional representation of a weight."""
    return np.format_float_positional(float(value), unique=True, trim="-")


def checkpoint_stem(weights, iteration):
    """Build a checkpoint stem without rounding small model weights to zero."""
    fields = [format_model_weight(value) for value in weights]
    return "y-{}-{}".format("-".join(fields), str(iteration).zfill(5))

def full_filename(fname, outdir, ext='.csv', suffix=''):
    if suffix:
        res = str((Path(outdir) / fname).resolve()) + "_" + suffix + ext
    else:
        res = str((Path(outdir) / fname).resolve())  + ext
    return res

def clean_outdir(outdir):
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    print("deleting all files in {dir}".format(dir=path))
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(str(child))
        else:
            child.unlink()

def logscale(x0, c, n):
    """logscale"""
    return [x0 * (c**i) for i in range(n)] 
