from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np

def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    config["config_path"] = str(path)
    config["inputs"] = {k: str((path.parent / v).resolve()) if v else None for k,v in config["inputs"].items()}
    return config

def sha256(path):
    digest = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def code_hash():
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    return digest.hexdigest()

def json_write(path, obj):
    def default(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,default=default,allow_nan=False)+"\n")
    tmp.replace(path)

def probe_video(path):
    return json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(path)]))

def project(matrices, points):
    """Batch normalized-pitch coordinates -> pixels, with invalid values preserved."""
    homogeneous = np.concatenate([points,np.ones(points.shape[:-1]+(1,))],axis=-1)
    if points.ndim==3:
        out=np.einsum("nij,nkj->nki",matrices,homogeneous)
    else: out=np.einsum("nij,nj->ni",matrices,homogeneous)
    with np.errstate(invalid="ignore",divide="ignore"):
        return (out[...,:2]/out[...,2:]).astype(np.float32)
