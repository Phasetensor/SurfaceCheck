"""Portable reports with explicit evidence categories and input fingerprints."""
from __future__ import annotations

import csv
from dataclasses import asdict,dataclass,is_dataclass
from hashlib import sha256
from html import escape
from importlib.metadata import version
import io
import json
from datetime import datetime
from typing import Any
import numpy as np


def native(x: Any) -> Any:
    if is_dataclass(x) and not isinstance(x,type):
        return native(asdict(x))
    if isinstance(x,np.ndarray):
        return native(x.tolist())
    if isinstance(x,np.generic):
        return native(x.item())
    if isinstance(x,datetime):
        return x.isoformat()
    if isinstance(x,dict):
        return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [native(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):
        return {"undefined":str(x)}
    return x


@dataclass(frozen=True)
class Report:
    valuation: datetime
    provenance: str
    conventions: dict
    diagnostics: dict
    inputs: Any

    def render(self,fmt: str="json") -> str:
        if self.valuation.utcoffset() is None or not self.provenance.strip():
            raise ValueError("Valuation timestamp and provenance required")
        inputs=json.dumps(native(self.inputs),sort_keys=True,allow_nan=False,separators=(",",":"))
        data=dict(valuation=self.valuation.isoformat(),provenance=self.provenance,conventions=self.conventions,diagnostics=self.diagnostics,input_sha256=sha256(inputs.encode()).hexdigest(),versions={n:version(n) for n in ("scipy","numpy")})
        rows={k:json.dumps(native(v),sort_keys=True,allow_nan=False) for k,v in data.items()}
        if fmt=="json":
            return json.dumps(native(data),sort_keys=True,allow_nan=False)
        if fmt=="html":
            return "<table>"+ "".join("<tr><th>"+escape(k)+"</th><td>"+escape(v)+"</td></tr>" for k,v in rows.items())+"</table>"
        if fmt=="csv":
            out=io.StringIO()
            writer=csv.writer(out)
            writer.writerow(("field","json_value"))
            writer.writerows(rows.items())
            return out.getvalue()
        raise ValueError("Unknown output format")
