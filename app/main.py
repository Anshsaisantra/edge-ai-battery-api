"""
app/main.py -- STARTER battery API. Look for TODO.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 7, "params": 1281, "val_mae": 3.4}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    # TODO Mission 3: a doorbell will send you glitches like the ones you found in Mission 1.
    # Give every field a sensible range, e.g. Field(ge=..., le=...), so a glitch is refused
    # with HTTP 422 instead of being answered with a made-up number. Base the limits on what a
    # real sensor can report, not on your data's min and max: real readings must still get answers.
    # Limits come from the hardware, not the training data (allow_inf_nan=False also refuses NaN/inf).
    battery_v: float = Field(ge=2.5, le=4.5, allow_inf_nan=False)        # Li-ion: below ~2.5 V it is dead/cut off, max charge ~4.2-4.35 V
    temp_c: float = Field(ge=-55, le=125, allow_inf_nan=False)           # DS18B20 spec range; -127 is its error code
    rings_per_day: float = Field(ge=0, le=1000, allow_inf_nan=False)     # a count: never negative
    wifi_rssi_dbm: float = Field(ge=-120, lt=0, allow_inf_nan=False)     # RSSI is always negative; 0 means "no reading"
    age_months: float = Field(ge=0, le=240, allow_inf_nan=False)         # cannot be negative; 20 years is a generous ceiling


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


@app.get("/model_info")
def model_info():
    # TODO Mission 2 + 4: return your per-layer table, worked out FROM THE NETWORK (net.net),
    # not typed in by hand, so it stays right if you change the layer sizes. Exactly these keys:
    # {"layers": [{"name": "linear1", "in": 5, "out": 32, "params": ..., "bytes_f32": ..., "macs": ...}, ...],
    #  "total_params": ..., "total_bytes_f32": ..., "total_bytes_int8": ..., "total_macs": ...,
    #  "fits_300kb_sram_f32": true/false, "fits_300kb_sram_int8": true/false}
    # Count only the Linear layers (ReLU has no parameters). Clue: a Linear layer m has
    # m.in_features and m.out_features.
    return layer_table(net)


def layer_table(model, sram_bytes=300 * 1024):
    layers = []
    for m in model.net:
        if isinstance(m, torch.nn.Linear):
            a, b = m.in_features, m.out_features
            p = a * b + b
            layers.append({"name": f"linear{len(layers) + 1}", "in": a, "out": b,
                           "params": p, "bytes_f32": 4 * p, "macs": a * b})
    total = sum(l["params"] for l in layers)
    return {"layers": layers, "total_params": total, "total_bytes_f32": 4 * total,
            "total_bytes_int8": total, "total_macs": sum(l["macs"] for l in layers),
            "fits_300kb_sram_f32": 4 * total <= sram_bytes, "fits_300kb_sram_int8": total <= sram_bytes}


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": json.loads(e.json(include_input=False))})  # input may be NaN: not JSON-safe
    t["parse"] = ms(t0)
    rows = data.rows if isinstance(data, Batch) else [data]
    # TODO Mission 3: time each of the three stages below the same way "parse" is timed above,
    # storing them as t["normalise"], t["inference"] and t["build"].
    t0 = time.perf_counter()
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float32) - MU) / SD
    t["normalise"] = ms(t0)
    t0 = time.perf_counter()
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0)
    t0 = time.perf_counter()
    days = [round(float(v), 2) for v in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0)
    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


@app.post("/predict")
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    return await run(request, Batch)
