# =============================
# File: src/chaos_llm/services/al_uls_client.py
# =============================
import os
import time
import asyncio
from typing import Dict, Any, List, Tuple
import httpx

JULIA_SERVER_URL = os.environ.get("JULIA_SERVER_URL", "http://localhost:8088")
CACHE_TTL_SECONDS = float(os.environ.get("ALULS_HTTP_TTL", 30))

class TTLCache:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self._store: Dict[Tuple[str, Tuple[str, ...]], Tuple[float, Dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0

    def _now(self) -> float:
        return time.monotonic()

    def _key(self, name: str, args: List[str]) -> Tuple[str, Tuple[str, ...]]:
        return (name.upper(), tuple(args))

    def get(self, name: str, args: List[str]) -> Dict[str, Any] | None:
        k = self._key(name, args)
        v = self._store.get(k)
        if not v:
            self.misses += 1
            return None
        ts, data = v
        if self._now() - ts <= self.ttl:
            self.hits += 1
            return data
        self._store.pop(k, None)
        self.misses += 1
        return None

    def set(self, name: str, args: List[str], value: Dict[str, Any]) -> None:
        self._store[self._key(name, args)] = (self._now(), value)

    def stats(self) -> Dict[str, Any]:
        return {"entries": len(self._store), "hits": self.hits, "misses": self.misses, "ttl": self.ttl}

class ALULSClient:
    def __init__(self, base_url: str | None = None):
        self.base = base_url or JULIA_SERVER_URL
        self.client = httpx.AsyncClient(timeout=10)
        self.cache = TTLCache(CACHE_TTL_SECONDS)

    async def health(self) -> Dict[str, Any]:
        try:
            r = await self.client.get(f"{self.base}/health")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def parse(self, text: str) -> Dict[str, Any]:
        try:
            r = await self.client.post(f"{self.base}/v1/symbolic/parse", json={"text": text})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def eval(self, name: str, args: List[str]) -> Dict[str, Any]:
        cached = self.cache.get(name, args)
        if cached is not None:
            return {**cached, "_cached": True}
        try:
            r = await self.client.post(f"{self.base}/v1/symbolic/eval", json={"name": name, "args": args})
            r.raise_for_status()
            data = r.json()
            if data.get("ok"):
                self.cache.set(name, args, data)
            return data
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def batch_eval(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Use cache per-call; run only misses concurrently
        to_run: List[Tuple[int, Dict[str, Any]]] = []
        results: List[Dict[str, Any]] = [{} for _ in calls]
        for i, c in enumerate(calls):
            name = c.get("name", "").upper(); args = c.get("args", [])
            cached = self.cache.get(name, args)
            if cached is not None:
                results[i] = {**cached, "_cached": True}
            else:
                to_run.append((i, {"name": name, "args": args}))
        tasks = [self.eval(c["name"], c["args"]) for _, c in to_run]
        outs = await asyncio.gather(*tasks, return_exceptions=True)
        for (i, _), out in zip(to_run, outs):
            results[i] = out if not isinstance(out, Exception) else {"ok": False, "error": str(out)}
        return results

al_uls_client = ALULSClient()

# =============================
# File: src/chaos_llm/services/al_uls_ws_client.py
# =============================
import os
import json
import asyncio
from typing import Dict, Any, List, Tuple
import websockets

JULIA_WS_URL = os.environ.get("JULIA_WS_URL", "ws://localhost:8089")
CACHE_TTL_WS = float(os.environ.get("ALULS_WS_TTL", 30))

class TTLCacheWS:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self._store: Dict[Tuple[str, Tuple[str, ...]], Tuple[float, Dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0

    def _now(self) -> float:
        return asyncio.get_event_loop().time()

    def _key(self, name: str, args: List[str]) -> Tuple[str, Tuple[str, ...]]:
        return (name.upper(), tuple(args))

    def get(self, name: str, args: List[str]) -> Dict[str, Any] | None:
        k = self._key(name, args)
        v = self._store.get(k)
        if not v:
            self.misses += 1; return None
        ts, data = v
        if self._now() - ts <= self.ttl:
            self.hits += 1; return data
        self._store.pop(k, None)
        self.misses += 1; return None

    def set(self, name: str, args: List[str], value: Dict[str, Any]) -> None:
        self._store[self._key(name, args)] = (self._now(), value)

    def stats(self) -> Dict[str, Any]:
        return {"entries": len(self._store), "hits": self.hits, "misses": self.misses, "ttl": self.ttl}

class ALULSWSClient:
    def __init__(self, ws_url: str | None = None):
        self.ws_url = ws_url or JULIA_WS_URL
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.cache = TTLCacheWS(CACHE_TTL_WS)

    async def connect(self):
        if (self.websocket is None) or self.websocket.closed:
            self.websocket = await websockets.connect(self.ws_url)
        return self.websocket

    async def _roundtrip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ws = await self.connect()
            await ws.send(json.dumps(payload))
            resp = await ws.recv()
            return json.loads(resp)
        except Exception as e:
            # Reset socket on error to force reconnect later
            try:
                if self.websocket:
                    await self.websocket.close()
            finally:
                self.websocket = None
            return {"ok": False, "error": str(e)}

    async def parse(self, text: str) -> Dict[str, Any]:
        return await self._roundtrip({"type": "parse", "text": text})

    async def eval(self, name: str, args: List[str]) -> Dict[str, Any]:
        cached = self.cache.get(name, args)
        if cached is not None:
            return {**cached, "_cached": True}
        res = await self._roundtrip({"type": "eval", "name": name, "args": args})
        if isinstance(res, dict) and res.get("ok"):
            self.cache.set(name, args, res)
        return res

    async def batch_eval(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # try a single WS roundtrip; if it fails or invalid, fall back per-call
        res = await self._roundtrip({"type": "batch_eval", "calls": calls})
        if isinstance(res, dict) and "results" in res and isinstance(res["results"], list):
            # populate cache for successes
            out: List[Dict[str, Any]] = []
            for c, r in zip(calls, res["results"]):
                if isinstance(r, dict) and r.get("ok"):
                    self.cache.set(c.get("name", ""), c.get("args", []), r)
                out.append(r if isinstance(r, dict) else {"ok": False, "error": "invalid item"})
            return out
        # fallback: per-call
        return [await self.eval(c.get("name", ""), c.get("args", [])) for c in calls]

al_uls_ws_client = ALULSWSClient()

# =============================
# File: src/chaos_llm/services/al_uls.py
# =============================
import os
from typing import Dict, Any, List, Optional
from .al_uls_client import al_uls_client
from .al_uls_ws_client import al_uls_ws_client
import re

CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)$")
PREFER_WS = os.environ.get("ALULS_PREFER_WS", "1") in {"1", "true", "TRUE", "yes"}

class ALULS:
    def is_symbolic_call(self, text: str) -> bool:
        return bool(CALL_RE.search((text or "").strip()))

    def parse_symbolic_call(self, text: str) -> Dict[str, Any]:
        m = CALL_RE.search((text or "").strip())
        if not m:
            return {"name": None, "args": []}
        name, argstr = m.group(1), m.group(2)
        args = [a.strip() for a in argstr.split(",") if a.strip()]
        return {"name": name.upper(), "args": args}

    async def health(self) -> Dict[str, Any]:
        # Only HTTP has /health; use it as liveness check
        return await al_uls_client.health()

    async def eval_symbolic_call_async(self, call: Dict[str, Any]) -> Dict[str, Any]:
        name = call.get("name", ""); args = call.get("args", [])
        if PREFER_WS:
            res = await al_uls_ws_client.eval(name, args)
            if isinstance(res, dict) and (res.get("ok") or res.get("_cached")):
                return res
        return await al_uls_client.eval(name, args)

    async def batch_eval_symbolic_calls(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if PREFER_WS:
            res = await al_uls_ws_client.batch_eval(calls)
            # If any valid item present, accept; else fallback
            if isinstance(res, list) and any(isinstance(r, dict) for r in res):
                return res
        return await al_uls_client.batch_eval(calls)

al_uls = ALULS()

# =============================
# File: src/chaos_llm/services/qgi.py (excerpt showing async apply)
# =============================
from typing import Any, Dict, List
from .entropy_engine import entropy_engine
from .matrix_processor import matrix_processor
from .al_uls import al_uls
from .motif_engine import motif_engine
from .suggestions import SUGGESTIONS


def _prefix_match(prefix: str, state: str) -> List[str]:
    pre = (prefix or "").upper(); pool = SUGGESTIONS.get(state, [])
    return [t for t in pool if t.startswith(pre)]


def _apply_token_to_qgi(qgi: Dict[str, Any], token_text: str) -> None:
    entropy_score = entropy_engine.score_token(token_text)
    volatility_signal = entropy_engine.get_volatility_signal(token_text)
    qgi.setdefault("entropy_scores", []).append(entropy_score)
    qgi["volatility"] = volatility_signal
    if al_uls.is_symbolic_call(token_text):
        qgi.setdefault("symbolic_calls", []).append(al_uls.parse_symbolic_call(token_text))
    for t in motif_engine.detect_tags(token_text):
        if t not in qgi.setdefault("motif_tags", []):
            qgi["motif_tags"].append(t)


async def _apply_token_to_qgi_async(qgi: Dict[str, Any], token_text: str) -> None:
    _apply_token_to_qgi(qgi, token_text)
    # Evaluate only the last detected call to keep latency low
    if qgi.get("symbolic_calls"):
        last = qgi["symbolic_calls"][ -1]
        res = await al_uls.eval_symbolic_call_async(last)
        qgi.setdefault("symbolic_results", []).append(res)


async def api_suggest_async(prefix: str = "", state: str = "S0", use_semantic: bool = True) -> Dict[str, Any]:
    qgi: Dict[str, Any] = {
        "state": state,
        "prefix": prefix,
        "selects": [],
        "filters": [],
        "group_by": [],
        "order": None,
        "tokens": [],
        "entropy_scores": [],
        "volatility": None,
        "symbolic_calls": [],
        "symbolic_results": [],
        "retrieval_routes": [],
        "motif_tags": []
    }
    qgi["tokens"].append(prefix)
    await _apply_token_to_qgi_async(qgi, prefix)
    suggestions = matrix_processor.semantic_state_suggest(prefix, state) if use_semantic and matrix_processor.available() else _prefix_match(prefix, state)
    return {"suggestions": suggestions, "qgi": qgi}

# =============================
# File: src/chaos_llm/api.py (excerpt: async toggle + batch + status)
# =============================
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List
from .services.qgi import api_suggest, api_suggest_async
from .services.retrieval import ingest_texts, search
from .services.unitary_mixer import route_mixture, choose_route
from .services.al_uls import al_uls

app = FastAPI(title="Chaos LLM MVP", version="0.4.0")

class SuggestRequest(BaseModel):
    prefix: str = ""
    state: str = "S0"
    use_semantic: bool = True
    async_eval: bool = False

class SuggestResponse(BaseModel):
    suggestions: List[str]
    qgi: Dict[str, Any]
    mixture: Dict[str, float]
    route: str

class IngestRequest(BaseModel):
    docs: List[str]
    namespace: str = "default"

class SearchRequest(BaseModel):
    query: str
    namespace: str = "default"
    top_k: int = 5

class BatchSymbolicRequest(BaseModel):
    calls: List[Dict[str, Any]]

@app.get("/symbolic/status")
async def symbolic_status() -> Dict[str, Any]:
    return await al_uls.health()

@app.post("/batch_symbolic")
async def batch_symbolic(payload: BatchSymbolicRequest) -> Dict[str, Any]:
    results = await al_uls.batch_eval_symbolic_calls(payload.calls)
    return {"results": results}

@app.post("/suggest", response_model=SuggestResponse)
async def suggest(payload: SuggestRequest) -> SuggestResponse:
    result = await api_suggest_async(prefix=payload.prefix, state=payload.state, use_semantic=payload.use_semantic) if payload.async_eval \
             else api_suggest(prefix=payload.prefix, state=payload.state, use_semantic=payload.use_semantic)
    mixture = route_mixture(result["qgi"]) ; route = choose_route(mixture)
    result["qgi"].setdefault("retrieval_routes", []).append(route)
    return SuggestResponse(suggestions=result["suggestions"], qgi=result["qgi"], mixture=mixture, route=route)

# =============================
# File: docker-compose.yml (healthchecks + env)
# =============================
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - MIXER_DEFAULT_SPLIT=0.5
      - USE_FAISS=0
      - DATABASE_URL=sqlite+aiosqlite:///./data/qgi.db
      - JULIA_SERVER_URL=http://julia:8088
      - JULIA_WS_URL=ws://julia:8089
      - ALULS_PREFER_WS=1
      - ALULS_HTTP_TTL=30
      - ALULS_WS_TTL=30
    depends_on:
      julia:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:8000/"]
      interval: 15s
      timeout: 5s
      retries: 10
    volumes:
      - ./data:/app/data
      - ./src:/app/src

  julia:
    build:
      context: .
      dockerfile: julia_server/Dockerfile
    ports: ["8088:8088", "8089:8089"]
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:8088/health"]
      interval: 10s
      timeout: 5s
      retries: 10

# =============================
# File: julia_server/Project.toml
# =============================
name = "ChaosServer"
uuid = "b3c4b0c1-2a8b-4c3a-9f44-7ad1c2ec9e1f"
version = "0.2.0"

[deps]
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
JSON3 = "0f8b85d8-1172-5c60-9a20-2f6a0a8b4d9c"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
WebSockets = "104b5d7c-3166-5388-85b0-cb73d876171c"
# Optional advanced math libs (comment in if you implement)
# DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
# FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"

# =============================
# File: julia_server/src/Server.jl
# =============================
module ChaosServer

using HTTP, JSON3, Logging, Dates, Symbolics, WebSockets

const ALLOWED_FUNCS = Set(["SUM","MEAN","VAR","DIFF","SIMPLIFY"])  # extend as needed

struct AppState
    started_at::DateTime
    http_count::Int
    ws_count::Int
end
const STATE = Ref{AppState}()

_json(x) = JSON3.write(x)

function _parse_symbolic_call(s::AbstractString)
    m = match(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)$", strip(s))
    if m === nothing
        return Dict("name"=>nothing, "args"=>String[])
    end
    name = uppercase(String(m.captures[1]))
    args_str = String(m.captures[2])
    args = isempty(strip(args_str)) ? String[] : [strip(x) for x in split(args_str, ",")]
    return Dict("name"=>name, "args"=>args)
end

function _eval_symbolic(name::String, args::Vector{String})
    if !(name in ALLOWED_FUNCS)
        return Dict("ok"=>false, "error"=>"function not allowed", "name"=>name)
    end
    try
        if name == "SUM"
            vals = parse.(Float64, args)
            return Dict("ok"=>true, "result"=>sum(vals))
        elseif name == "MEAN"
            vals = parse.(Float64, args)
            return Dict("ok"=>true, "result"=>sum(vals)/max(length(vals),1))
        elseif name == "VAR"
            vals = parse.(Float64, args)
            μ = sum(vals)/max(length(vals),1)
            v = sum((x-μ)^2 for x in vals)/max(length(vals),1)
            return Dict("ok"=>true, "result"=>v)
        elseif name == "DIFF"
            f = Symbolics.parse_expr(args[1])
            sym = Symbolics.parse_expr(args[2])
            return Dict("ok"=>true, "result"=>string(Symbolics.derivative(f, sym)))
        elseif name == "SIMPLIFY"
            expr = Symbolics.parse_expr(args[1])
            return Dict("ok"=>true, "result"=>string(Symbolics.simplify(expr)))
        end
    catch e
        return Dict("ok"=>false, "error"=>string(e), "name"=>name)
    end
end

# HTTP routes
function route(req::HTTP.Request)
    try
        if req.target == "/health"
            return HTTP.Response(200, _json(Dict(
                "ok"=>true,
                "service"=>"Chaos Julia Server",
                "started_at"=>string(STATE[].started_at),
                "http_count"=>STATE[].http_count,
                "ws_count"=>STATE[].ws_count,
            )))
        elseif req.target == "/v1/symbolic/parse" && HTTP.method(req) == "POST"
            data = JSON3.read(String(req.body))
            parsed = _parse_symbolic_call(get(data, "text", ""))
            STATE[].http_count += 1
            return HTTP.Response(200, _json(Dict("ok"=>true, "parsed"=>parsed)))
        elseif req.target == "/v1/symbolic/eval" && HTTP.method(req) == "POST"
            data = JSON3.read(String(req.body))
            name = uppercase(String(get(data, "name", "")))
            args = Vector{String}(get(data, "args", String[]))
            result = _eval_symbolic(name, args)
            STATE[].http_count += 1
            return HTTP.Response(200, _json(result))
        else
            return HTTP.Response(404, _json(Dict("ok"=>false, "error"=>"not found")))
        end
    catch e
        @warn "Route error" error=e
        return HTTP.Response(500, _json(Dict("ok"=>false, "error"=>string(e))))
    end
end

# WebSocket handler
function ws_handler(ws)
    try
        while !eof(ws)
            data = String(readavailable(ws))
            msg = JSON3.read(data)
            if get(msg, "type", "") == "parse"
                parsed = _parse_symbolic_call(get(msg, "text", ""))
                write(ws, _json(Dict("type"=>"parse_result", "parsed"=>parsed)))
            elseif get(msg, "type", "") == "eval"
                name = uppercase(String(get(msg, "name", "")))
                args = Vector{String}(get(msg, "args", String[]))
                result = _eval_symbolic(name, args)
                write(ws, _json(Dict("type"=>"eval_result", "result"=>result)))
            elseif get(msg, "type", "") == "batch_eval"
                calls = get(msg, "calls", [])
                results = [_eval_symbolic(c["name"], c["args"]) for c in calls]
                write(ws, _json(Dict("type"=>"batch_eval_result", "results"=>results)))
            else
                write(ws, _json(Dict("type"=>"error", "error"=>"unknown message type")))
            end
            STATE[].ws_count += 1
        end
    catch e
        @warn "WebSocket error" error=e
    end
end

function start(; host="0.0.0.0", http_port::Integer=8088, ws_port::Integer=8089)
    STATE[] = AppState(now(), 0, 0)
    @info "Starting Chaos Julia Server" host http_port ws_port
    @async HTTP.serve(route, host, http_port; verbose=false)
    @async WebSockets.listen(host, ws_port, ws_handler)
    @info "Servers started. Ctrl+C to stop."
    try
        while true
            sleep(1)
        end
    catch
        @info "Shutting down"
    end
end

end # module

# =============================
# File: test_enhanced_system.py
# =============================
import asyncio
import os

os.environ.setdefault("JULIA_SERVER_URL", "http://localhost:8088")
os.environ.setdefault("JULIA_WS_URL", "ws://localhost:8089")

from chaos_llm.services.al_uls_client import al_uls_client
from chaos_llm.services.al_uls_ws_client import al_uls_ws_client

async def main():
    print("HTTP health:", await al_uls_client.health())
    res1 = await al_uls_client.eval("SUM", ["1","2","3"]) ; print("HTTP SUM:", res1)
    res2 = await al_uls_ws_client.eval("MEAN", ["4","5","6"]) ; print("WS MEAN:", res2)
    batch = await al_uls_ws_client.batch_eval([
        {"name":"SUM","args":["1","2","3"]},
        {"name":"VAR","args":["10","20","30"]}
    ])
    print("WS batch:", batch)

if __name__ == "__main__":
    asyncio.run(main())

# =============================
# README — usage snippets
# =============================
# Build & Run
# docker compose up --build
# 
# Test health:
#   curl -s localhost:8088/health | jq
# Async suggest with Julia eval:
#   curl -sX POST localhost:8000/suggest -H 'content-type: application/json' -d '{"prefix":"VAR(1,2,3)","state":"S0","use_semantic":true,"async_eval":true}' | jq
# Batch symbolic:
# 🚀 Phase 3 Evolution: WebSocket Preference + Caching

Let's implement WebSocket preference with HTTP fallback and add caching for repeated symbolic calls!
✅ 1. Add Caching to AL-ULS Clients

Update src/chaos_llm/services/al_uls_client.py:
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄

# src/chaos_llm/services/al_uls_client.py
import os
import httpx
import asyncio
from typing import Dict, Any, List
from functools import lru_cache

JULIA_SERVER_URL = os.environ.get("JULIA_SERVER_URL", "http://localhost:8088")

class ALULSClient:
    def __init__(self, base_url: str | None = None):
        self.base = base_url or JULIA_SERVER_URL
        self.client = httpx.AsyncClient(timeout=10)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _make_cache_key(self, name: str, args: List[str]) -> str:
        return f"{name}:{'|'.join(args)}"

    async def parse(self, text: str) -> Dict[str, Any]:
        cache_key = f"parse:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            r = await self.client.post(f"{self.base}/v1/symbolic/parse", json={"text": text})
            r.raise_for_status()
            result = r.json()
            self._cache[cache_key] = result
            return result
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def eval(self, name: str, args: List[str]) -> Dict[str, Any]:
        cache_key = self._make_cache_key(name, args)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            r = await self.client.post(f"{self.base}/v1/symbolic/eval", json={"name": name, "args": args})
            r.raise_for_status()
            result = r.json()
            self._cache[cache_key] = result
            return result
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def batch_eval(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Check cache for individual calls
        cached_results = []
        uncached_calls = []
        uncached_indices = []
        
        for i, call in enumerate(calls):
            name = call.get("name", "")
            args = call.get("args", [])
            cache_key = self._make_cache_key(name, args)
            if cache_key in self._cache:
                cached_results.append((i, self._cache[cache_key]))
            else:
                uncached_calls.append(call)
                uncached_indices.append(i)
        
        # Evaluate uncached calls
        if uncached_calls:
            tasks = [self.eval(c.get("name", ""), c.get("args", [])) for c in uncached_calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            out: List[Dict[str, Any]] = []
            for res in results:
                if isinstance(res, Exception):
                    out.append({"ok": False, "error": str(res)})
                else:
                    out.append(res)
            
            # Update cache and merge results
            for idx, result in zip(uncached_indices, out):
                cache_key = self._make_cache_key(calls[idx].get("name", ""), calls[idx].get("args", []))
                self._cache[cache_key] = result
        else:
            out = []
        
        # Reconstruct full results list
        final_results = [None] * len(calls)
        for i, result in cached_results:
            final_results[i] = result
        for i, result in zip(uncached_indices, out):
            final_results[i] = result
            
        return [r for r in final_results if r is not None]

al_uls_client = ALULSClient()

✅ 2. Enhanced WebSocket Client with Caching

Update src/chaos_llm/services/al_uls_ws_client.py:
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄

# src/chaos_llm/services/al_uls_ws_client.py
import os
import asyncio
import websockets
import json
from typing import Dict, Any, List

JULIA_WS_URL = os.environ.get("JULIA_WS_URL", "ws://localhost:8089")

class ALULSWSClient:
    def __init__(self, ws_url: str | None = None):
        self.ws_url = ws_url or JULIA_WS_URL
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _make_cache_key(self, name: str, args: List[str]) -> str:
        return f"{name}:{'|'.join(args)}"

    async def connect(self):
        if (self.websocket is None) or self.websocket.closed:
            self.websocket = await websockets.connect(self.ws_url)
        return self.websocket

    async def _roundtrip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ws = await self.connect()
            await ws.send(json.dumps(payload))
            resp = await ws.recv()
            return json.loads(resp)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def parse(self, text: str) -> Dict[str, Any]:
        cache_key = f"parse:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = await self._roundtrip({"type": "parse", "text": text})
        self._cache[cache_key] = result
        return result

    async def eval(self, name: str, args: List[str]) -> Dict[str, Any]:
        cache_key = self._make_cache_key(name, args)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = await self._roundtrip({"type": "eval", "name": name, "args": args})
        self._cache[cache_key] = result
        return result

    async def batch_eval(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Check cache for individual calls
        cached_results = []
        uncached_calls = []
        uncached_indices = []
        
        for i, call in enumerate(calls):
            name = call.get("name", "")
            args = call.get("args", [])
            cache_key = self._make_cache_key(name, args)
            if cache_key in self._cache:
                cached_results.append((i, self._cache[cache_key]))
            else:
                uncached_calls.append(call)
                uncached_indices.append(i)
        
        # Evaluate uncached calls via WebSocket
        if uncached_calls:
            res = await self._roundtrip({"type": "batch_eval", "calls": uncached_calls})
            results = res.get("results", []) if isinstance(res, dict) else [{"ok": False, "error": "invalid response"}]
            
            # Update cache
            for idx, result in zip(uncached_indices, results):
                cache_key = self._make_cache_key(uncached_calls[idx].get("name", ""), uncached_calls[idx].get("args", []))
                self._cache[cache_key] = result
        else:
            results = []
        
        # Reconstruct full results list
        final_results = [None] * len(calls)
        for i, result in cached_results:
            final_results[i] = result
        for i, result in zip(uncached_indices, results):
            final_results[i] = result
            
        return [r for r in final_results if r is not None]

al_uls_ws_client = ALULSWSClient()

✅ 3. Updated AL-ULS with WebSocket Preference

Update src/chaos_llm/services/al_uls.py:
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄

# src/chaos_llm/services/al_uls.py
import os
import re
from typing import Dict, Any, Optional, List
from .al_uls_client import al_uls_client
from .al_uls_ws_client import al_uls_ws_client

CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)$")
USE_WEBSOCKET = bool(os.environ.get("JULIA_WS_URL"))

class ALULS:
    def is_symbolic_call(self, text: str) -> bool:
        return bool(CALL_RE.search((text or "").strip()))

    def parse_symbolic_call(self, text: str) -> Dict[str, Any]:
        m = CALL_RE.search((text or "").strip())
        if not m:
            return {"name": None, "args": []}
        name, argstr = m.group(1), m.group(2)
        args = [a.strip() for a in argstr.split(",") if a.strip()]
        return {"name": name.upper(), "args": args}

    async def eval_symbolic_call_async(self, call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if USE_WEBSOCKET:
                return await al_uls_ws_client.eval(call.get("name", ""), call.get("args", []))
            else:
                return await al_uls_client.eval(call.get("name", ""), call.get("args", []))
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def batch_eval_symbolic_calls(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if USE_WEBSOCKET:
            return await al_uls_ws_client.batch_eval(calls)
        else:
            return await al_uls_client.batch_eval(calls)

al_uls = ALULS()

✅ 4. Enhanced QGI with Better Caching Integration

Update src/chaos_llm/services/qgi.py:
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄

# src/chaos_llm/services/qgi.py
from typing import Any, Dict, List
from .entropy_engine import entropy_engine
from .matrix_processor import matrix_processor
from .al_uls import al_uls
from .motif_engine import motif_engine
from .suggestions import SUGGESTIONS

def _prefix_match(prefix: str, state: str) -> List[str]:
    pre = (prefix or "").upper()
    pool = SUGGESTIONS.get(state, [])
    return [t for t in pool if t.startswith(pre)]

def _apply_token_to_qgi(qgi: Dict[str, Any], token_text: str) -> None:
    entropy_score = entropy_engine.score_token(token_text)
    volatility_signal = entropy_engine.get_volatility_signal(token_text)
    qgi.setdefault("entropy_scores", []).append(entropy_score)
    qgi["volatility"] = volatility_signal
    if al_uls.is_symbolic_call(token_text):
        symbolic_func = al_uls.parse_symbolic_call(token_text)
        qgi.setdefault("symbolic_calls", []).append(symbolic_func)
    tags = motif_engine.detect_tags(token_text)
    if tags:
        existing = set(qgi.get("motif_tags", []))
        for t in tags:
            if t not in existing:
                qgi.setdefault("motif_tags", []).append(t)
                existing.add(t)

async def _apply_token_to_qgi_async(qgi: Dict[str, Any], token_text: str) -> None:
    _apply_token_to_qgi(qgi, token_text)
    if qgi.get("symbolic_calls"):
        last = qgi["symbolic_calls"][-1]
        res = await al_uls.eval_symbolic_call_async(last)
        qgi.setdefault("symbolic_results", []).append(res)

def api_suggest(prefix: str = "", state: str = "S0", use_semantic: bool = True) -> Dict[str, Any]:
    qgi: Dict[str, Any] = {
        "state": state,
        "prefix": prefix,
        "selects": [],
        "filters": [],
        "group_by": [],
        "order": None,
        "tokens": [],
        "entropy_scores": [],
        "volatility": None,
        "symbolic_calls": [],
        "symbolic_results": [],
        "retrieval_routes": [],
        "motif_tags": []
    }
    qgi["tokens"].append(prefix)
    _apply_token_to_qgi(qgi, prefix)
    if use_semantic and matrix_processor.available():
        suggestions = matrix_processor.semantic_state_suggest(prefix, state)
    else:
        suggestions = _prefix_match(prefix, state)
    return {"suggestions": suggestions, "qgi": qgi}

async def api_suggest_async(prefix: str = "", state: str = "S0", use_semantic: bool = True) -> Dict[str, Any]:
    qgi: Dict[str, Any] = {
        "state": state,
        "prefix": prefix,
        "selects": [],
        "filters": [],
        "group_by": [],
        "order": None,
        "tokens": [],
        "entropy_scores": [],
        "volatility": None,
        "symbolic_calls": [],
        "symbolic_results": [],
        "retrieval_routes": [],
        "motif_tags": []
    }
    qgi["tokens"].append(prefix)
    await _apply_token_to_qgi_async(qgi, prefix)
    if use_semantic and matrix_processor.available():
        suggestions = matrix_processor.semantic_state_suggest(prefix, state)
    else:
        suggestions = _prefix_match(prefix, state)
    return {"suggestions": suggestions, "qgi": qgi}

✅ 5. Update Docker Compose with Cache Volume
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28

version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MIXER_DEFAULT_SPLIT=0.5
      - USE_FAISS=0
      - DATABASE_URL=sqlite+aiosqlite:///./data/qgi.db
      - JULIA_SERVER_URL=http://julia:8088
      - JULIA_WS_URL=ws://julia:8089
    volumes:
      - ./data:/app/data
      - ./src:/app/src
      - ./cache:/app/cache  # ← Add cache volume
    depends_on:
      - julia

  julia:
    build:
      context: .
      dockerfile: julia_server/Dockerfile
    ports:
      - "8088:8088"
      - "8089:8089"

✅ 6. Enhanced Julia Server with Request Logging

Update julia_server/src/Server.jl:
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51

# julia_server/src/Server.jl
module ChaosServer

using HTTP, JSON3, Logging, Dates, Symbolics, WebSockets

const ALLOWED_FUNCS = Set(["SUM","MEAN","VAR","DIFF","SIMPLIFY"])
const REQUEST_COUNT = Ref(0)

struct AppState
    started_at::DateTime
end
const STATE = Ref{AppState}()

# utils
_json(x) = JSON3.write(x)

function _parse_symbolic_call(s::AbstractString)
    m = match(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)$", strip(s))
    if m === nothing
        return Dict("name"=>nothing, "args"=>String[])
    end
    name = uppercase(String(m.captures[1]))
    args_str = String(m.captures[2])
    args = isempty(strip(args_str)) ? String[] : [strip(x) for x in split(args_str, ",")]
    return Dict("name"=>name, "args"=>args)
end

function _eval_symbolic(name::String, args::Vector{String})
    if !(name in ALLOWED_FUNCS)
        return Dict("ok"=>false, "error"=>"function not allowed", "name"=>name)
    end
    try
        if name == "SUM"
            vals = parse.(Float64, args)
            return Dict("ok"=>true, "result"=>sum(vals))
        elseif name == "MEAN"
            vals = parse.(Float64, args)
            return Dict("ok"=>true, "result"=>sum(vals)/max(length(vals),1))
        elseif name == "VAR"
            vals = parse.(Float64, args)
            μ = sum(vals)/max(length(vals),1)
            v = sum((x-μ)^2 for x in vals)/max(length(vals),1)
            return Dict("ok"=>true, "result"=>v)
        elseif name == "DIFF"
            f = Symbolics.parse_expr(args[1])
            sym = Symbolics.parse_expr(args[2])
            return Dict("ok"=>true, "result"=>string(Symbolics.derivative(f, sym)))
        elseif name == "SIMPLIFY"
            expr = Symbolics.parse_expr(args[1])
            return Dict("ok"=>true, "result"=>string(Symbolics.simplify(expr)))
✅ 7. Example Usage with Caching
WebSocket Preferred (if JULIA_WS_URL is set):
1
2
3
4
5

curl -sX POST localhost:8000/suggest \
  -H 'content-type: application/json' \
  -d '{"prefix":"VAR(1,2,3)","state":"S0","use_semantic":true,"async_eval":true}' | jq

Batch with Caching:
1
2
3
4
5

curl -sX POST localhost:8000/batch_symbolic \
  -H 'content-type: application/json' \
  -d '{"calls":[{"name":"SUM","args":["1","2","3"]},{"name":"MEAN","args":["4","5","6"]}]}' | jq

Health Check (shows request count):
1
2
3

curl -s localhost:8088/health | jq

  curl -sX POST localhost:8000/batch_symbolic -H 'content-type: application/json' -d '{"calls":[{"name":"SUM","args":["1","2","3"]},{"name":"MEAN","args":["4","5","6"]}]}' | jq

Here's a comprehensive implementation of the enhanced Julia integration with WebSocket preference and caching:
Enhanced Julia Integration - Complete Implementation
File: src/chaos_llm/services/al_uls_client.py
python

import os
import time
import asyncio
from typing import Dict, Any, List, Tuple
import httpx

JULIA_SERVER_URL = os.environ.get("JULIA_SERVER_URL", "http://localhost:8088")
CACHE_TTL_SECONDS = float(os.environ.get("ALULS_HTTP_TTL", 30))

class TTLCache:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self._store: Dict[Tuple[str, Tuple[str, ...]], Tuple[float, Dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0

    def _now(self) -> float:
        return time.monotonic()

    def _key(self, name: str, args: List[str]) -> Tuple[str, Tuple[str, ...]]:
        return (name.upper(), tuple(args))

    def get(self, name: str, args: List[str]) -> Dict[str, Any] | None:
        k = self._key(name, args)
        v = self._store.get(k)
        if not v:
            self.misses += 1
            return None
        ts, data = v
        if self._now() - ts <= self.ttl:
            self.hits += 1
            return data
        self._store.pop(k, None)
        self.misses += 1
        return None

    def set(self, name: str, args: List[str], value: Dict[str, Any]) -> None:
        self._store[self._key(name, args)] = (self._now(), value)

    def stats(self) -> Dict[str, Any]:
        return {"entries": len(self._store), "hits": self.hits, "misses": self.misses, "ttl": self.ttl}

class ALULSClient:
    def __init__(self, base_url: str | None = None):
        self.base = base_url or JULIA_SERVER_URL
        self.client = httpx.AsyncClient(timeout=10)
        self.cache = TTLCache(CACHE_TTL_SECONDS)

    async def health(self) -> Dict[str, Any]:
        try:
            r = await self.client.get(f"{self.base}/health")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def parse(self, text: str) -> Dict[str, Any]:
        try:
            r = await self.client.post(f"{self.base}/v1/symbolic/parse", json={"text": text})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def eval(self, name: str, args: List[str]) -> Dict[str, Any]:
        cached = self.cache.get(name, args)
        if cached is not None:
            return {**cached, "_cached": True}
        try:
            r = await self.client.post(f"{self.base}/v1/symbolic/eval", json={"name": name, "args": args})
            r.raise_for_status()
            data = r.json()
            if data.get("ok"):
                self.cache.set(name, args, data)
            return data
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def batch_eval(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        to_run: List[Tuple[int, Dict[str, Any]]] = []
        results: List[Dict[str, Any]] = [{} for _ in calls]
        
        for i, c in enumerate(calls):
            name = c.get("name", "").upper()
            args = c.get("args", [])
            cached = self.cache.get(name, args)
            if cached is not None:
                results[i] = {**cached, "_cached": True}
            else:
                to_run.append((i, {"name": name, "args": args}))
        
        tasks = [self.eval(c["name"], c["args"]) for _, c in to_run]
        outs = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (i, _), out in zip(to_run, outs):
            results[i] = out if not isinstance(out, Exception) else {"ok": False, "error": str(out)}
        
        return results

al_uls_client = ALULSClient()

File: src/chaos_llm/services/al_uls_ws_client.py
python

import os
import json
import asyncio
from typing import Dict, Any, List, Tuple
import websockets

JULIA_WS_URL = os.environ.get("JULIA_WS_URL", "ws://localhost:8089")
CACHE_TTL_WS = float(os.environ.get("ALULS_WS_TTL", 30))

class TTLCacheWS:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self._store: Dict[Tuple[str, Tuple[str, ...]], Tuple[float, Dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0

    def _now(self) -> float:
        return asyncio.get_event_loop().time()

    def _key(self, name: str, args: List[str]) -> Tuple[str, Tuple[str, ...]]:
        return (name.upper(), tuple(args))

    def get(self, name: str, args: List[str]) -> Dict[str, Any] | None:
        k = self._key(name, args)
        v = self._store.get(k)
        if not v:
            self.misses += 1
            return None
        ts, data = v
        if self._now() - ts <= self.ttl:
            self.hits += 1
            return data
        self._store.pop(k, None)
        self.misses += 1
        return None

    def set(self, name: str, args: List[str], value: Dict[str, Any]) -> None:
        self._store[self._key(name, args)] = (self._now(), value)

    def stats(self) -> Dict[str, Any]:
        return {"entries": len(self._store), "hits": self.hits, "misses": self.misses, "ttl": self.ttl}

class ALULSWSClient:
    def __init__(self, ws_url: str | None = None):
        self.ws_url = ws_url or JULIA_WS_URL
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.cache = TTLCacheWS(CACHE_TTL_WS)

    async def connect(self):
        if (self.websocket is None) or self.websocket.closed:
            self.websocket = await websockets.connect(self.ws_url)
        return self.websocket

    async def _roundtrip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ws = await self.connect()
            await ws.send(json.dumps(payload))
            resp = await ws.recv()
            return json.loads(resp)
        except Exception as e:
            try:
                if self.websocket:
                    await self.websocket.close()
            finally:
                self.websocket = None
            return {"ok": False, "error": str(e)}

    async def parse(self, text: str) -> Dict[str, Any]:
        return await self._roundtrip({"type": "parse", "text": text})

    async def eval(self, name: str, args: List[str]) -> Dict[str, Any]:
        cached = self.cache.get(name, args)
        if cached is not None:
            return {**cached, "_cached": True}
        res = await self._roundtrip({"type": "eval", "name": name, "args": args})
        if isinstance(res, dict) and res.get("ok"):
            self.cache.set(name, args, res)
        return res

    async def batch_eval(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        res = await self._roundtrip({"type": "batch_eval", "calls": calls})
        if isinstance(res, dict) and "results" in res and isinstance(res["results"], list):
            out: List[Dict[str, Any]] = []
            for c, r in zip(calls, res["results"]):
                if isinstance(r, dict) and r.get("ok"):
                    self.cache.set(c.get("name", ""), c.get("args", []), r)
                out.append(r if isinstance(r, dict) else {"ok": False, "error": "invalid item"})
            return out
        return [await self.eval(c.get("name", ""), c.get("args", [])) for c in calls]

al_uls_ws_client = ALULSWSClient()

File: src/chaos_llm/services/al_uls.py
python

import os
import re
from typing import Dict, Any, List
from .al_uls_client import al_uls_client
from .al_uls_ws_client import al_uls_ws_client

CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)$")
PREFER_WS = os.environ.get("ALULS_PREFER_WS", "1") in {"1", "true", "TRUE", "yes"}

class ALULS:
    def is_symbolic_call(self, text: str) -> bool:
        return bool(CALL_RE.search((text or "").strip()))

    def parse_symbolic_call(self, text: str) -> Dict[str, Any]:
        m = CALL_RE.search((text or "").strip())
        if not m:
            return {"name": None, "args": []}
        name, argstr = m.group(1), m.group(2)
        args = [a.strip() for a in argstr.split(",") if a.strip()]
        return {"name": name.upper(), "args": args}

    async def health(self) -> Dict[str, Any]:
        return await al_uls_client.health()

    async def eval_symbolic_call_async(self, call: Dict[str, Any]) -> Dict[str, Any]:
        name = call.get("name", "")
        args = call.get("args", [])
        if PREFER_WS:
            res = await al_uls_ws_client.eval(name, args)
            if isinstance(res, dict) and (res.get("ok") or res.get("_cached")):
                return res
        return await al_uls_client.eval(name, args)

    async def batch_eval_symbolic_calls(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if PREFER_WS:
            res = await al_uls_ws_client.batch_eval(calls)
            if isinstance(res, list) and any(isinstance(r, dict) for r in res):
                return res
        return await al_uls_client.batch_eval(calls)

al_uls = ALULS()

File: src/chaos_llm/services/qgi.py (excerpt showing async apply)
python

from typing import Any, Dict, List
from .entropy_engine import entropy_engine
from .matrix_processor import matrix_processor
from .al_uls import al_uls
from .motif_engine import motif_engine
from .suggestions import SUGGESTIONS

def _prefix_match(prefix: str, state: str) -> List[str]:
    pre = (prefix or "").upper()
    pool = SUGGESTIONS.get(state, [])
    return [t for t in pool if t.startswith(pre)]

def _apply_token_to_qgi(qgi: Dict[str, Any], token_text: str) -> None:
    entropy_score = entropy_engine.score_token(token_text)
    volatility_signal = entropy_engine.get_volatility_signal(token_text)
    qgi.setdefault("entropy_scores", []).append(entropy_score)
    qgi["volatility"] = volatility_signal
    if al_uls.is_symbolic_call(token_text):
        qgi.setdefault("symbolic_calls", []).append(al_uls.parse_symbolic_call(token_text))
    for t in motif_engine.detect_tags(token_text):
        if t not in qgi.setdefault("motif_tags", []):
            qgi["motif_tags"].append(t)

async def _apply_token_to_qgi_async(qgi: Dict[str, Any], token_text: str) -> None:
    _apply_token_to_qgi(qgi, token_text)
    if qgi.get("symbolic_calls"):
        last = qgi["symbolic_calls"][-1]
        res = await al_uls.eval_symbolic_call_async(last)
        qgi.setdefault("symbolic_results", []).append(res)

async def api_suggest_async(prefix: str = "", state: str = "S0", use_semantic: bool = True) -> Dict[str, Any]:
    qgi: Dict[str, Any] = {
        "state": state,
        "prefix": prefix,
        "selects": [],
        "filters": [],
        "group_by": [],
        "order": None,
        "tokens": [],
        "entropy_scores": [],
        "volatility": None,
        "symbolic_calls": [],
        "symbolic_results": [],
        "retrieval_routes": [],
        "motif_tags": []
    }
    qgi["tokens"].append(prefix)
    await _apply_token_to_qgi_async(qgi, prefix)
    suggestions = matrix_processor.semantic_state_suggest(prefix, state) if use_semantic and matrix_processor.available() else _prefix_match(prefix, state)
    return {"suggestions": suggestions, "qgi": qgi}

File: src/chaos_llm/api.py (excerpt: async toggle + batch + status)
python

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List
from .services.qgi import api_suggest, api_suggest_async
from .services.retrieval import ingest_texts, search
from .services.unitary_mixer import route_mixture, choose_route
from .services.al_uls import al_uls

app = FastAPI(title="Chaos LLM MVP", version="0.4.0")

class SuggestRequest(BaseModel):
    prefix: str = ""
    state: str = "S0"
    use_semantic: bool = True
    async_eval: bool = False

class SuggestResponse(BaseModel):
    suggestions: List[str]
    qgi: Dict[str, Any]
    mixture: Dict[str, float]
    route: str

class IngestRequest(BaseModel):
    docs: List[str]
    namespace: str = "default"

class SearchRequest(BaseModel):
    query: str
    namespace: str = "default"
    top_k: int = 5

class BatchSymbolicRequest(BaseModel):
    calls: List[Dict[str, Any]]

@app.get("/symbolic/status")
async def symbolic_status() -> Dict[str, Any]:
    return await al_uls.health()

@app.post("/batch_symbolic")
async def batch_symbolic(payload: BatchSymbolicRequest) -> Dict[str, Any]:
    results = await al_uls.batch_eval_symbolic_calls(payload.calls)
    return {"results": results}

@app.post("/suggest", response_model=SuggestResponse)
async def suggest(payload: SuggestRequest) -> SuggestResponse:
    result = await api_suggest_async(prefix=payload.prefix, state=payload.state, use_semantic=payload.use_semantic) if payload.async_eval \
             else api_suggest(prefix=payload.prefix, state=payload.state, use_semantic=payload.use_semantic)
    mixture = route_mixture(result["qgi"])
    route = choose_route(mixture)
    result["qgi"].setdefault("retrieval_routes", []).append(route)
    return SuggestResponse(suggestions=result["suggestions"], qgi=result["qgi"], mixture=mixture, route=route)

File: docker-compose.yml
yaml

version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - MIXER_DEFAULT_SPLIT=0.5
      - USE_FAISS=0
      - DATABASE_URL=sqlite+aiosqlite:///./data/qgi.db
      - JULIA_SERVER_URL=http://julia:8088
      - JULIA_WS_URL=ws://julia:8089
      - ALULS_PREFER_WS=1
      - ALULS_HTTP_TTL=30
      - ALULS_WS_TTL=30
    depends_on:
      julia:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:8000/"]
      interval: 15s
      timeout: 5s
      retries: 10
    volumes:
      - ./data:/app/data
      - ./src:/app/src

  julia:
    build:
      context: .
      dockerfile: julia_server/Dockerfile
    ports: ["8088:8088", "8089:8089"]
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:8088/health"]
      interval: 10s
      timeout: 5s
      retries: 10

File: julia_server/Project.toml
toml

name = "ChaosServer"
uuid = "b3c4b0c1-2a8b-4c3a-9f44-7ad1c2ec9e1f"
version = "0.2.0"

[deps]
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
JSON3 = "0f8b85d8-1172-5c60-9a20-2f6a0a8b4d9c"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
WebSockets = "104b5d7c-3166-5388-85b0-cb73d876171c"

File: julia_server/src/Server.jl
julia

module ChaosServer

using HTTP, JSON3, Logging, Dates, Symbolics, WebSockets

const ALLOWED_FUNCS = Set(["SUM","MEAN","VAR","DIFF","SIMPLIFY"])

struct AppState
    started_at::DateTime
    http_count::Int
    ws_count::Int
end
const STATE = Ref{AppState}()

_json(x) = JSON3.write(x)

function _parse_symbolic_call(s::AbstractString)
    m = match(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)$", strip(s))
    if m === nothing
        return Dict("name"=>nothing, "args"=>String[])
    end
    name = uppercase(String(m.captures[1]))
    args_str = String(m.captures[2])
    args = isempty(strip(args_str)) ? String[] : [strip(x) for x in split(args_str, ",")]
    return Dict("name"=>name, "args"=>args)
end

function _eval_symbolic(name::String, args::Vector{String})
    if !(name in ALLOWED_FUNCS)
        return Dict("ok"=>false, "error"=>"function not allowed", "name"=>name)
    end
    try
        if name == "SUM"
            vals = parse.(Float64, args)
            return Dict("ok"=>true, "result"=>sum(vals))
        elseif name == "MEAN"
            vals = parse.(Float64, args)
            return Dict("ok"=>true, "result"=>sum(vals)/max(length(vals),1))
        elseif name == "VAR"
            vals = parse.(Float64, args)
            μ = sum(vals)/max(length(vals),1)
            v = sum((x-μ)^2 for x in vals)/max(length(vals),1)
            return Dict("ok"=>true, "result"=>v)
        elseif name == "DIFF"
            f = Symbolics.parse_expr(args[1])
            sym = Symbolics.parse_expr(args[2])
            return Dict("ok"=>true, "result"=>string(Symbolics.derivative(f, sym)))
        elseif name == "SIMPLIFY"
            expr = Symbolics.parse_expr(args[1])
            return Dict("ok"=>true, "result"=>string(Symbolics.simplify(expr)))
        end
    catch e
        return Dict("ok"=>false, "error"=>string(e), "name"=>name)
    end
end

function route(req::HTTP.Request)
    try
        if req.target == "/health"
            return HTTP.Response(200, _json(Dict(
                "ok"=>true,
                "service"=>"Chaos Julia Server",
                "started_at"=>string(STATE[].started_at),
                "http_count"=>STATE[].http_count,
                "ws_count"=>STATE[].ws_count,
            )))
        elseif req.target == "/v1/symbolic/parse" && HTTP.method(req) == "POST"
            data = JSON3.read(String(req.body))
            parsed = _parse_symbolic_call(get(data, "text", ""))
            STATE[].http_count += 1
            return HTTP.Response(200, _json(Dict("ok"=>true, "parsed"=>parsed)))
        elseif req.target == "/v1/symbolic/eval" && HTTP.method(req) == "POST"
            data = JSON3.read(String(req.body))
            name = uppercase(String(get(data, "name", "")))
            args = Vector{String}(get(data, "args", String[]))
            result = _eval_symbolic(name, args)
            STATE[].http_count += 1
            return HTTP.Response(200, _json(result))
        else
            return HTTP.Response(404, _json(Dict("ok"=>false, "error"=>"not found")))
        end
    catch e
        @warn "Route error" error=e
        return HTTP.Response(500, _json(Dict("ok"=>false, "error"=>string(e))))
    end
end

function ws_handler(ws)
    try
        while !eof(ws)
            data = String(readavailable(ws))
            msg = JSON3.read(data)
            if get(msg, "type", "") == "parse"
                parsed = _parse_symbolic_call(get(msg, "text", ""))
                write(ws, _json(Dict("type"=>"parse_result", "parsed"=>parsed)))
            elseif get(msg, "type", "") == "eval"
                name = uppercase(String(get(msg, "name", "")))
                args = Vector{String}(get(msg, "args", String[]))
                result = _eval_symbolic(name, args)
                write(ws, _json(Dict("type"=>"eval_result", "result"=>result)))
            elseif get(msg, "type", "") == "batch_eval"
                calls = get(msg, "calls", [])
                results = [_eval_symbolic(c["name"], c["args"]) for c in calls]
                write(ws, _json(Dict("type"=>"batch_eval_result", "results"=>results)))
            else
                write(ws, _json(Dict("type"=>"error", "error"=>"unknown message type")))
            end
            STATE[].ws_count += 1
        end
    catch e
        @warn "WebSocket error" error=e
    end
end

function start(; host="0.0.0.0", http_port::Integer=8088, ws_port::Integer=8089)
    STATE[] = AppState(now(), 0, 0)
    @info "Starting Chaos Julia Server" host http_port ws_port
    @async HTTP.serve(route, host, http_port; verbose=false)
    @async WebSockets.listen(host, ws_port, ws_handler)
    @info "Servers started. Ctrl+C to stop."
    try
        while true
            sleep(1)
        end
    catch
        @info "Shutting down"
    end
end

end # module

File: test_enhanced_system.py
python

import asyncio
import os

os.environ.setdefault("JULIA_SERVER_URL", "http://localhost:8088")
os.environ.setdefault("JULIA_WS_URL", "ws://localhost:8089")

from chaos_llm.services.al_uls_client import al_uls_client
from chaos_llm.services.al_uls_ws_client import al_uls_ws_client

async def main():
    print("HTTP health:", await al_uls_client.health())
    res1 = await al_uls_client.eval("SUM", ["1","2","3"])
    print("HTTP SUM:", res1)
    res2 = await al_uls_ws_client.eval("MEAN", ["4","5","6"])
    print("WS MEAN:", res2)
    batch = await al_uls_ws_client.batch_eval([
        {"name":"SUM","args":["1","2","3"]},
        {"name":"VAR","args":["10","20","30"]}
    ])
    print("WS batch:", batch)

if __name__ == "__main__":
    asyncio.run(main())

Usage Examples
Build & Run
bash

docker compose up --build

Test health:
bash

curl -s localhost:8088/health | jq

Async suggest with Julia eval:
bash

curl -sX POST localhost:8000/suggest \
  -H 'content-type: application/json' \
  -d '{"prefix":"VAR(1,2,3)","state":"S0","use_semantic":true,"async_eval":true}' | jq

Batch symbolic:
bash

curl -sX POST localhost:8000/batch_symbolic \
  -H 'content-type: application/json' \
  -d '{"calls":[{"name":"SUM","args":["1","2","3"]},{"name":"MEAN","args":["4","5","6"]}]}' | jq

This implementation provides:

    WebSocket preference with HTTP fallback

    TTL-based caching for both HTTP and WebSocket clients

    Batch processing with intelligent caching

    Health monitoring and request statistics

    Robust error handling and reconnection logic

    Comprehensive testing framework

The system now efficiently handles symbolic computations with minimal latency and optimal resource utilization.
#!/usr/bin/env python3
# dual_llm_wavecaster_enhanced.py
# SPDX-License-Identifier: MIT
"""
Enhanced Dual LLM WaveCaster
---------------------------
Two-LLM orchestration (local final inference + remote resource-only summaries) → framed bits
→ modulated waveform (BFSK/BPSK/QPSK/16QAM/AFSK/OFDM) → WAV/IQ files (+ optional audio out)
with visualization, simple FEC, encryption, watermarking, and metadata.

Deps (minimum):
  pip install numpy scipy requests

Optional:
  pip install matplotlib sounddevice pycryptodome

Quick start:
  python dual_llm_wavecaster_enhanced.py modulate --text "hello airwaves" --scheme qpsk --wav --iq
  python dual_llm_wavecaster_enhanced.py cast --prompt "2-paragraph plan" \
      --resource-file notes.txt --local-url http://127.0.0.1:8080 --local-mode llama-cpp \
      --remote-url https://api.openai.com --remote-key $OPENAI_API_KEY --scheme bfsk --wav
"""

from __future__ import annotations
import argparse, base64, binascii, hashlib, json, logging, math, os, struct, sys, time, warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
from enum import Enum, auto
from datetime import datetime

# ---------- Hard requirements ----------
try:
    import numpy as np
    from scipy import signal as sp_signal
    from scipy.fft import rfft, rfftfreq
except Exception as e:
    raise SystemExit("numpy and scipy are required: pip install numpy scipy") from e

# ---------- Optional dependencies ----------
try:
    import requests
except Exception:
    requests = None  # HTTP backends disabled if missing

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Protocol.KDF import PBKDF2
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("wavecaster")

# =========================================================
# Enums / Config
# =========================================================

class ModulationScheme(Enum):
    BFSK = auto()
    BPSK = auto()
    QPSK = auto()
    QAM16 = auto()
    AFSK = auto()
    OFDM = auto()
    DSSS_BPSK = auto()

class FEC(Enum):
    NONE = auto()
    HAMMING74 = auto()
    REED_SOLOMON = auto()   # stub
    LDPC = auto()           # stub
    TURBO = auto()          # stub

@dataclass
class HTTPConfig:
    base_url: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 60
    mode: str = "openai-chat"  # ["openai-chat","openai-completions","llama-cpp","textgen-webui"]
    verify_ssl: bool = True
    max_retries: int = 2
    retry_delay: float = 0.8

@dataclass
class OrchestratorSettings:
    temperature: float = 0.7
    max_tokens: int = 512
    style: str = "concise"
    max_context_chars: int = 8000

@dataclass
class ModConfig:
    sample_rate: int = 48000
    symbol_rate: int = 1200
    amplitude: float = 0.7
    f0: float = 1200.0     # BFSK 0
    f1: float = 2200.0     # BFSK 1
    fc: float = 1800.0     # PSK/QAM audio carrier (for WAV)
    clip: bool = True
    # OFDM (toy)
    ofdm_subc: int = 64
    cp_len: int = 16
    # DSSS
    dsss_chip_rate: int = 4800

@dataclass
class FrameConfig:
    use_crc32: bool = True
    use_crc16: bool = False
    preamble: bytes = b"\x55" * 8  # 01010101 * 8
    version: int = 1

# =========================================================
# Utilities
# =========================================================

def now_ms() -> int:
    return int(time.time() * 1000)

def crc32_bytes(data: bytes) -> bytes:
    return binascii.crc32(data).to_bytes(4, "big")

def crc16_ccitt(data: bytes) -> bytes:
    poly, crc = 0x1021, 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc.to_bytes(2, "big")

def to_bits(data: bytes) -> List[int]:
    return [(byte >> i) & 1 for byte in data for i in range(7, -1, -1)]

def from_bits(bits: Sequence[int]) -> bytes:
    if len(bits) % 8 != 0:
        bits = list(bits) + [0] * (8 - len(bits) % 8)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | (1 if b else 0)
        out.append(byte)
    return bytes(out)

def chunk_bits(bits: Sequence[int], n: int) -> List[List[int]]:
    return [list(bits[i:i+n]) for i in range(0, len(bits), n)]

def safe_json(obj: Any) -> str:
    def enc(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, complex):
            return {"real": float(x.real), "imag": float(x.imag)}
        if isinstance(x, datetime):
            return x.isoformat()
        return str(x)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=enc)

def write_wav_mono(path: Path, signal: np.ndarray, sample_rate: int):
    import wave
    sig = np.clip(signal, -1.0, 1.0)
    pcm = (sig * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())

def write_iq_f32(path: Path, iq: np.ndarray):
    if iq.ndim != 1 or not np.iscomplexobj(iq):
        raise ValueError("iq must be 1-D complex array")
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    path.write_bytes(interleaved.tobytes())

def plot_wave_and_spectrum(path_png: Path, x: np.ndarray, sr: int, title: str):
    if not HAS_MPL: 
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,5))
    t = np.arange(len(x))/sr
    ax1.plot(t[:min(len(t), 0.05*sr)], x[:min(len(x), int(0.05*sr))])
    ax1.set_title(f"{title} (first 50ms)")
    ax1.set_xlabel("s"); ax1.set_ylabel("amplitude")
    spec = np.abs(rfft(x)) + 1e-12
    freqs = rfftfreq(len(x), 1.0/sr)
    ax2.semilogy(freqs, spec/spec.max())
    ax2.set_xlim(0, min(8000, sr//2)); ax2.set_xlabel("Hz"); ax2.set_ylabel("norm |X(f)|")
    plt.tight_layout(); fig.savefig(path_png); plt.close(fig)

def play_audio(x: np.ndarray, sr: int):
    if not HAS_AUDIO:
        log.warning("sounddevice not installed; cannot play audio")
        return
    sd.play(x, sr); sd.wait()

# =========================================================
# FEC (simple Hamming 7,4; heavy codes are stubs)
# =========================================================

def hamming74_encode(data_bits: List[int]) -> List[int]:
    if len(data_bits) % 4 != 0:
        data_bits = data_bits + [0] * (4 - len(data_bits) % 4)
    out = []
    for i in range(0, len(data_bits), 4):
        d0, d1, d2, d3 = data_bits[i:i+4]
        p1 = d0 ^ d1 ^ d3
        p2 = d0 ^ d2 ^ d3
        p3 = d1 ^ d2 ^ d3
        out += [p1, p2, d0, p3, d1, d2, d3]
    return out

def fec_encode(bits: List[int], scheme: FEC) -> List[int]:
    if scheme == FEC.NONE:
        return list(bits)
    if scheme == FEC.HAMMING74:
        return hamming74_encode(bits)
    if scheme in (FEC.REED_SOLOMON, FEC.LDPC, FEC.TURBO):
        raise NotImplementedError(f"{scheme.name} encoding not implemented in this minimal build")
    raise ValueError("Unknown FEC")

# =========================================================
# Framing / Security / Watermark
# =========================================================

@dataclass
class SecurityConfig:
    password: Optional[str] = None           # AES-GCM if provided
    watermark: Optional[str] = None          # prepended SHA256[0:8]
    hmac_key: Optional[str] = None           # HMAC-SHA256 appended

def aes_gcm_encrypt(plaintext: bytes, password: str) -> bytes:
    if not HAS_CRYPTO:
        raise RuntimeError("pycryptodome required for encryption")
    salt = get_random_bytes(16)
    key = PBKDF2(password, salt, dkLen=32, count=200_000)
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(plaintext)
    return b"AGCM" + salt + nonce + tag + ct

def apply_hmac(data: bytes, hkey: str) -> bytes:
    import hmac
    key = hashlib.sha256(hkey.encode("utf-8")).digest()
    mac = hmac.new(key, data, hashlib.sha256).digest()
    return data + b"HMAC" + mac

def add_watermark(data: bytes, wm: str) -> bytes:
    return hashlib.sha256(wm.encode("utf-8")).digest()[:8] + data

def frame_payload(payload: bytes, fcfg: FrameConfig) -> bytes:
    header = struct.pack(">BBI", 0xA5, fcfg.version, now_ms() & 0xFFFFFFFF)
    core = header + payload
    tail = b""
    if fcfg.use_crc32:
        tail += crc32_bytes(core)
    if fcfg.use_crc16:
        tail += crc16_ccitt(core)
    return fcfg.preamble + core + tail

def encode_text(
    text: str,
    fcfg: FrameConfig,
    sec: SecurityConfig,
    fec_scheme: FEC,
) -> List[int]:
    data = text.encode("utf-8")
    if sec.watermark:
        data = add_watermark(data, sec.watermark)
    if sec.password:
        data = aes_gcm_encrypt(data, sec.password)
    framed = frame_payload(data, fcfg)
    if sec.hmac_key:
        framed = apply_hmac(framed, sec.hmac_key)
    bits = to_bits(framed)
    bits = fec_encode(bits, fec_scheme)
    return bits

# =========================================================
# Modulators (audio & IQ)
# =========================================================

class Modulators:
    @staticmethod
    def bfsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        sr, rb = cfg.sample_rate, cfg.symbol_rate
        spb = int(sr / rb)
        t = np.arange(spb) / sr
        s = []
        a = cfg.amplitude
        for b in bits:
            f = cfg.f1 if b else cfg.f0
            s.append(a * np.sin(2*np.pi*f*t))
        y = np.concatenate(s) if s else np.zeros(0, dtype=np.float64)
        return np.clip(y, -1, 1).astype(np.float32) if cfg.clip else y.astype(np.float32)

    @staticmethod
    def bpsK(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        t = np.arange(spb) / sr
        a = cfg.amplitude
        audio_blocks, iq_blocks = [], []
        for b in bits:
            phase = 0.0 if b else np.pi
            audio_blocks.append(a * np.sin(2*np.pi*fc*t + phase))
            iq_blocks.append(a * (np.cos(phase) + 1j*np.sin(phase)) * np.ones_like(t, dtype=np.complex64))
        audio = np.concatenate(audio_blocks) if audio_blocks else np.zeros(0, dtype=np.float64)
        iq = np.concatenate(iq_blocks) if iq_blocks else np.zeros(0, dtype=np.complex64)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32), iq

    @staticmethod
    def qpsK(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        # Gray map: 00->(1+1j), 01->(-1+1j), 11->(-1-1j), 10->(1-1j)
        pairs = chunk_bits(bits, 2)
        syms = []
        for p in pairs:
            b0, b1 = (p + [0,0])[:2]
            if (b0, b1) == (0,0): s = 1 + 1j
            elif (b0, b1) == (0,1): s = -1 + 1j
            elif (b0, b1) == (1,1): s = -1 - 1j
            else: s = 1 - 1j
            syms.append(s / math.sqrt(2))  # unit energy
        return Modulators._psk_qam_to_audio_iq(np.array(syms, dtype=np.complex64), cfg)

    @staticmethod
    def qam16(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        quads = chunk_bits(bits, 4)
        def map2(b0,b1):
            # Gray 2-bit to {-3,-1,1,3}
            val = (b0<<1) | b1
            return [-3,-1,1,3][val]
        syms = []
        for q in quads:
            b0,b1,b2,b3 = (q+[0,0,0,0])[:4]
            I = map2(b0,b1); Q = map2(b2,b3)
            syms.append((I + 1j*Q)/math.sqrt(10)) # unit average power
        return Modulators._psk_qam_to_audio_iq(np.array(syms, dtype=np.complex64), cfg)

    @staticmethod
    def _psk_qam_to_audio_iq(syms: np.ndarray, cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        a = cfg.amplitude
        # Upsample each symbol to 'spb' samples (rectangular pulse)
        i = np.repeat(syms.real.astype(np.float32), spb)
        q = np.repeat(syms.imag.astype(np.float32), spb)
        t = np.arange(len(i)) / sr
        audio = a * (i*np.cos(2*np.pi*fc*t) - q*np.sin(2*np.pi*fc*t))
        iq = (a * i) + 1j*(a * q)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32), iq.astype(np.complex64)

    @staticmethod
    def afsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        return Modulators.bfsK(bits, cfg)

    @staticmethod
    def dsss_bpsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        # Very simple DSSS: chip with PN sequence at cfg.dsss_chip_rate
        pn = np.array([1, -1, 1, 1, -1, 1, -1, -1], dtype=np.float32)  # toy PN8
        sr = cfg.sample_rate
        chips_per_symbol = max(1, int(cfg.dsss_chip_rate / cfg.symbol_rate))
        spb = int(sr / (cfg.dsss_chip_rate))
        base = []
        for b in bits:
            bit_val = 1.0 if b else -1.0
            ch = bit_val * pn
            ch = np.repeat(ch, spb)
            base.append(ch)
        baseband = np.concatenate(base) if base else np.zeros(0, dtype=np.float32)
        # Upconvert to audio carrier
        t = np.arange(len(baseband))/sr
        audio = cfg.amplitude * baseband * np.sin(2*np.pi*cfg.fc*t)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32)

    @staticmethod
    def ofdm(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        # Toy OFDM: QPSK mapping across N subcarriers, IFFT, add cyclic prefix
        N = cfg.ofdm_subc
        spb_sym = int(cfg.sample_rate / cfg.symbol_rate)  # samples per OFDM symbol (approx shaping)
        chunks = chunk_bits(bits, 2*N)
        a = cfg.amplitude
        wave = []
        iq = []
        for ch in chunks:
            # map 2 bits -> QPSK symbol
            qsyms = []
            pairs = chunk_bits(ch, 2)
            for p in pairs:
                b0,b1 = (p+[0,0])[:2]
                if (b0,b1)==(0,0): s = 1+1j
                elif (b0,b1)==(0,1): s = -1+1j
                elif (b0,b1)==(1,1): s = -1-1j
                else: s = 1-1j
                qsyms.append(s/math.sqrt(2))
            # pad to N
            if len(qsyms) < N:
                qsyms += [0j]*(N-len(qsyms))
            Xk = np.array(qsyms, dtype=np.complex64)
            xt = np.fft.ifft(Xk)  # time domain symbol (complex)
            # cyclic prefix
            cp = xt[-cfg.cp_len:]
            sym = np.concatenate([cp, xt])
            # stretch to samples-per-symbol for audio mixing
            reps = max(1, int(spb_sym/len(sym)))
            sym_up = np.repeat(sym, reps)
            # audio upconvert
            t = np.arange(len(sym_up))/cfg.sample_rate
            audio = a*(sym_up.real*np.cos(2*np.pi*cfg.fc*t) - sym_up.imag*np.sin(2*np.pi*cfg.fc*t))
            wave.append(audio.astype(np.float32))
            iq.append((a*sym_up).astype(np.complex64))
        audio = np.concatenate(wave) if wave else np.zeros(0, dtype=np.float32)
        iqc = np.concatenate(iq) if iq else np.zeros(0, dtype=np.complex64)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio, iqc

# =========================================================
# LLM backends (Local final inference; Remote resource-only)
# =========================================================

class BaseLLM:
    def generate(self, prompt: str, **kwargs) -> str: raise NotImplementedError

class LocalLLM(BaseLLM):
    def __init__(self, configs: List[HTTPConfig]):
        if requests is None:
            raise RuntimeError("LocalLLM requires 'requests' (pip install requests)")
        self.configs = configs
        self.idx = 0

    def generate(self, prompt: str, **kwargs) -> str:
        last = None
        for _ in range(len(self.configs)):
            cfg = self.configs[self.idx]
            try:
                out = self._call(cfg, prompt, **kwargs)
                return out
            except Exception as e:
                last = e
                self.idx = (self.idx + 1) % len(self.configs)
        raise last or RuntimeError("All local LLM configs failed")

    def _post(self, cfg: HTTPConfig, url: str, headers: dict, body: dict) -> dict:
        s = requests.Session()
        for attempt in range(cfg.max_retries):
            try:
                r = s.post(url, headers=headers, json=body, timeout=cfg.timeout, verify=cfg.verify_ssl)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt < cfg.max_retries-1:
                    time.sleep(cfg.retry_delay*(2**attempt))
                else:
                    raise

    def _call(self, cfg: HTTPConfig, prompt: str, **kwargs) -> str:
        mode = cfg.mode
        if mode == "openai-chat":
            url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: headers["Authorization"] = f"Bearer {cfg.api_key}"
            body = {
                "model": cfg.model or "gpt-4o-mini",
                "messages": [{"role":"user","content":prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["message"]["content"]
        if mode == "openai-completions":
            url = f"{cfg.base_url.rstrip('/')}/v1/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: headers["Authorization"] = f"Bearer {cfg.api_key}"
            body = {
                "model": cfg.model or "gpt-3.5-turbo-instruct",
                "prompt": prompt,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["text"]
        if mode == "llama-cpp":
            url = f"{cfg.base_url.rstrip('/')}/completion"
            body = {"prompt": prompt, "temperature": kwargs.get("temperature",0.7), "n_predict": kwargs.get("max_tokens",512)}
            data = self._post(cfg, url, {}, body)
            if "content" in data: return data["content"]
            if "choices" in data and data["choices"]: return data["choices"][0].get("text","")
            return data.get("text","")
        if mode == "textgen-webui":
            url = f"{cfg.base_url.rstrip('/')}/api/v1/generate"
            body = {"prompt": prompt, "max_new_tokens": kwargs.get("max_tokens",512), "temperature": kwargs.get("temperature",0.7)}
            data = self._post(cfg, url, {}, body)
            return data.get("results",[{}])[0].get("text","")
        raise ValueError(f"Unsupported mode: {mode}")

class ResourceLLM(BaseLLM):
    def __init__(self, cfg: Optional[HTTPConfig] = None):
        self.cfg = cfg

    def generate(self, prompt: str, **kwargs) -> str:
        # Constrained to resources-only summarization
        if self.cfg is None or requests is None:
            return LocalSummarizer().summarize(prompt)
        url = f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type":"application/json"}
        if self.cfg.api_key: headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        system = ("You are a constrained assistant. ONLY summarize/structure the provided INPUT RESOURCES. "
                  "Do not add external knowledge.")
        body = {
            "model": self.cfg.model or "gpt-4o-mini",
            "messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
            "temperature": kwargs.get("temperature",0.2),
            "max_tokens": kwargs.get("max_tokens",512),
        }
        s = requests.Session()
        r = s.post(url, headers=headers, json=body, timeout=self.cfg.timeout, verify=self.cfg.verify_ssl)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

class LocalSummarizer:
    def __init__(self):
        self.stop = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with","by","is","are",
            "was","were","be","been","being","have","has","had","do","does","did","will","would",
            "could","should","from","that","this","it","as"
        }
    def summarize(self, text: str) -> str:
        txt = " ".join(text.split())
        if not txt: return "No content to summarize."
        sents = [s.strip() for s in txt.replace("?",".").replace("!",".").split(".") if s.strip()]
        if not sents: return txt[:300] + ("..." if len(txt)>300 else "")
        # score sentences by length + term frequency (very light heuristic)
        words = [w.lower().strip(",;:()[]") for w in txt.split()]
        freq: Dict[str,int] = {}
        for w in words:
            if w and w not in self.stop: freq[w] = freq.get(w,0)+1
        scored = []
        for s in sents:
            sw = [w.lower().strip(",;:()[]") for w in s.split()]
            score = len(s) * 0.1 + sum(freq.get(w,0) for w in sw)
            scored.append((s, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        keep = [s for s,_ in scored[: min(6,len(scored))]]
        keep.sort(key=lambda k: sents.index(k))
        out = " ".join(keep)
        return out[:800] + ("..." if len(out)>800 else "")

# =========================================================
# Orchestrator
# =========================================================

class DualLLMOrchestrator:
    def __init__(self, local: LocalLLM, resource: ResourceLLM, settings: OrchestratorSettings):
        self.local, self.resource, self.set = local, resource, settings

    def _load_resources(self, paths: List[str], inline: List[str]) -> str:
        parts = []
        for p in paths:
            pa = Path(p)
            if pa.exists() and pa.is_file():
                try:
                    parts.append(pa.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    parts.append(f"[[UNREADABLE_FILE:{pa.name}]]")
            else:
                parts.append(f"[[MISSING_FILE:{pa}]]")
        parts += [str(x) for x in inline]
        blob = "\n\n".join(parts)
        return blob[: self.set.max_context_chars]

    def compose(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Tuple[str,str]:
        res_text = self._load_resources(resource_paths, inline_resources)
        res_summary = self.resource.generate(
            f"INPUT RESOURCES:\n{res_text}\n\nTASK: Summarize/structure ONLY the content above.",
            temperature=0.2, max_tokens=self.set.max_tokens
        )
        final_prompt = (
            "You are a LOCAL expert system. Use ONLY the structured summary below; do not invent facts.\n\n"
            f"=== STRUCTURED SUMMARY ===\n{res_summary}\n\n"
            f"=== USER PROMPT ===\n{user_prompt}\n\n"
            f"STYLE: {self.set.style}. Be clear and directly actionable."
        )
        return final_prompt, res_summary

    def run(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Dict[str,str]:
        fp, summary = self.compose(user_prompt, resource_paths, inline_resources)
        ans = self.local.generate(fp, temperature=self.set.temperature, max_tokens=self.set.max_tokens)
        return {"summary": summary, "final": ans, "prompt": fp}

# =========================================================
# End-to-end casting
# =========================================================

@dataclass
class OutputPaths:
    wav: Optional[Path] = None
    iq: Optional[Path] = None
    meta: Optional[Path] = None
    png: Optional[Path] = None

def bits_to_signals(bits: List[int], scheme: ModulationScheme, mcfg: ModConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if scheme == ModulationScheme.BFSK:
        return Modulators.bfsK(bits, mcfg), None
    if scheme == ModulationScheme.AFSK:
        return Modulators.afsK(bits, mcfg), None
    if scheme == ModulationScheme.BPSK:
        return Modulators.bpsK(bits, mcfg)
    if scheme == ModulationScheme.QPSK:
        return Modulators.qpsK(bits, mcfg)
    if scheme == ModulationScheme.QAM16:
        return Modulators.qam16(bits, mcfg)
    if scheme == ModulationScheme.OFDM:
        return Modulators.ofdm(bits, mcfg)
    if scheme == ModulationScheme.DSSS_BPSK:
        return Modulators.dsss_bpsK(bits, mcfg), None
    raise ValueError("Unknown modulation scheme")

def cast_to_files(
    text: str,
    outdir: Path,
    scheme: ModulationScheme,
    mcfg: ModConfig,
    want_wav: bool,
    want_iq: bool,
    title: str = "WaveCaster"
) -> OutputPaths:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    base = outdir / f"cast_{scheme.name.lower()}_{ts}"
    # Minimal frame (no FEC/security here; caller handles)
    fcfg = FrameConfig()
    bits = to_bits(frame_payload(text.encode("utf-8"), fcfg))
    audio, iq = bits_to_signals(bits, scheme, mcfg)
    paths = OutputPaths()
    if want_wav and audio is not None and len(audio)>0:
        paths.wav = base.with_suffix(".wav"); write_wav_mono(paths.wav, audio, mcfg.sample_rate)
    if want_iq:
        if iq is None and audio is not None:
            # make a naive hilbert to IQ for convenience
            try:
                q = np.imag(sp_signal.hilbert(audio))
                iq = audio.astype(np.float32) + 1j*q.astype(np.float32)
            except Exception:
                iq = (audio.astype(np.float32) + 1j*np.zeros_like(audio, dtype=np.float32))
        if iq is not None:
            paths.iq = base.with_suffix(".iqf32"); write_iq_f32(paths.iq, iq)
    # Visualization
    if audio is not None and len(audio)>0 and HAS_MPL:
        paths.png = base.with_suffix(".png"); plot_wave_and_spectrum(paths.png, audio, mcfg.sample_rate, title)
    # Meta
    meta = {
        "timestamp": ts, "scheme": scheme.name, "sample_rate": mcfg.sample_rate,
        "symbol_rate": mcfg.symbol_rate, "framesec": len(audio)/mcfg.sample_rate if audio is not None else 0,
    }
    paths.meta = base.with_suffix(".json")
    paths.meta.write_text(safe_json(meta), encoding="utf-8")
    return paths

def full_cast_and_save(
    text: str,
    outdir: Path,
    scheme: ModulationScheme,
    mcfg: ModConfig,
    fcfg: FrameConfig,
    sec: SecurityConfig,
    fec_scheme: FEC,
    want_wav: bool,
    want_iq: bool,
    title: str = "WaveCaster"
) -> OutputPaths:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    base = outdir / f"cast_{scheme.name.lower()}_{ts}"
    bits = encode_text(text, fcfg, sec, fec_scheme)
    audio, iq = bits_to_signals(bits, scheme, mcfg)
    paths = OutputPaths()
    if want_wav and audio is not None and len(audio)>0:
        paths.wav = base.with_suffix(".wav"); write_wav_mono(paths.wav, audio, mcfg.sample_rate)
    if want_iq:
        if iq is None and audio is not None:
            try:
                q = np.imag(sp_signal.hilbert(audio))
                iq = audio.astype(np.float32) + 1j*q.astype(np.float32)
            except Exception:
                iq = (audio.astype(np.float32) + 1j*np.zeros_like(audio, dtype=np.float32))
        if iq is not None:
            paths.iq = base.with_suffix(".iqf32"); write_iq_f32(paths.iq, iq)
    if audio is not None and len(audio)>0 and HAS_MPL:
        paths.png = base.with_suffix(".png"); plot_wave_and_spectrum(paths.png, audio, mcfg.sample_rate, title)
    meta = {
        "timestamp": ts, "scheme": scheme.name, "sample_rate": mcfg.sample_rate,
        "symbol_rate": mcfg.symbol_rate, "framesec": len(audio)/mcfg.sample_rate if audio is not None else 0,
        "fec": fec_scheme.name, "encrypted": bool(sec.password), "watermark": bool(sec.watermark),
        "hmac": bool(sec.hmac_key),
    }
    paths.meta = base.with_suffix(".json"); paths.meta.write_text(safe_json(meta), encoding="utf-8")
    return paths

# =========================================================
# CLI
# =========================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dual_llm_wavecaster_enhanced", description="Two-LLM orchestration → modulated waveform")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_mod_args(sp):
        sp.add_argument("--scheme", choices=[s.name.lower() for s in ModulationScheme], default="bfsk")
        sp.add_argument("--sample-rate", type=int, default=48000)
        sp.add_argument("--symbol-rate", type=int, default=1200)
        sp.add_argument("--amplitude", type=float, default=0.7)
        sp.add_argument("--f0", type=float, default=1200.0)
        sp.add_argument("--f1", type=float, default=2200.0)
        sp.add_argument("--fc", type=float, default=1800.0)
        sp.add_argument("--no-clip", action="store_true")
        sp.add_argument("--outdir", type=str, default="casts")
        sp.add_argument("--wav", action="store_true")
        sp.add_argument("--iq", action="store_true")
        sp.add_argument("--play", action="store_true", help="Play audio to soundcard (if available)")

        # OFDM / DSSS
        sp.add_argument("--ofdm-subc", type=int, default=64)
        sp.add_argument("--cp-len", type=int, default=16)
        sp.add_argument("--dsss-chip-rate", type=int, default=4800)

    # cast: 2-LLM orchestration then modulate
    sp_cast = sub.add_parser("cast", help="Compose via dual LLMs then modulate")
    sp_cast.add_argument("--prompt", type=str, required=True)
    sp_cast.add_argument("--resource-file", nargs="*", default=[])
    sp_cast.add_argument("--resource-text", nargs="*", default=[])
    # Local LLM
    sp_cast.add_argument("--local-url", type=str, default="http://127.0.0.1:8080")
    sp_cast.add_argument("--local-mode", choices=["openai-chat","openai-completions","llama-cpp","textgen-webui"], default="llama-cpp")
    sp_cast.add_argument("--local-model", type=str, default="local-gguf")
    sp_cast.add_argument("--local-key", type=str, default=None)
    # Remote Resource LLM
    sp_cast.add_argument("--remote-url", type=str, default=None)
    sp_cast.add_argument("--remote-model", type=str, default="gpt-4o-mini")
    sp_cast.add_argument("--remote-key", type=str, default=None)
    # Orchestration params
    sp_cast.add_argument("--style", type=str, default="concise")
    sp_cast.add_argument("--max-tokens", type=int, default=512)
    sp_cast.add_argument("--temperature", type=float, default=0.7)
    # Security / FEC
    sp_cast.add_argument("--password", type=str, default=None)
    sp_cast.add_argument("--watermark", type=str, default=None)
    sp_cast.add_argument("--hmac-key", type=str, default=None)
    sp_cast.add_argument("--fec", choices=[f.name.lower() for f in FEC], default="hamming74")
    add_mod_args(sp_cast)

    # modulate: direct text to waveform
    sp_mod = sub.add_parser("modulate", help="Modulate provided text directly")
    sp_mod.add_argument("--text", type=str, required=True)
    sp_mod.add_argument("--password", type=str, default=None)
    sp_mod.add_argument("--watermark", type=str, default=None)
    sp_mod.add_argument("--hmac-key", type=str, default=None)
    sp_mod.add_argument("--fec", choices=[f.name.lower() for f in FEC], default="none")
    add_mod_args(sp_mod)

    # visualize existing WAV
    sp_vis = sub.add_parser("visualize", help="Plot waveform + spectrum from WAV")
    sp_vis.add_argument("--wav", type=str, required=True)
    sp_vis.add_argument("--out", type=str, default=None)

    # analyze: print basic metrics
    sp_an = sub.add_parser("analyze", help="Basic audio metrics of WAV")
    sp_an.add_argument("--wav", type=str, required=True)

    return p

def make_modcfg(args: argparse.Namespace) -> ModConfig:
    return ModConfig(
        sample_rate=args.sample_rate, symbol_rate=args.symbol_rate, amplitude=args.amplitude,
        f0=args.f0, f1=args.f1, fc=args.fc, clip=not args.no_clip,
        ofdm_subc=getattr(args, "ofdm_subc", 64), cp_len=getattr(args,"cp_len",16),
        dsss_chip_rate=getattr(args,"dsss_chip_rate",4800),
    )

def parse_scheme(s: str) -> ModulationScheme:
    return ModulationScheme[s.upper()]

def parse_fec(s: str) -> FEC:
    return FEC[s.upper()]

def cmd_cast(args: argparse.Namespace) -> int:
    # Build LLMs
    local = LocalLLM([HTTPConfig(
        base_url=args.local_url, model=args.local_model, mode=args.local_mode, api_key=args.local_key
    )])
    rcfg = HTTPConfig(base_url=args.remote_url, model=args.remote_model, api_key=args.remote_key) if args.remote_url else None
    resource = ResourceLLM(rcfg)
    orch = DualLLMOrchestrator(local, resource, OrchestratorSettings(
        temperature=args.temperature, max_tokens=args.max_tokens, style=args.style
    ))
    result = orch.run(args.prompt, args.resource_file, args.resource_text)
    # Build pipeline
    mcfg = make_modcfg(args)
    fcfg = FrameConfig()
    sec = SecurityConfig(password=args.password, watermark=args.watermark, hmac_key=args.hmac_key)
    scheme = parse_scheme(args.scheme)
    fec_s = parse_fec(args.fec)
    paths = full_cast_and_save(
        text=result["final"], outdir=Path(args.outdir), scheme=scheme, mcfg=mcfg, fcfg=fcfg,
        sec=sec, fec_scheme=fec_s, want_wav=args.wav or (not args.iq), want_iq=args.iq,
        title=f"{scheme.name} | DualLLM Wave"
    )
    if args.play and paths.wav and HAS_AUDIO:
        import soundfile as sf
        try:
            data, sr = sf.read(str(paths.wav), dtype="float32")
            play_audio(data, sr)
        except Exception:
            # Fallback: play from generated buffer (we already have audio only inside full_cast if we changed it to return)
            log.warning("Install soundfile for playback of saved WAV, or use --play with 'modulate'")
    print(safe_json({
        "files": {"wav": str(paths.wav) if paths.wav else None,
                  "iq": str(paths.iq) if paths.iq else None,
                  "meta": str(paths.meta) if paths.meta else None,
                  "png": str(paths.png) if paths.png else None},
        "preview": result["final"][:400],
        "summary": result["summary"][:400],
    }))
    return 0

def cmd_modulate(args: argparse.Namespace) -> int:
    mcfg = make_modcfg(args)
    fcfg = FrameConfig()
    sec = SecurityConfig(password=args.password, watermark=args.watermark, hmac_key=args.hmac_key)
    scheme = parse_scheme(args.scheme)
    fec_s = parse_fec(args.fec)
    paths = full_cast_and_save(
        text=args.text, outdir=Path(args.outdir), scheme=scheme, mcfg=mcfg, fcfg=fcfg,
        sec=sec, fec_scheme=fec_s, want_wav=args.wav or (not args.iq), want_iq=args.iq,
        title=f"{scheme.name} | Direct Mod"
    )
    if args.play and paths.wav:
        try:
            import soundfile as sf
            data, sr = sf.read(str(paths.wav), dtype="float32"); play_audio(data, sr)
        except Exception:
            log.warning("Install soundfile for playback of saved WAV")
    print(safe_json({"files": {"wav": str(paths.wav) if paths.wav else None,
                               "iq": str(paths.iq) if paths.iq else None,
                               "meta": str(paths.meta) if paths.meta else None,
                               "png": str(paths.png) if paths.png else None}}))
    return 0

def cmd_visualize(args: argparse.Namespace) -> int:
    if not HAS_MPL:
        print("matplotlib is not installed.")
        return 1
    import wave
    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        s = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
    out = Path(args.out or (Path(args.wav).with_suffix(".png")))
    plot_wave_and_spectrum(out, s, sr, f"Visualize: {Path(args.wav).name}")
    print(safe_json({"png": str(out), "sample_rate": sr, "seconds": len(s)/sr}))
    return 0

def cmd_analyze(args: argparse.Namespace) -> int:
    import wave
    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate(); n = w.getnframes()
        s = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
    dur = len(s)/sr
    rms = float(np.sqrt(np.mean(s**2)))
    peak = float(np.max(np.abs(s)))
    spec = np.abs(rfft(s)); spec /= (spec.max()+1e-12)
    # simple SNR estimate
    snr = 10*np.log10(np.mean(s**2) / (np.var(s - np.mean(s)) + 1e-12))
    print(safe_json({"sample_rate": sr, "seconds": dur, "rms": rms, "peak": peak, "snr_db": float(snr)}))
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "cast": return cmd_cast(args)
    if args.cmd == "modulate": return cmd_modulate(args)
    if args.cmd == "visualize": return cmd_visualize(args)
    if args.cmd == "analyze": return cmd_analyze(args)
    p.print_help(); return 2

if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3

mirror_cast_enhanced.py — Neuro-Symbolic Adaptive Reflective Engine + Digital Link

SPDX-License-Identifier: MIT

import argparse import json import math import time import uuid import hashlib import zlib import os from dataclasses import dataclass from typing import Any,Dict, List, Optional, Tuple

import numpy as np

Optional plotting

try: import matplotlib.pyplot as plt HAVE_MPL = True except Exception: HAVE_MPL = False

========================= Core Analytics Modules ============================

class EntropyAnalyzer: def measure(self, data: Any) -> float: s = str(data) if not s: return 0.0 counts: Dict[str, int] = {} for c in s: counts[c] = counts.get(c, 0) + 1 n = len(s) ent = 0.0 for cnt in counts.values(): p = cnt / n if p > 0: ent -= p * math.log2(p) return ent

class DianneReflector: def reflect(self, data: Any) -> Dict[str, Any]: patterns = self._detect_patterns(data) head = str(data)[:40].replace("\n", " ") if "high_repetition" in patterns: insight = f"Cyclical resonance detected in Reflecting essence of: {head}..." elif "hierarchical_structure" in patterns: insight = f"Nested reality layers within Reflecting essence of: {head}..." else: insight = f"Linear transformation potential in Reflecting essence of: {head}..." return {"insight": insight, "patterns": patterns, "symbolic_depth": self._depth(data)}

class MatrixTransformer: def project(self, data: Any) -> Dict[str, Any]: dims = self._analyze(data) h = hash(str(data)) & 0xFFFFFFFF rank = int(dims["rank"]) eivals = [math.sin(h * 0.001 * i) for i in range(max(1, min(3, rank)))] return { "projected_rank": dims["rank"], "structure": dims["structure"], "eigenvalues": eivals, "determinant": math.cos(h * 0.0001), "trace": (math.tan(h * 0.00001) if (h % 100) else 0.0), }

class JuliaSymbolEngine: def analyze(self, data: Any) -> Dict[str, Any]: coeffs = self._coeffs(data) return { "chebyshev_polynomial": self._poly(coeffs), "coefficients": coeffs, "derivatives": self._derivs(coeffs), "critical_points": self._crit(coeffs), }

class ChoppyProcessor: def chunk(self, data: Any, chunk_size: int = 64, overlap: int = 16) -> Dict[str, Any]: s = str(data) step = max(1, chunk_size - overlap) std = [s[i:i + chunk_size] for i in range(0, len(s), step)] words = s.split(); wsize = max(1, chunk_size // 5) wchunks = [" ".join(words[i:i + wsize]) for i in range(0, len(words), wsize)] return { "standard": std, "semantic": wchunks, "fibonacci": self._fib(s), "statistics": {"total_length": len(s), "chunk_count": len(std), "average_chunk_size": len(s) / max(1, len(std))}, }

class EndpointCaster: def generate(self, data: Any) -> Dict[str, Any]: sig = hashlib.sha256(json.dumps(data, default=str, sort_keys=True).encode()).hexdigest()[:12] base = uuid.uuid4().hex[:6] return { "primary_endpoint": f"/api/v1/cast/{base}", "versioned_endpoints": [
                f"/api/v1/cast/{base}/reflect",
                f"/api/v1/cast/{base}/transform",
                f"/api/v1/cast/{base}/metadata",
                f"/api/v2/mirror/{sig}",
            ], "artifact_id": f"art-{uuid.uuid4().hex[:8]}", "metadata": {"content_type": self._ctype(data), "estimated_size": len(str(data)), "complexity": self._cpx(data)}, }

class CarryOnManager: def init(self, max_history: int = 200): self.memory: Dict[str, Any] = {} self.history: List[Dict[str, Any]] = [] self.max_history = max_history self.access: Dict[str, int] = {}

class SemanticMapper: def init(self): self.semantic_networks = { "reflection": ["mirror", "echo", "reverberation", "contemplation", "introspection"], "transformation": ["metamorphosis", "mutation", "evolution", "adaptation", "transmutation"], "analysis": ["examination", "scrutiny", "dissection", "investigation", "exploration"], "synthesis": ["combination", "fusion", "amalgamation", "integration", "unification"], }

class LoveReflector: def infuse(self, data: Any) -> Dict[str, Any]: text = str(data) return {"poetic": self._poem(text), "emotional_resonance": self._emo(text), "love_quotient": self._lq(text), "harmony_index": self._hi(text)}

class FractalResonator: def init(self, max_depth: int = 8): self.max_depth = max_depth

===================== Neuro-Symbolic Control & Memory =======================

class FeatureExtractor: """ Lightweight local features + optional imported embedding. - text n-gram hashing → fixed-width vector - optional external embedding (pass via CLI JSON file or string) """ def init(self, dim: int = 64, ngram: int = 3): self.dim = dim self.ngram = ngram

class NeuroSymbolicFusion: """ Fuse neural features + symbolic metrics (fractal, entropy, reflector tags) Produce a decision suggestion and scores. """ def init(self): # Learned (static) weights for demo; could be trained via RL self.w_neuro = 0.55 self.w_symbol = 0.45

class DecisionLogger: def init(self): self.events: List[Dict[str, Any]] = []

class ReflectiveDB: """JSON file for self-tuning memory of configs & outcomes.""" def init(self, path: str = "reflective_db.json"): self.path = path self._data: List[Dict[str, Any]] = [] self._load()

class RLAgent: """ Tiny contextual bandit: state = bins(peak_entropy_depth, ssi, snr) actions = {'bpsk', 'qpsk', 'ofdm'} update via incremental mean reward. """ def init(self, actions: List[str] = None, eps: float = 0.1): self.actions = actions or ["bpsk", "qpsk", "ofdm"] self.eps = eps self.q: Dict[Tuple[int, int, int], Dict[str, Dict[str, float]]] = {}  # state -> action -> {q, n}

=========================== Digital Link (SC + OFDM) ========================

@dataclass class LinkConfig: mod: str = "qpsk"          # 'bpsk' | 'qpsk' | 'ofdm' sps: int = 8               # samples per symbol (SC) symbol_rate: int = 2000 snr_db: float = 30.0 rrc_alpha: float = 0.35 rrc_taps: int = 121 # OFDM params nfft: int = 256 cp_len: int = 32 subcarrier_mask: Optional[List[int]] = None  # indices to use (None -> all data carriers)

class Modem: """SC BPSK/QPSK + minimal OFDM with QPSK mapping."""

======================= Mirror Cast + Adaptive Planner =======================

class MirrorCastEngine: def init(self): self.entropy = EntropyAnalyzer() self.reflector = DianneReflector() self.matrix = MatrixTransformer() self.symbols = JuliaSymbolEngine() self.choppy = ChoppyProcessor() self.endpoints = EndpointCaster() self.memory = CarryOnManager() self.semantic = SemanticMapper() self.love = LoveReflector() self.fractal = FractalResonator()

class AdaptiveLinkPlanner: """ Neuro-Symbolic + RL planner: 1) Extract features, fuse with symbolic metrics 2) RL agent selects action (mod) 3) Map ssi -> rolloff, peak -> sps, action->mod/OFDM mask 4) Produce decision log & explanation """ def init(self, db_path: str = "reflective_db.json"): self.extractor = FeatureExtractor() self.fusion = NeuroSymbolicFusion() self.agent = RLAgent(actions=["bpsk", "qpsk", "ofdm"], eps=0.1) self.db = ReflectiveDB(db_path) self.log = DecisionLogger()

=============================== Visualization ===============================

def maybe_plot_constellation(syms: np.ndarray, title: str = "Constellation"): if not HAVE_MPL: return plt.figure() plt.scatter(np.real(syms), np.imag(syms), s=8) plt.title(title); plt.xlabel("I"); plt.ylabel("Q"); plt.grid(True)

def maybe_plot_fractal_layers(fractal: Dict[str, Any]): if not HAVE_MPL: return xs = [L["depth"] for L in fractal["layers"]] ys = [L["entropy"] for L in fractal["layers"]] plt.figure(); plt.plot(xs, ys, marker="o") plt.title("Fractal Entropy vs Depth"); plt.xlabel("Depth"); plt.ylabel("Entropy"); plt.grid(True)

def maybe_plot_decisions(decisions: List[Dict[str, Any]]): if not HAVE_MPL: return # simple timeline of events plt.figure() ys = list(range(len(decisions))) labels = [f"{d['step']}:{d['reason']}" for d in decisions] plt.plot(ys, ys, marker="o") for i, txt in enumerate(labels): plt.text(i, i, txt, fontsize=8) plt.title("Decision Pathway"); plt.xlabel("event idx"); plt.ylabel("event idx"); plt.grid(True)

================================== CLI ======================================

def cmd_report(args: argparse.Namespace) -> None: eng = MirrorCastEngine() data = json.loads(args.input) if args.json else args.input report = eng.cast(data) print(json.dumps(report, indent=2)) if args.plot and HAVE_MPL: maybe_plot_fractal_layers(report["fractal"]) plt.show()

def cmd_txrx(args: argparse.Namespace) -> None: cfg = LinkConfig( mod=args.mod.lower(), sps=args.sps, symbol_rate=args.symbol_rate, snr_db=args.snr, rrc_alpha=args.alpha, rrc_taps=args.taps, nfft=args.nfft, cp_len=args.cp, ) modem = Modem(cfg) payload = args.input.encode("utf-8") tx = modem.transmit(payload) ch = modem.channel_awgn(tx) rx_bytes, info = modem.receive(ch) out = {"config": cfg.dict, "status": info, "transmitted_len": len(payload), "received_len": len(rx_bytes), "received_text": (rx_bytes.decode("utf-8", errors="replace") if info.get("ok") else "")} print(json.dumps(out, indent=2)) if args.plot and HAVE_MPL and cfg.mod in ("qpsk", "bpsk"): # reconstruct baseband symbols for visualization (approx using downsampled MF output if SC) pass

def cmd_fractal(args: argparse.Namespace) -> None: fr = FractalResonator(max_depth=args.max_depth) result = fr.cascade(args.input) print(json.dumps(result, indent=2)) if args.plot and HAVE_MPL: maybe_plot_fractal_layers(result); plt.show()

def cmd_txrx_adapt(args: argparse.Namespace) -> None: engine = MirrorCastEngine() analysis = engine.cast(args.input)  # includes fractal, entropy, reflection planner = AdaptiveLinkPlanner(db_path=args.db) cfg, explanation = planner.plan(args.input, LinkConfig( mod="qpsk", sps=args.sps, symbol_rate=args.symbol_rate, snr_db=args.snr, rrc_alpha=args.alpha, rrc_taps=args.taps, nfft=args.nfft, cp_len=args.cp ), analysis, snr_db=args.snr)

def cmd_learn(args: argparse.Namespace) -> None: """ Run multiple episodes on the same input (or a seed list) to let the RL agent adapt. """ engine = MirrorCastEngine() planner = AdaptiveLinkPlanner(db_path=args.db) texts = [args.input] if args.input else [
        "industrial IoT telemetry window 42 pressure spike",
        "AR/VR immersive stream scene: waterfall shimmer",
        "assistive comms: phrasebook quick intent",
        "healthcare vitals batch 28dB link margin"
    ] for ep in range(args.episodes): t = texts[ep % len(texts)] analysis = engine.cast(t) cfg, explanation = planner.plan(t, LinkConfig(snr_db=args.snr, nfft=args.nfft, cp_len=args.cp), analysis, snr_db=args.snr) modem = Modem(cfg) payload = t.encode("utf-8") tx = modem.transmit(payload) ch = modem.channel_awgn(tx) rx_bytes, info = modem.receive(ch) decode_ok = bool(info.get("ok")) harmony = engine.love.infuse(t)["harmony_index"] planner.reward_and_record(t, cfg, explanation, decode_ok, args.snr, harmony, len(payload), len(rx_bytes)) print(json.dumps({ "episode": ep+1, "text_sig": hashlib.sha256(t.encode()).hexdigest()[:12], "config": cfg.dict, "decode_ok": decode_ok }, indent=2)) print(f"Learning complete. DB at {args.db}")

def build_parser() -> argparse.ArgumentParser: p = argparse.ArgumentParser(description="Mirror Cast Enhanced — Neuro-Symbolic Adaptive Link") sub = p.add_subparsers(dest="cmd", required=True)

def main(): parser = build_parser() args = parser.parse_args() args.func(args)

if name == "main": main()
Enhanced Dual LLM WaveCaster with Neuro-Symbolic Adaptive Reflective Engine

Here's the complete enhanced implementation combining both the Dual LLM WaveCaster and the Neuro-Symbolic Adaptive Reflective Engine:
python

#!/usr/bin/env python3
# dual_llm_wavecaster_enhanced.py
# SPDX-License-Identifier: MIT
"""
Enhanced Dual LLM WaveCaster with Neuro-Symbolic Adaptive Reflective Engine
----------------------------------------------------------------------------
Two-LLM orchestration (local final inference + remote resource-only summaries) → framed bits
→ modulated waveform (BFSK/BPSK/QPSK/16QAM/AFSK/OFDM) → WAV/IQ files (+ optional audio out)
with visualization, simple FEC, encryption, watermarking, and metadata.

Deps (minimum):
  pip install numpy scipy requests

Optional:
  pip install matplotlib sounddevice pycryptodome

Quick start:
  python dual_llm_wavecaster_enhanced.py modulate --text "hello airwaves" --scheme qpsk --wav --iq
  python dual_llm_wavecaster_enhanced.py cast --prompt "2-paragraph plan" \
      --resource-file notes.txt --local-url http://127.0.0.1:8080 --local-mode llama-cpp \
      --remote-url https://api.openai.com --remote-key $OPENAI_API_KEY --scheme bfsk --wav
"""

from __future__ import annotations
import argparse, base64, binascii, hashlib, json, logging, math, os, struct, sys, time, warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
from enum import Enum, auto
from datetime import datetime

# ---------- Hard requirements ----------
try:
    import numpy as np
    from scipy import signal as sp_signal
    from scipy.fft import rfft, rfftfreq
except Exception as e:
    raise SystemExit("numpy and scipy are required: pip install numpy scipy") from e

# ---------- Optional dependencies ----------
try:
    import requests
except Exception:
    requests = None  # HTTP backends disabled if missing

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Protocol.KDF import PBKDF2
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("wavecaster")

# =========================================================
# Enums / Config
# =========================================================

class ModulationScheme(Enum):
    BFSK = auto()
    BPSK = auto()
    QPSK = auto()
    QAM16 = auto()
    AFSK = auto()
    OFDM = auto()
    DSSS_BPSK = auto()

class FEC(Enum):
    NONE = auto()
    HAMMING74 = auto()
    REED_SOLOMON = auto()   # stub
    LDPC = auto()           # stub
    TURBO = auto()          # stub

@dataclass
class HTTPConfig:
    base_url: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 60
    mode: str = "openai-chat"  # ["openai-chat","openai-completions","llama-cpp","textgen-webui"]
    verify_ssl: bool = True
    max_retries: int = 2
    retry_delay: float = 0.8

@dataclass
class OrchestratorSettings:
    temperature: float = 0.7
    max_tokens: int = 512
    style: str = "concise"
    max_context_chars: int = 8000

@dataclass
class ModConfig:
    sample_rate: int = 48000
    symbol_rate: int = 1200
    amplitude: float = 0.7
    f0: float = 1200.0     # BFSK 0
    f1: float = 2200.0     # BFSK 1
    fc: float = 1800.0     # PSK/QAM audio carrier (for WAV)
    clip: bool = True
    # OFDM (toy)
    ofdm_subc: int = 64
    cp_len: int = 16
    # DSSS
    dsss_chip_rate: int = 4800

@dataclass
class FrameConfig:
    use_crc32: bool = True
    use_crc16: bool = False
    preamble: bytes = b"\x55" * 8  # 01010101 * 8
    version: int = 1

# =========================================================
# Neuro-Symbolic Adaptive Reflective Engine
# =========================================================

class EntropyAnalyzer:
    def measure(self, data: Any) -> float:
        s = str(data)
        if not s:
            return 0.0
        counts: Dict[str, int] = {}
        for c in s:
            counts[c] = counts.get(c, 0) + 1
        n = len(s)
        ent = 0.0
        for cnt in counts.values():
            p = cnt / n
            if p > 0:
                ent -= p * math.log2(p)
        return ent

class DianneReflector:
    def reflect(self, data: Any) -> Dict[str, Any]:
        patterns = self._detect_patterns(data)
        head = str(data)[:40].replace("\n", " ")
        if "high_repetition" in patterns:
            insight = f"Cyclical resonance detected in Reflecting essence of: {head}..."
        elif "hierarchical_structure" in patterns:
            insight = f"Nested reality layers within Reflecting essence of: {head}..."
        else:
            insight = f"Linear transformation potential in Reflecting essence of: {head}..."
        return {"insight": insight, "patterns": patterns, "symbolic_depth": self._depth(data)}
    
    def _detect_patterns(self, data: Any) -> List[str]:
        s = str(data)
        patterns = []
        if len(s) > 100 and len(set(s)) < 20:
            patterns.append("high_repetition")
        if s.count('\n') > 5 and any(c in s for c in ['{', '[', '(', '<']):
            patterns.append("hierarchical_structure")
        return patterns
    
    def _depth(self, data: Any) -> int:
        s = str(data)
        return min(10, len(s) // 100)

class MatrixTransformer:
    def project(self, data: Any) -> Dict[str, Any]:
        dims = self._analyze(data)
        h = hash(str(data)) & 0xFFFFFFFF
        rank = int(dims["rank"])
        eivals = [math.sin(h * 0.001 * i) for i in range(max(1, min(3, rank)))]
        return {
            "projected_rank": dims["rank"],
            "structure": dims["structure"],
            "eigenvalues": eivals,
            "determinant": math.cos(h * 0.0001),
            "trace": (math.tan(h * 0.00001) if (h % 100) else 0.0),
        }
    
    def _analyze(self, data: Any) -> Dict[str, Any]:
        s = str(data)
        return {
            "rank": min(10, len(s) // 50),
            "structure": "sparse" if len(set(s)) < 20 else "dense"
        }

class JuliaSymbolEngine:
    def analyze(self, data: Any) -> Dict[str, Any]:
        coeffs = self._coeffs(data)
        return {
            "chebyshev_polynomial": self._poly(coeffs),
            "coefficients": coeffs,
            "derivatives": self._derivs(coeffs),
            "critical_points": self._crit(coeffs),
        }
    
    def _coeffs(self, data: Any) -> List[float]:
        s = str(data)
        return [math.sin(hash(s[i:i+4]) % 100) if i < len(s) else 0.0 for i in range(5)]
    
    def _poly(self, coeffs: List[float]) -> str:
        return f"{coeffs[0]:.3f} + {coeffs[1]:.3f}x + {coeffs[2]:.3f}x²"
    
    def _derivs(self, coeffs: List[float]) -> List[float]:
        return [coeffs[1], 2*coeffs[2], 0.0, 0.0, 0.0]
    
    def _crit(self, coeffs: List[float]) -> List[float]:
        if abs(coeffs[2]) > 1e-6:
            return [-coeffs[1]/(2*coeffs[2])]
        return []

class ChoppyProcessor:
    def chunk(self, data: Any, chunk_size: int = 64, overlap: int = 16) -> Dict[str, Any]:
        s = str(data)
        step = max(1, chunk_size - overlap)
        std = [s[i:i + chunk_size] for i in range(0, len(s), step)]
        words = s.split(); wsize = max(1, chunk_size // 5)
        wchunks = [" ".join(words[i:i + wsize]) for i in range(0, len(words), wsize)]
        return {
            "standard": std,
            "semantic": wchunks,
            "fibonacci": self._fib(s),
            "statistics": {"total_length": len(s), "chunk_count": len(std), "average_chunk_size": len(s) / max(1, len(std))},
        }
    
    def _fib(self, s: str) -> List[str]:
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        chunks = []
        pos = 0
        for f in fib:
            if pos >= len(s):
                break
            chunks.append(s[pos:pos+f])
            pos += f
        return chunks

class EndpointCaster:
    def generate(self, data: Any) -> Dict[str, Any]:
        sig = hashlib.sha256(json.dumps(data, default=str, sort_keys=True).encode()).hexdigest()[:12]
        base = uuid.uuid4().hex[:6]
        return {
            "primary_endpoint": f"/api/v1/cast/{base}",
            "versioned_endpoints": [
                f"/api/v1/cast/{base}/reflect",
                f"/api/v1/cast/{base}/transform",
                f"/api/v1/cast/{base}/metadata",
                f"/api/v2/mirror/{sig}",
            ],
            "artifact_id": f"art-{uuid.uuid4().hex[:8]}",
            "metadata": {"content_type": self._ctype(data), "estimated_size": len(str(data)), "complexity": self._cpx(data)},
        }
    
    def _ctype(self, data: Any) -> str:
        s = str(data)
        if len(s) < 100:
            return "text/plain"
        if any(c in s for c in ['{', '[', '(']):
            return "application/json"
        return "text/plain"
    
    def _cpx(self, data: Any) -> float:
        s = str(data)
        return min(1.0, len(set(s)) / max(1, len(s)))

class CarryOnManager:
    def __init__(self, max_history: int = 200):
        self.memory: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.access: Dict[str, int] = {}
    
    def store(self, key: str, value: Any) -> None:
        self.memory[key] = value
        self.access[key] = time.time()
        self.history.append({"key": key, "value": str(value)[:100], "time": time.time()})
        if len(self.history) > self.max_history:
            self.history.pop(0)

class SemanticMapper:
    def __init__(self):
        self.semantic_networks = {
            "reflection": ["mirror", "echo", "reverberation", "contemplation", "introspection"],
            "transformation": ["metamorphosis", "mutation", "evolution", "adaptation", "transmutation"],
            "analysis": ["examination", "scrutiny", "dissection", "investigation", "exploration"],
            "synthesis": ["combination", "fusion", "amalgamation", "integration", "unification"],
        }
    
    def map(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        scores = {}
        for category, words in self.semantic_networks.items():
            score = sum(1 for word in words if word in text_lower)
            scores[category] = score / len(words)
        return scores

class LoveReflector:
    def infuse(self, data: Any) -> Dict[str, Any]:
        text = str(data)
        return {"poetic": self._poem(text), "emotional_resonance": self._emo(text), "love_quotient": self._lq(text), "harmony_index": self._hi(text)}
    
    def _poem(self, text: str) -> str:
        words = text.split()
        if len(words) < 3:
            return text
        return f"{words[0]} {words[1]} {words[-1]}"
    
    def _emo(self, text: str) -> float:
        emotional_words = ['love', 'hate', 'joy', 'sad', 'happy', 'angry', 'peace', 'war']
        return sum(1 for word in emotional_words if word in text.lower()) / len(emotional_words)
    
    def _lq(self, text: str) -> float:
        return min(1.0, text.count('love') / max(1, len(text.split())))
    
    def _hi(self, text: str) -> float:
        return 0.7  # Placeholder

class FractalResonator:
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
    
    def cascade(self, data: Any) -> Dict[str, Any]:
        s = str(data)
        layers = []
        for depth in range(1, min(self.max_depth + 1, len(s) // 10 + 1)):
            chunk = s[:depth * 10]
            entropy = EntropyAnalyzer().measure(chunk)
            layers.append({"depth": depth, "entropy": entropy, "content": chunk})
        return {"layers": layers, "max_depth_reached": len(layers)}

class FeatureExtractor:
    def __init__(self, dim: int = 64, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram
    
    def extract(self, text: str) -> List[float]:
        s = text.lower()
        features = [0.0] * self.dim
        for i in range(len(s) - self.ngram + 1):
            ngram = s[i:i+self.ngram]
            idx = hash(ngram) % self.dim
            features[idx] += 1.0
        # Normalize
        total = sum(features)
        if total > 0:
            features = [f / total for f in features]
        return features

class NeuroSymbolicFusion:
    def __init__(self):
        self.w_neuro = 0.55
        self.w_symbol = 0.45
    
    def fuse(self, neuro_features: List[float], symbolic_metrics: Dict[str, float]) -> Dict[str, Any]:
        neuro_score = sum(neuro_features) / len(neuro_features) if neuro_features else 0.0
        symbol_score = sum(symbolic_metrics.values()) / len(symbolic_metrics) if symbolic_metrics else 0.0
        fused = self.w_neuro * neuro_score + self.w_symbol * symbol_score
        return {
            "neuro_score": neuro_score,
            "symbol_score": symbol_score,
            "fused_score": fused,
            "decision": "transmit" if fused > 0.5 else "hold"
        }

class DecisionLogger:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
    
    def log(self, event: Dict[str, Any]) -> None:
        self.events.append({**event, "timestamp": time.time()})

class ReflectiveDB:
    def __init__(self, path: str = "reflective_db.json"):
        self.path = path
        self._data: List[Dict[str, Any]] = []
        self._load()
    
    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self._data = json.load(f)
            except:
                self._data = []
    
    def save(self) -> None:
        with open(self.path, 'w') as f:
            json.dump(self._data, f, indent=2)
    
    def add_record(self, record: Dict[str, Any]) -> None:
        self._data.append(record)
        self.save()

class RLAgent:
    def __init__(self, actions: List[str] = None, eps: float = 0.1):
        self.actions = actions or ["bpsk", "qpsk", "ofdm"]
        self.eps = eps
        self.q: Dict[Tuple[int, int, int], Dict[str, Dict[str, float]]] = {}
    
    def choose_action(self, state: Tuple[int, int, int]) -> str:
        if np.random.random() < self.eps or state not in self.q:
            return np.random.choice(self.actions)
        action_values = {a: self.q[state][a]["q"] for a in self.actions if a in self.q[state]}
        return max(action_values.items(), key=lambda x: x[1])[0] if action_values else np.random.choice(self.actions)
    
    def update(self, state: Tuple[int, int, int], action: str, reward: float) -> None:
        if state not in self.q:
            self.q[state] = {a: {"q": 0.0, "n": 0} for a in self.actions}
        self.q[state][action]["n"] += 1
        n = self.q[state][action]["n"]
        old_q = self.q[state][action]["q"]
        self.q[state][action]["q"] = old_q + (reward - old_q) / n

class MirrorCastEngine:
    def __init__(self):
        self.entropy = EntropyAnalyzer()
        self.reflector = DianneReflector()
        self.matrix = MatrixTransformer()
        self.symbols = JuliaSymbolEngine()
        self.choppy = ChoppyProcessor()
        self.endpoints = EndpointCaster()
        self.memory = CarryOnManager()
        self.semantic = SemanticMapper()
        self.love = LoveReflector()
        self.fractal = FractalResonator()
    
    def cast(self, data: Any) -> Dict[str, Any]:
        return {
            "entropy": self.entropy.measure(data),
            "reflection": self.reflector.reflect(data),
            "matrix": self.matrix.project(data),
            "symbolic": self.symbols.analyze(data),
            "chunks": self.choppy.chunk(data),
            "endpoints": self.endpoints.generate(data),
            "semantic": self.semantic.map(str(data)),
            "love": self.love.infuse(data),
            "fractal": self.fractal.cascade(data),
            "timestamp": time.time()
        }

class AdaptiveLinkPlanner:
    def __init__(self, db_path: str = "reflective_db.json"):
        self.extractor = FeatureExtractor()
        self.fusion = NeuroSymbolicFusion()
        self.agent = RLAgent(actions=["bpsk", "qpsk", "ofdm"], eps=0.1)
        self.db = ReflectiveDB(db_path)
        self.log = DecisionLogger()
    
    def plan(self, text: str, current_config: ModConfig, analysis: Dict[str, Any], snr_db: float) -> Tuple[ModConfig, str]:
        # Extract features
        features = self.extractor.extract(text)
        
        # Create symbolic metrics from analysis
        symbolic_metrics = {
            "entropy": analysis.get("entropy", 0.0),
            "complexity": analysis.get("endpoints", {}).get("metadata", {}).get("complexity", 0.5),
            "semantic_density": sum(analysis.get("semantic", {}).values()) / max(1, len(analysis.get("semantic", {})))
        }
        
        # Fuse neuro-symbolic
        fusion_result = self.fusion.fuse(features, symbolic_metrics)
        
        # Create state representation
        entropy_bin = min(9, int(analysis.get("entropy", 0.0) * 10))
        complexity_bin = min(9, int(symbolic_metrics["complexity"] * 10))
        snr_bin = min(9, int(snr_db / 10))
        state = (entropy_bin, complexity_bin, snr_bin)
        
        # Choose action
        action = self.agent.choose_action(state)
        
        # Update config based on action
        new_config = ModConfig(
            sample_rate=current_config.sample_rate,
            symbol_rate=current_config.symbol_rate,
            amplitude=current_config.amplitude,
            f0=current_config.f0,
            f1=current_config.f1,
            fc=current_config.fc,
            clip=current_config.clip,
            ofdm_subc=current_config.ofdm_subc,
            cp_len=current_config.cp_len,
            dsss_chip_rate=current_config.dsss_chip_rate
        )
        
        # Adjust based on action
        if action == "bpsk":
            new_config.symbol_rate = 1200
        elif action == "qpsk":
            new_config.symbol_rate = 2400
        elif action == "ofdm":
            new_config.symbol_rate = 4800
        
        explanation = f"Neuro-symbolic score: {fusion_result['fused_score']:.3f}, chose {action.upper()} for state {state}"
        
        self.log.log({
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:8],
            "state": state,
            "action": action,
            "fusion_result": fusion_result,
            "explanation": explanation
        })
        
        return new_config, explanation
    
    def reward_and_record(self, text: str, config: ModConfig, explanation: str, success: bool, snr_db: float, harmony: float, tx_len: int, rx_len: int) -> None:
        # Simple reward: +1 for success, -1 for failure
        reward = 1.0 if success else -1.0
        # Adjust based on harmony (higher is better)
        reward *= harmony
        
        # Get current state
        analysis = MirrorCastEngine().cast(text)
        entropy_bin = min(9, int(analysis.get("entropy", 0.0) * 10))
        complexity_bin = min(9, int(analysis.get("endpoints", {}).get("metadata", {}).get("complexity", 0.5) * 10))
        snr_bin = min(9, int(snr_db / 10))
        state = (entropy_bin, complexity_bin, snr_bin)
        
        # Determine action from config
        if config.symbol_rate <= 1200:
            action = "bpsk"
        elif config.symbol_rate <= 2400:
            action = "qpsk"
        else:
            action = "ofdm"
        
        # Update Q-values
        self.agent.update(state, action, reward)
        
        # Record to DB
        self.db.add_record({
            "timestamp": time.time(),
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:8],
            "state": state,
            "action": action,
            "reward": reward,
            "success": success,
            "snr_db": snr_db,
            "harmony": harmony,
            "tx_len": tx_len,
            "rx_len": rx_len,
            "explanation": explanation
        })

# =========================================================
# Utilities
# =========================================================

def now_ms() -> int:
    return int(time.time() * 1000)

def crc32_bytes(data: bytes) -> bytes:
    return binascii.crc32(data).to_bytes(4, "big")

def crc16_ccitt(data: bytes) -> bytes:
    poly, crc = 0x1021, 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc.to_bytes(2, "big")

def to_bits(data: bytes) -> List[int]:
    return [(byte >> i) & 1 for byte in data for i in range(7, -1, -1)]

def from_bits(bits: Sequence[int]) -> bytes:
    if len(bits) % 8 != 0:
        bits = list(bits) + [0] * (8 - len(bits) % 8)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | (1 if b else 0)
        out.append(byte)
    return bytes(out)

def chunk_bits(bits: Sequence[int], n: int) -> List[List[int]]:
    return [list(bits[i:i+n]) for i in range(0, len(bits), n)]

def safe_json(obj: Any) -> str:
    def enc(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, complex):
            return {"real": float(x.real), "imag": float(x.imag)}
        if isinstance(x, datetime):
            return x.isoformat()
        return str(x)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=enc)

def write_wav_mono(path: Path, signal: np.ndarray, sample_rate: int):
    import wave
    sig = np.clip(signal, -1.0, 1.0)
    pcm = (sig * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())

def write_iq_f32(path: Path, iq: np.ndarray):
    if iq.ndim != 1 or not np.iscomplexobj(iq):
        raise ValueError("iq must be 1-D complex array")
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    path.write_bytes(interleaved.tobytes())

def plot_wave_and_spectrum(path_png: Path, x: np.ndarray, sr: int, title: str):
    if not HAS_MPL: 
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,5))
    t = np.arange(len(x))/sr
    ax1.plot(t[:min(len(t), 0.05*sr)], x[:min(len(x), int(0.05*sr))])
    ax1.set_title(f"{title} (first 50ms)")
    ax1.set_xlabel("s"); ax1.set_ylabel("amplitude")
    spec = np.abs(rfft(x)) + 1e-12
    freqs = rfftfreq(len(x), 1.0/sr)
    ax2.semilogy(freqs, spec/spec.max())
    ax2.set_xlim(0, min(8000, sr//2)); ax2.set_xlabel("Hz"); ax2.set_ylabel("norm |X(f)|")
    plt.tight_layout(); fig.savefig(path_png); plt.close(fig)

def play_audio(x: np.ndarray, sr: int):
    if not HAS_AUDIO:
        log.warning("sounddevice not installed; cannot play audio")
        return
    sd.play(x, sr); sd.wait()

# =========================================================
# FEC (simple Hamming 7,4; heavy codes are stubs)
# =========================================================

def hamming74_encode(data_bits: List[int]) -> List[int]:
    if len(data_bits) % 4 != 0:
        data_bits = data_bits + [0] * (4 - len(data_bits) % 4)
    out = []
    for i in range(0, len(data_bits), 4):
        d0, d1, d2, d3 = data_bits[i:i+4]
        p1 = d0 ^ d1 ^ d3
        p2 = d0 ^ d2 ^ d3
        p3 = d1 ^ d2 ^ d3
        out += [p1, p2, d0, p3, d1, d2, d3]
    return out

def fec_encode(bits: List[int], scheme: FEC) -> List[int]:
    if scheme == FEC.NONE:
        return list(bits)
    if scheme == FEC.HAMMING74:
        return hamming74_encode(bits)
    if scheme in (FEC.REED_SOLOMON, FEC.LDPC, FEC.TURBO):
        raise NotImplementedError(f"{scheme.name} encoding not implemented in this minimal build")
    raise ValueError("Unknown FEC")

# =========================================================
# Framing / Security / Watermark
# =========================================================

@dataclass
class SecurityConfig:
    password: Optional[str] = None           # AES-GCM if provided
    watermark: Optional[str] = None          # prepended SHA256[0:8]
    hmac_key: Optional[str] = None           # HMAC-SHA256 appended

def aes_gcm_encrypt(plaintext: bytes, password: str) -> bytes:
    if not HAS_CRYPTO:
        raise RuntimeError("pycryptodome required for encryption")
    salt = get_random_bytes(16)
    key = PBKDF2(password, salt, dkLen=32, count=200_000)
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(plaintext)
    return b"AGCM" + salt + nonce + tag + ct

def apply_hmac(data: bytes, hkey: str) -> bytes:
    import hmac
    key = hashlib.sha256(hkey.encode("utf-8")).digest()
    mac = hmac.new(key, data, hashlib.sha256).digest()
    return data + b"HMAC" + mac

def add_watermark(data: bytes, wm: str) -> bytes:
    return hashlib.sha256(wm.encode("utf-8")).digest()[:8] + data

def frame_payload(payload: bytes, fcfg: FrameConfig) -> bytes:
    header = struct.pack(">BBI", 0xA5, fcfg.version, now_ms() & 0xFFFFFFFF)
    core = header + payload
    tail = b""
    if fcfg.use_crc32:
        tail += crc32_bytes(core)
    if fcfg.use_crc16:
        tail += crc16_ccitt(core)
    return fcfg.preamble + core + tail

def encode_text(
    text: str,
    fcfg: FrameConfig,
    sec: SecurityConfig,
    fec_scheme: FEC,
) -> List[int]:
    data = text.encode("utf-8")
    if sec.watermark:
        data = add_watermark(data, sec.watermark)
    if sec.password:
        data = aes_gcm_encrypt(data, sec.password)
    framed = frame_payload(data, fcfg)
    if sec.hmac_key:
        framed = apply_hmac(framed, sec.hmac_key)
    bits = to_bits(framed)
    bits = fec_encode(bits, fec_scheme)
    return bits

# =========================================================
# Modulators (audio & IQ)
# =========================================================

class Modulators:
    @staticmethod
    def bfsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        sr, rb = cfg.sample_rate, cfg.symbol_rate
        spb = int(sr / rb)
        t = np.arange(spb) / sr
        s = []
        a = cfg.amplitude
        for b in bits:
            f = cfg.f1 if b else cfg.f0
            s.append(a * np.sin(2*np.pi*f*t))
        y = np.concatenate(s) if s else np.zeros(0, dtype=np.float64)
        return np.clip(y, -1, 1).astype(np.float32) if cfg.clip else y.astype(np.float32)

    @staticmethod
    def bpsK(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        t = np.arange(spb) / sr
        a = cfg.amplitude
        audio_blocks, iq_blocks = [], []
        for b in bits:
            phase = 0.0 if b else np.pi
            audio_blocks.append(a * np.sin(2*np.pi*fc*t + phase))
            iq_blocks.append(a * (np.cos(phase) + 1j*np.sin(phase)) * np.ones_like(t, dtype=np.complex64))
        audio = np.concatenate(audio_blocks) if audio_blocks else np.zeros(0, dtype=np.float64)
        iq = np.concatenate(iq_blocks) if iq_blocks else np.zeros(0, dtype=np.complex64)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32), iq

    @staticmethod
    def qpsK(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        # Gray map: 00->(1+1j), 01->(-1+1j), 11->(-1-1j), 10->(1-1j)
        pairs = chunk_bits(bits, 2)
        syms = []
        for p in pairs:
            b0, b1 = (p + [0,0])[:2]
            if (b0, b1) == (0,0): s = 1 + 1j
            elif (b0, b1) == (0,1): s = -1 + 1j
            elif (b0, b1) == (1,1): s = -1 - 1j
            else: s = 1 - 1j
            syms.append(s / math.sqrt(2))  # unit energy
        return Modulators._psk_qam_to_audio_iq(np.array(syms, dtype=np.complex64), cfg)

    @staticmethod
    def qam16(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        quads = chunk_bits(bits, 4)
        def map2(b0,b1):
            # Gray 2-bit to {-3,-1,1,3}
            val = (b0<<1) | b1
            return [-3,-1,1,3][val]
        syms = []
        for q in quads:
            b0,b1,b2,b3 = (q+[0,0,0,0])[:4]
            I = map2(b0,b1); Q = map2(b2,b3)
            syms.append((I + 1j*Q)/math.sqrt(10)) # unit average power
        return Modulators._psk_qam_to_audio_iq(np.array(syms, dtype=np.complex64), cfg)

    @staticmethod
    def _psk_qam_to_audio_iq(syms: np.ndarray, cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        a = cfg.amplitude
        # Upsample each symbol to 'spb' samples (rectangular pulse)
        i = np.repeat(syms.real.astype(np.float32), spb)
        q = np.repeat(syms.imag.astype(np.float32), spb)
        t = np.arange(len(i)) / sr
        audio = a * (i*np.cos(2*np.pi*fc*t) - q*np.sin(2*np.pi*fc*t))
        iq = (a * i) + 1j*(a * q)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.ast(np.float32), iq.astype(np.complex64)

    @staticmethod
    def afsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        return Modulators.bfsK(bits, cfg)

    @staticmethod
    def dsss_bpsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        # Very simple DSSS: chip with PN sequence at cfg.dsss_chip_rate
        pn = np.array([1, -1, 1, 1, -1, 1, -1, -1], dtype=np.float32)  # toy PN8
        sr = cfg.sample_rate
        chips_per_symbol = max(1, int(cfg.dsss_chip_rate / cfg.symbol_rate))
        spb = int(sr / (cfg.dsss_chip_rate))
        base = []
        for b in bits:
            bit_val = 1.0 if b else -1.0
            ch = bit_val * pn
            ch = np.repeat(ch, spb)
            base.append(ch)
        baseband = np.concatenate(base) if base else np.zeros(0, dtype=np.float32)
        # Upconvert to audio carrier
        t = np.arange(len(baseband))/sr
        audio = cfg.amplitude * baseband * np.sin(2*np.pi*cfg.fc*t)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32)

    @staticmethod
    def ofdm(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        # Toy OFDM: QPSK mapping across N subcarriers, IFFT, add cyclic prefix
        N = cfg.ofdm_subc
        spb_sym = int(cfg.sample_rate / cfg.symbol_rate)  # samples per OFDM symbol (approx shaping)
        chunks = chunk_bits(bits, 2*N)
        a = cfg.amplitude
        wave = []
        iq = []
        for ch in chunks:
            # map 2 bits -> QPSK symbol
            qsyms = []
            pairs = chunk_bits(ch, 2)
            for p in pairs:
                b0,b1 = (p+[0,0])[:2]
                if (b0,b1)==(0,0): s = 1+1j
                elif (b0,b1)==(0,1): s = -1+1j
                elif (b0,b1)==(1,1): s = -1-1j
                else: s = 1-1j
                qsyms.append(s/math.sqrt(2))
            # pad to N
            if len(qsyms) < N:
                qsyms += [0j]*(N-len(qsyms))
            Xk = np.array(qsyms, dtype=np.complex64)
            xt = np.fft.ifft(Xk)  # time domain symbol (complex)
            # cyclic prefix
            cp = xt[-cfg.cp_len:]
            sym = np.concatenate([cp, xt])
            # stretch to samples-per-symbol for audio mixing
            reps = max(1, int(spb_sym/len(sym)))
            sym_up = np.repeat(sym, reps)
            # audio upconvert
            t = np.arange(len(sym_up))/cfg.sample_rate
            audio = a*(sym_up.real*np.cos(2*np.pi*cfg.fc*t) - sym_up.imag*np.sin(2*np.pi*cfg.fc*t))
            wave.append(audio.astype(np.float32))
            iq.append((a*sym_up).astype(np.complex64))
        audio = np.concatenate(wave) if wave else np.zeros(0, dtype=np.float32)
        iqc = np.concatenate(iq) if iq else np.zeros(0, dtype=np.complex64)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio, iqc

# =========================================================
# LLM backends (Local final inference; Remote resource-only)
# =========================================================

class BaseLLM:
    def generate(self, prompt: str, **kwargs) -> str: raise NotImplementedError

class LocalLLM(BaseLLM):
    def __init__(self, configs: List[HTTPConfig]):
        if requests is None:
            raise RuntimeError("LocalLLM requires 'requests' (pip install requests)")
        self.configs = configs
        self.idx = 0

    def generate(self, prompt: str, **kwargs) -> str:
        last = None
        for _ in range(len(self.configs)):
            cfg = self.configs[self.idx]
            try:
                out = self._call(cfg, prompt, **kwargs)
                return out
            except Exception as e:
                last = e
                self.idx = (self.idx + 1) % len(self.configs)
        raise last or RuntimeError("All local LLM configs failed")

    def _post(self, cfg: HTTPConfig, url: str, headers: dict, body: dict) -> dict:
        s = requests.Session()
        for attempt in range(cfg.max_retries):
            try:
                r = s.post(url, headers=headers, json=body, timeout=cfg.timeout, verify=cfg.verify_ssl)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt < cfg.max_retries-1:
                    time.sleep(cfg.retry_delay*(2**attempt))
                else:
                    raise

    def _call(self, cfg: HTTPConfig, prompt: str, **kwargs) -> str:
        mode = cfg.mode
        if mode == "openai-chat":
            url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: headers["Authorization"] = f"Bearer {cfg.api_key}"
            body = {
                "model": cfg.model or "gpt-4o-mini",
                "messages": [{"role":"user","content":prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["message"]["content"]
        if mode == "openai-completions":
            url = f"{cfg.base_url.rstrip('/')}/v1/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: headers["Authorization"] = f"Bearer {cfg.api_key}"
            body = {
                "model": cfg.model or "gpt-3.5-turbo-instruct",
                "prompt": prompt,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["text"]
        if mode == "llama-cpp":
            url = f"{cfg.base_url.rstrip('/')}/completion"
            body = {"prompt": prompt, "temperature": kwargs.get("temperature",0.7), "n_predict": kwargs.get("max_tokens",512)}
            data = self._post(cfg, url, {}, body)
            if "content" in data: return data["content"]
            if "choices" in data and data["choices"]: return data["choices"][0].get("text","")
            return data.get("text","")
        if mode == "textgen-webui":
            url = f"{cfg.base_url.rstrip('/')}/api/v1/generate"
            body = {"prompt": prompt, "max_new_tokens": kwargs.get("max_tokens",512), "temperature": kwargs.get("temperature",0.7)}
            data = self._post(cfg, url, {}, body)
            return data.get("results",[{}])[0].get("text","")
        raise ValueError(f"Unsupported mode: {mode}")

class ResourceLLM(BaseLLM):
    def __init__(self, cfg: Optional[HTTPConfig] = None):
        self.cfg = cfg

    def generate(self, prompt: str, **kwargs) -> str:
        # Constrained to resources-only summarization
        if self.cfg is None or requests is None:
            return LocalSummarizer().summarize(prompt)
        url = f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type":"application/json"}
        if self.cfg.api_key: headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        system = ("You are a constrained assistant. ONLY summarize/structure the provided INPUT RESOURCES. "
                  "Do not add external knowledge.")
        body = {
            "model": self.cfg.model or "gpt-4o-mini",
            "messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
            "temperature": kwargs.get("temperature",0.2),
            "max_tokens": kwargs.get("max_tokens",512),
        }
        s = requests.Session()
        r = s.post(url, headers=headers, json=body, timeout=self.cfg.timeout, verify=self.cfg.verify_ssl)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

class LocalSummarizer:
    def __init__(self):
        self.stop = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with","by","is","are",
            "was","were","be","been","being","have","has","had","do","does","did","will","would",
            "could","should","from","that","this","it","as"
        }
    def summarize(self, text: str) -> str:
        txt = " ".join(text.split())
        if not txt: return "No content to summarize."
        sents = [s.strip() for s in txt.replace("?",".").replace("!",".").split(".") if s.strip()]
        if not sents: return txt[:300] + ("..." if len(txt)>300 else "")
        # score sentences by length + term frequency (very light heuristic)
        words = [w.lower().strip(",;:()[]") for w in txt.split()]
        freq: Dict[str,int] = {}
        for w in words:
            if w and w not in self.stop: freq[w] = freq.get(w,0)+1
        scored = []
        for s in sents:
            sw = [w.lower().strip(",;:()[]") for w in s.split()]
            score = len(s) * 0.1 + sum(freq.get(w,0) for w in sw)
            scored.append((s, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        keep = [s for s,_ in scored[: min(6,len(scored))]]
        keep.sort(key=lambda k: sents.index(k))
        out = " ".join(keep)
        return out[:800] + ("..." if len(out)>800 else "")

# =========================================================
# Orchestrator
# =========================================================

class DualLLMOrchestrator:
    def __init__(self, local: LocalLLM, resource: ResourceLLM, settings: OrchestratorSettings):
        self.local, self.resource, self.set = local, resource, settings

    def _load_resources(self, paths: List[str], inline: List[str]) -> str:
        parts = []
        for p in paths:
            pa = Path(p)
            if pa.exists() and pa.is_file():
                try:
                    parts.append(pa.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    parts.append(f"[[UNREADABLE_FILE:{pa.name}]]")
            else:
                parts.append(f"[[MISSING_FILE:{pa}]]")
        parts += [str(x) for x in inline]
        blob = "\n\n".join(parts)
        return blob[: self.set.max_context_chars]

    def compose(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Tuple[str,str]:
        res_text = self._load_resources(resource_paths, inline_resources)
        res_summary = self.resource.generate(
            f"INPUT RESOURCES:\n{res_text}\n\nTASK: Summarize/structure ONLY the content above.",
            temperature=0.2, max_tokens=self.set.max_tokens
        )
        final_prompt = (
            "You are a LOCAL expert system. Use ONLY the structured summary below; do not invent facts.\n\n"
            f"=== STRUCTURED SUMMARY ===\n{res_summary}\n\n"
            f"=== USER PROMPT ===\n{user_prompt}\n\n"
            f"STYLE: {self.set.style}. Be clear and directly actionable."
        )
        return final_prompt, res_summary

    def run(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Dict[str,str]:
        fp, summary = self.compose(user_prompt, resource_paths, inline_resources)
        ans = self.local.generate(fp, temperature=self.set.temperature, max_tokens=self.set.max_tokens)
        return {"summary": summary, "final": ans, "prompt": fp}

# =========================================================
# End-to-end casting
# =========================================================

@dataclass
class OutputPaths:
    wav: Optional[Path] = None
    iq: Optional[Path] = None
    meta: Optional[Path] = None
    png: Optional[Path] = None

def bits_to_signals(bits: List[int], scheme: ModulationScheme, mcfg: ModConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if scheme == ModulationScheme.BFSK:
        return Modulators.bfsK(bits, mcfg), None
    if scheme == ModulationScheme.AFSK:
        return Modulators.afsK(bits, mcfg), None
    if scheme == ModulationScheme.BPSK:
        return Modulators.bpsK(bits, mcfg)
    if scheme == ModulationScheme.QPSK:
        return Modulators.qpsK(bits, mcfg)
    if scheme == ModulationScheme.QAM16:
        return Modulators.qam16(bits, mcfg)
    if scheme == ModulationScheme.OFDM:
        return Modulators.ofdm(bits, mcfg)
    if scheme == ModulationScheme.DSSS_BPSK:
        return Modulators.dsss_bpsK(bits, mcfg), None
    raise ValueError("Unknown modulation scheme")

def cast_to_files(
    text: str,
    outdir: Path,
    scheme: ModulationScheme,
    mcfg: ModConfig,
    want_wav: bool,
    want_iq: bool,
    title: str = "WaveCaster"
) -> OutputPaths:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    base = outdir / f"cast_{scheme.name.lower()}_{ts}"
    # Minimal frame (no FEC/security here; caller handles)
    fcfg = FrameConfig()
    bits = to_bits(frame_payload(text.encode("utf-8"), fcfg))
    audio, iq = bits_to_signals(bits, scheme, mcfg)
    paths = OutputPaths()
    if want_wav and audio is not None and len(audio)>0:
        paths.wav = base.with_suffix(".wav"); write_wav_mono(paths.wav, audio, mcfg.sample_rate)
    if want_iq:
        if iq is None and audio is not None:
            # make a naive hilbert to IQ for convenience
            try:
                q = np.imag(sp_signal.hilbert(audio))
                iq = audio.astype(np.float32) + 1j*q.astype(np.float32)
            except Exception:
                iq = (audio.astype(np.float32) + 1j*np.zeros_like(audio, dtype=np.float32))
        if iq is not None:
            paths.iq = base.with_suffix(".iqf32"); write_iq_f32(paths.iq, iq)
    # Visualization
    if audio is not None and len(audio)>0 and HAS_MPL:
        paths.png = base.with_suffix(".png"); plot_wave_and_spectrum(paths.png, audio, mcfg.sample_rate, title)
    # Meta
    meta = {
        "timestamp": ts, "scheme": scheme.name, "sample_rate": mcfg.sample_rate,
        "symbol_rate": mcfg.symbol_rate, "framesec": len(audio)/mcfg.sample_rate if audio is not None else 0,
    }
    paths.meta = base.with_suffix(".json")
    paths.meta.write_text(safe_json(meta), encoding="utf-8")
    return paths

def full_cast_and_save(
    text: str,
    outdir: Path,
    scheme: ModulationScheme,
    mcfg: ModConfig,
    fcfg: FrameConfig,
    sec: SecurityConfig,
    fec_scheme: FEC,
    want_wav: bool,
    want_iq: bool,
    title: str = "WaveCaster"
) -> OutputPaths:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    base = outdir / f"cast_{scheme.name.lower()}_{ts}"
    bits = encode_text(text, fcfg, sec, fec_scheme)
    audio, iq = bits_to_signals(bits, scheme, mcfg)
    paths = OutputPaths()
    if want_wav and audio is not None and len(audio)>0:
        paths.wav = base.with_suffix(".wav"); write_wav_mono(paths.wav, audio, mcfg.sample_rate)
    if want_iq:
        if iq is None and audio is not None:
            try:
                q = np.imag(sp_signal.hilbert(audio))
                iq = audio.astype(np.float32) + 1j*q.astype(np.float32)
            except Exception:
                iq = (audio.astype(np.float32) + 1j*np.zeros_like(audio, dtype=np.float32))
        if iq is not None:
            paths.iq = base.with_suffix(".iqf32"); write_iq_f32(paths.iq, iq)
    if audio is not None and len(audio)>0 and HAS_MPL:
        paths.png = base.with_suffix(".png"); plot_wave_and_spectrum(paths.png, audio, mcfg.sample_rate, title)
    meta = {
        "timestamp": ts, "scheme": scheme.name, "sample_rate": mcfg.sample_rate,
        "symbol_rate": mcfg.symbol_rate, "framesec": len(audio)/mcfg.sample_rate if audio is not None else 0,
        "fec": fec_scheme.name, "encrypted": bool(sec.password), "watermark": bool(sec.watermark),
        "hmac": bool(sec.hmac_key),
    }
    paths.meta = base.with_suffix(".json"); paths.meta.write_text(safe_json(meta), encoding="utf-8")
    return paths

# =========================================================
# CLI
# =========================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dual_llm_wavecaster_enhanced", description="Two-LLM orchestration → modulated waveform")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_mod_args(sp):
        sp.add_argument("--scheme", choices=[s.name.lower() for s in ModulationScheme], default="bfsk")
        sp.add_argument("--sample-rate", type=int, default=48000)
        sp.add_argument("--symbol-rate", type=int, default=1200)
        sp.add_argument("--amplitude", type=float, default=0.7)
        sp.add_argument("--f0", type=float, default=1200.0)
        sp.add_argument("--f1", type=float, default=2200.0)
        sp.add_argument("--fc", type=float, default=1800.0)
        sp.add_argument("--no-clip", action="store_true")
        sp.add_argument("--outdir", type=str, default="casts")
        sp.add_argument("--wav", action="store_true")
        sp.add_argument("--iq", action="store_true")
        sp.add_argument("--play", action="store_true", help="Play audio to soundcard (if available)")

        # OFDM / DSSS
        sp.add_argument("--ofdm-subc", type=int, default=64)
        sp.add_argument("--cp-len", type=int, default=16)
        sp.add_argument("--dsss-chip-rate", type=int, default=4800)

    # cast: 2-LLM orchestration then modulate
    sp_cast = sub.add_parser("cast", help="Compose via dual LLMs then modulate")
    sp_cast.add_argument("--prompt", type=str, required=True)
    sp_cast.add_argument("--resource-file", nargs="*", default=[])
    sp_cast.add_argument("--resource-text", nargs="*", default=[])
    # Local LLM
    sp_cast.add_argument("--local-url", type=str, default="http://127.0.0.1:8080")
    sp_cast.add_argument("--local-mode", choices=["openai-chat","openai-completions","llama-cpp","textgen-webui"], default="llama-cpp")
    sp_cast.add_argument("--local-model", type=str, default="local-gguf")
    sp_cast.add_argument("--local-key", type=str, default=None)
    # Remote Resource LLM
    sp_cast.add_argument("--remote-url", type=str, default=None)
    sp_cast.add_argument("--remote-model", type=str, default="gpt-4o-mini")
    sp_cast.add_argument("--remote-key", type=str, default=None)
    # Orchestration params
    sp_cast.add_argument("--style", type=str, default="concise")
    sp_cast.add_argument("--max-tokens", type=int, default=512)
    sp_cast.add_argument("--temperature", type=float, default=0.7)
    # Security / FEC
    sp_cast.add_argument("--password", type=str, default=None)
    sp_cast.add_argument("--watermark", type=str, default=None)
    sp_cast.add_argument("--hmac-key", type=str, default=None)
    sp_cast.add_argument("--fec", choices=[f.name.lower() for f in FEC], default="hamming74")
    # Neuro-symbolic adaptive
    sp_cast.add_argument("--adaptive", action="store_true", help="Use neuro-symbolic adaptive link planning")
    sp_cast.add_argument("--db-path", type=str, default="reflective_db.json", help="Path for reflective database")
    sp_cast.add_argument("--snr-db", type=float, default=30.0, help="Estimated SNR in dB for adaptive planning")
    add_mod_args(sp_cast)

    # modulate: direct text to waveform
    sp_mod = sub.add_parser("modulate", help="Modulate provided text directly")
    sp_mod.add_argument("--text", type=str, required=True)
    sp_mod.add_argument("--password", type=str, default=None)
    sp_mod.add_argument("--watermark", type=str, default=None)
    sp_mod.add_argument("--hmac-key", type=str, default=None)
    sp_mod.add_argument("--fec", choices=[f.name.lower() for f in FEC], default="none")
    add_mod_args(sp_mod)

    # visualize existing WAV
    sp_vis = sub.add_parser("visualize", help="Plot waveform + spectrum from WAV")
    sp_vis.add_argument("--wav", type=str, required=True)
    sp_vis.add_argument("--out", type=str, default=None)

    # analyze: print basic metrics
    sp_an = sub.add_parser("analyze", help="Basic audio metrics of WAV")
    sp_an.add_argument("--wav", type=str, required=True)

    # neuro-symbolic analysis
    sp_ns = sub.add_parser("neuro-symbolic", help="Neuro-symbolic analysis of text")
    sp_ns.add_argument("--text", type=str, required=True)
    sp_ns.add_argument("--plot", action="store_true", help="Generate plots")

    # adaptive learning
    sp_learn = sub.add_parser("learn", help="Run adaptive learning episodes")
    sp_learn.add_argument("--text", type=str, required=True)
    sp_learn.add_argument("--episodes", type=int, default=10)
    sp_learn.add_argument("--snr-db", type=float, default=30.0)
    sp_learn.add_argument("--db-path", type=str, default="reflective_db.json")

    return p

def parse_scheme(s: str) -> ModulationScheme:
    return ModulationScheme[s.upper()]

def parse_fec(s: str) -> FEC:
    return FEC[s.upper()]

def make_modcfg(args: argparse.Namespace) -> ModConfig:
    return ModConfig(
        sample_rate=args.sample_rate, symbol_rate=args.symbol_rate, amplitude=args.amplitude,
        f0=args.f0, f1=args.f1, fc=args.fc, clip=not args.no_clip,
        ofdm_subc=getattr(args, "ofdm_subc", 64), cp_len=getattr(args,"cp_len",16),
        dsss_chip_rate=getattr(args,"dsss_chip_rate",4800),
    )

def cmd_cast(args: argparse.Namespace) -> int:
    # Build LLMs
    local = LocalLLM([HTTPConfig(
        base_url=args.local_url, model=args.local_model, mode=args.local_mode, api_key=args.local_key
    )])
    rcfg = HTTPConfig(base_url=args.remote_url, model=args.remote_model, api_key=args.remote_key) if args.remote_url else None
    resource = ResourceLLM(rcfg)
    orch = DualLLMOrchestrator(local, resource, OrchestratorSettings(
        temperature=args.temperature, max_tokens=args.max_tokens, style=args.style
    ))
    result = orch.run(args.prompt, args.resource_file, args.resource_text)
    
    # Neuro-symbolic analysis
    engine = MirrorCastEngine()
    analysis = engine.cast(result["final"])
    
    # Build pipeline
    mcfg = make_modcfg(args)
    fcfg = FrameConfig()
    sec = SecurityConfig(password=args.password, watermark=args.watermark, hmac_key=args.hmac_key)
    scheme = parse_scheme(args.scheme)
    fec_s = parse_fec(args.fec)
    
    # Adaptive link planning
    if args.adaptive:
        planner = AdaptiveLinkPlanner(db_path=args.db_path)
        mcfg, explanation = planner.plan(result["final"], mcfg, analysis, args.snr_db)
        result["adaptive_explanation"] = explanation
        result["adaptive_config"] = {
            "symbol_rate": mcfg.symbol_rate,
            "scheme": scheme.name
        }
    
    paths = full_cast_and_save(
        text=result["final"], outdir=Path(args.outdir), scheme=scheme, mcfg=mcfg, fcfg=fcfg,
        sec=sec, fec_scheme=fec_s, want_wav=args.wav or (not args.iq), want_iq=args.iq,
        title=f"{scheme.name} | DualLLM Wave"
    )
    
    if args.play and paths.wav and HAS_AUDIO:
        import soundfile as sf
        try:
            data, sr = sf.read(str(paths.wav), dtype="float32")
            play_audio(data, sr)
        except Exception:
            log.warning("Install soundfile for playback of saved WAV, or use --play with 'modulate'")
    
    result["files"] = {
        "wav": str(paths.wav) if paths.wav else None,
        "iq": str(paths.iq) if paths.iq else None,
        "meta": str(paths.meta) if paths.meta else None,
        "png": str(paths.png) if paths.png else None
    }
    
    print(safe_json(result))
    return 0

def cmd_modulate(args: argparse.Namespace) -> int:
    mcfg = make_modcfg(args)
    fcfg = FrameConfig()
    sec = SecurityConfig(password=args.password, watermark=args.watermark, hmac_key=args.hmac_key)
    scheme = parse_scheme(args.scheme)
    fec_s = parse_fec(args.fec)
    paths = full_cast_and_save(
        text=args.text, outdir=Path(args.outdir), scheme=scheme, mcfg=mcfg, fcfg=fcfg,
        sec=sec, fec_scheme=fec_s, want_wav=args.wav or (not args.iq), want_iq=args.iq,
        title=f"{scheme.name} | Direct Mod"
    )
    if args.play and paths.wav:
        try:
            import soundfile as sf
            data, sr = sf.read(str(paths.wav), dtype="float32"); play_audio(data, sr)
        except Exception:
            log.warning("Install soundfile for playback of saved WAV")
    print(safe_json({"files": {"wav": str(paths.wav) if paths.wav else None,
                               "iq": str(paths.iq) if paths.iq else None,
                               "meta": str(paths.meta) if paths.meta else None,
                               "png": str(paths.png) if paths.png else None}}))
    return 0

def cmd_visualize(args: argparse.Namespace) -> int:
    if not HAS_MPL:
        print("matplotlib is not installed.")
        return 1
    import wave
    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        s = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
    out = Path(args.out or (Path(args.wav).with_suffix(".png")))
    plot_wave_and_spectrum(out, s, sr, f"Visualize: {Path(args.wav).name}")
    print(safe_json({"png": str(out), "sample_rate": sr, "seconds": len(s)/sr}))
    return 0

def cmd_analyze(args: argparse.Namespace) -> int:
    import wave
    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate(); n = w.getnframes()
        s = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
    dur = len(s)/sr
    rms = float(np.sqrt(np.mean(s**2)))
    peak = float(np.max(np.abs(s)))
    spec = np.abs(rfft(s)); spec /= (spec.max()+1e-12)
    # simple SNR estimate
    snr = 10*np.log10(np.mean(s**2) / (np.var(s - np.mean(s)) + 1e-12))
    print(safe_json({"sample_rate": sr, "seconds": dur, "rms": rms, "peak": peak, "snr_db": float(snr)}))
    return 0

def cmd_neuro_symbolic(args: argparse.Namespace) -> int:
    engine = MirrorCastEngine()
    analysis = engine.cast(args.text)
    print(safe_json(analysis))
    
    if args.plot and HAS_MPL:
        # Plot fractal layers
        if "fractal" in analysis:
            layers = analysis["fractal"]["layers"]
            depths = [l["depth"] for l in layers]
            entropies = [l["entropy"] for l in layers]
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 2, 1)
            plt.plot(depths, entropies, 'o-')
            plt.title("Fractal Entropy vs Depth")
            plt.xlabel("Depth")
            plt.ylabel("Entropy")
            plt.grid(True)
        
        # Plot semantic mapping
        if "semantic" in analysis:
            plt.subplot(2, 2, 2)
            categories = list(analysis["semantic"].keys())
            scores = list(analysis["semantic"].values())
            plt.bar(categories, scores)
            plt.title("Semantic Mapping")
            plt.xticks(rotation=45)
            plt.grid(True)
        
        # Plot love infusion
        if "love" in analysis:
            plt.subplot(2, 2, 3)
            love_data = analysis["love"]
            metrics = ["emotional_resonance", "love_quotient", "harmony_index"]
            values = [love_data.get(m, 0) for m in metrics]
            plt.bar(metrics, values)
            plt.title("Love Infusion Metrics")
            plt.xticks(rotation=45)
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("neuro_symbolic_analysis.png")
        plt.show()
    
    return 0

def cmd_learn(args: argparse.Namespace) -> int:
    engine = MirrorCastEngine()
    planner = AdaptiveLinkPlanner(db_path=args.db_path)
    
    for ep in range(args.episodes):
        analysis = engine.cast(args.text)
        cfg, explanation = planner.plan(args.text, ModConfig(), analysis, args.snr_db)
        
        # Simulate transmission (placeholder - in real implementation, this would use actual modem)
        success = np.random.random() > 0.3  # 70% success rate for demo
        harmony = analysis.get("love", {}).get("harmony_index", 0.7)
        
        planner.reward_and_record(
            args.text, cfg, explanation, success, args.snr_db, 
            harmony, len(args.text), len(args.text) if success else 0
        )
        
        print(json.dumps({
            "episode": ep + 1,
            "config": {
                "symbol_rate": cfg.symbol_rate,
                "scheme": "bpsk" if cfg.symbol_rate <= 1200 else "qpsk" if cfg.symbol_rate <= 2400 else "ofdm"
            },
            "success": success,
            "explanation": explanation
        }, indent=2))
    
    print(f"Learning complete. DB saved to {args.db_path}")
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "cast": return cmd_cast(args)
    if args.cmd == "modulate": return cmd_modulate(args)
    if args.cmd == "visualize": return cmd_visualize(args)
    if args.cmd == "analyze": return cmd_analyze(args)
    if args.cmd == "neuro-symbolic": return cmd_neuro_symbolic(args)
    if args.cmd == "learn": return cmd_learn(args)
    p.print_help(); return 2

if __name__ == "__main__":
    raise SystemExit(main())
