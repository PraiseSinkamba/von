"""FastAPI server for Von implementing TypeSafe-compatible HTTP endpoints."""

import os
from typing import Any, Dict, Optional, Union
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import VonEngine
from .types import Question
from .types import SystemOneResponse

app = FastAPI(
    title="Von Decision Server",
    description="Drop-in open source System One decision engine in homage to John von Neumann and Ludwig von Mises",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SystemOneRequest(BaseModel):
    model: str = Field(default="von-latest")
    state: Any = Field(..., description="State object, string, or array to evaluate")
    questions: Dict[str, Dict[str, Any]] = Field(..., description="Dict of question definitions")


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "von-decision-server",
        "version": "1.0.0",
        "engine": "von-option-marker",
        "homage": "John von Neumann & Ludwig von Mises",
    }


@app.get("/v1/models")
def list_models():
    model_entries = [
        {"name": "von-latest", "description": "Flagship Von System One Decision Model", "release_date": "2026-09-19"},
        {"name": "von-1.0.0", "description": "Von-1.0 Stable Release", "release_date": "2026-09-19"},
        {"name": "von-option-marker", "description": "Von Option-Marker Single-Pass Joint Attention Model", "release_date": "2026-09-20"},
        {"name": "jev-latest", "description": "TypeSafe Jev Compatibility Alias", "release_date": "2026-09-19"},
    ]
    data_entries = [
        {"id": m["name"], "object": "model", "owned_by": "von"}
        for m in model_entries
    ]
    return {
        "models": model_entries,
        "object": "list",
        "data": data_entries,
    }


@app.post("/v1/systemone", response_model=SystemOneResponse)
async def system_one_endpoint(
    req: SystemOneRequest,
    authorization: Optional[str] = Header(None),
):
    expected_key = os.environ.get("VON_API_KEY")
    if expected_key:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Bearer token")
        token = authorization.split("Bearer ", 1)[1].strip()
        if token != expected_key:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")

    try:
        engine = VonEngine.get_instance()
        questions: Dict[str, Union[Question, Dict[str, Any]]] = dict(req.questions)
        response = engine.evaluate(
            state=req.state,
            questions=questions,
            model=req.model,
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
