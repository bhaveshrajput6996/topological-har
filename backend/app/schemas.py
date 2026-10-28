"""Pydantic schemas for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    torch_cuda: bool


class DatasetInfo(BaseModel):
    name: str
    kind: str
    sampling_rate: float
    channel_names: list[str]
    label_names: list[str]
    subjects: list[str]
    source_path: str
    notes: str


class DatasetLoadRequest(BaseModel):
    name: str
    root: str | None = None
    signal_mode: str | None = None
    subjects: list[str] | None = None
    channels: list[str] | None = None
    activity_filter: list[str] | None = None


class ExperimentRequest(BaseModel):
    config: dict[str, Any] = Field(..., description="inline experiment configuration")
    name: str | None = None


class PredictionRequest(BaseModel):
    run_id: str
    dataset: str | None = None
    subject: str | None = None
    window_index: int | None = None
    window_id: int | None = None
    data: list[list[float]] | None = None


class ReplayStartRequest(BaseModel):
    run_id: str | None = None
    dataset: str | None = None
    config: dict[str, Any] | None = None
    speed: float = 1.0
    subject: str | None = None
    max_windows: int | None = None


class ReplayControlRequest(BaseModel):
    session_id: str
    speed: float | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
