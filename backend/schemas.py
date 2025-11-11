from __future__ import annotations

from pydantic import BaseModel
from typing import Any


class UploadResponse(BaseModel):
    message: str
    doc_id: str | None = None
    risks: dict[str, Any] | None = None


class AskResponse(BaseModel):
    answer: str


class AbnormalitiesResponse(BaseModel):
    abnormalities: list[dict[str, str] | str]


class ClausesResponse(BaseModel):
    clauses: list[str]


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str | None = None
    current_version_id: str | None = None


class VersionCreate(BaseModel):
    label: str | None = None


class LeaseVersionOut(BaseModel):
    id: str
    project_id: str
    label: str | None = None
    status: str
    created_at: str | None = None


class VersionStatusResponse(BaseModel):
    id: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    stage: str | None = None
    progress: int | None = None


class RiskOut(BaseModel):
    payload: dict
    model: str | None = None
    created_at: str | None = None


class AbnormalitiesOut(BaseModel):
    payload: list
    model: str | None = None
    created_at: str | None = None

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class RiskAssessment(BaseModel):
    score: Optional[int] = Field(default=None)
    explanation: str


class UploadResponse(BaseModel):
    message: str
    doc_id: str
    risks: Dict[str, RiskAssessment]


class AskResponse(BaseModel):
    answer: str


class Abnormality(BaseModel):
    text: str
    impact: Literal["beneficial", "harmful", "neutral"]


class AbnormalitiesResponse(BaseModel):
    abnormalities: List[Abnormality]


class ClausesResponse(BaseModel):
    clauses: List[str]


# Project/Version schemas
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    current_version_id: Optional[str] = None


class VersionCreate(BaseModel):
    label: Optional[str] = None


class LeaseVersionOut(BaseModel):
    id: str
    project_id: str
    label: Optional[str] = None
    status: str
    created_at: Optional[str] = None


class VersionStatusResponse(BaseModel):
    id: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    stage: Optional[str] = None
    progress: Optional[int] = None


class RiskOut(BaseModel):
    payload: Dict[str, RiskAssessment]
    model: Optional[str] = None
    created_at: Optional[str] = None


class AbnormalitiesOut(BaseModel):
    payload: List[Abnormality]
    model: Optional[str] = None
    created_at: Optional[str] = None


class DiffChange(BaseModel):
    type: Literal["added", "removed", "modified"]
    clause_no: Optional[str] = None
    before: Optional[str] = None
    after: Optional[str] = None


class DiffResponse(BaseModel):
    base_version_id: str
    compare_version_id: str
    changes: List[DiffChange]


