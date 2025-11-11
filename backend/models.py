from __future__ import annotations

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Enum, ForeignKey, Text
from sqlalchemy.orm import relationship, Mapped, mapped_column
from enum import Enum as PyEnum
from backend.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


class LeaseVersionStatus(PyEnum):
    uploaded = "uploaded"
    processing = "processing"
    processed = "processed"
    failed = "failed"


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_version_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProjectMember(Base):
    __tablename__ = "project_members"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    role: Mapped[str | None] = mapped_column(String, nullable=True)


class LeaseVersion(Base):
    __tablename__ = "lease_versions"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"))
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[LeaseVersionStatus] = mapped_column(Enum(LeaseVersionStatus), default=LeaseVersionStatus.uploaded)
    # Storage and processing
    file_url: Mapped[str | None] = mapped_column(String, nullable=True)
    doc_id: Mapped[str | None] = mapped_column(String, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project = relationship("Project", primaryjoin="LeaseVersion.project_id==Project.id", foreign_keys=[project_id])


class RiskScore(Base):
    __tablename__ = "risk_scores"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    lease_version_id: Mapped[str] = mapped_column(String, ForeignKey("lease_versions.id"))
    payload: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AbnormalityRecord(Base):
    __tablename__ = "abnormalities"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    lease_version_id: Mapped[str] = mapped_column(String, ForeignKey("lease_versions.id"))
    payload: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
