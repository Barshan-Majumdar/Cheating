"""
Pydantic models for the AI Proctoring FastAPI server.

All request/response schemas are defined here for type safety,
automatic validation, and auto-generated API documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────

class ViolationType(str, Enum):
    """All known violation types in the proctoring system."""
    FACE_DISAPPEARED = "FACE_DISAPPEARED"
    FACE_REAPPEARED = "FACE_REAPPEARED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    OBJECT_DETECTED = "OBJECT_DETECTED"
    GAZE_AWAY = "GAZE_AWAY"
    MOUTH_MOVING = "MOUTH_MOVING"
    SPEECH_VIOLATION = "SPEECH_VIOLATION"
    VOICE_DETECTED = "VOICE_DETECTED"
    HAND_VIOLATION = "HAND_VIOLATION"
    AUDIO_DETECTED = "AUDIO_DETECTED"
    VIRTUAL_CAMERA_DETECTED = "VIRTUAL_CAMERA_DETECTED"
    MULTI_DISPLAY_DETECTED = "MULTI_DISPLAY_DETECTED"
    OBJECT_DETECTION_ERROR = "OBJECT_DETECTION_ERROR"
    UNKNOWN = "UNKNOWN"


class SessionStatus(str, Enum):
    """Proctoring session states."""
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


# ── Request Models ─────────────────────────────────────────────────────────

class AlertPayload(BaseModel):
    """Schema for alerts POSTed by the desktop proctoring client."""
    violation_type: str = Field(..., description="Type of violation detected (e.g. FACE_DISAPPEARED)")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of when the violation occurred")
    message: Optional[str] = Field(None, description="Human-readable description of the violation")
    metadata: Optional[dict] = Field(None, description="Additional context (image paths, frame data, etc.)")
    student_id: Optional[str] = Field(None, description="Student identifier")
    severity: Optional[int] = Field(None, ge=1, le=5, description="Severity level 1-5")

    model_config = {"json_schema_extra": {
        "examples": [{
            "violation_type": "FACE_DISAPPEARED",
            "timestamp": "2026-04-09T12:30:00",
            "message": "Face disappeared for more than 5 seconds",
            "student_id": "STUDENT_001",
            "severity": 3,
        }]
    }}


class SessionStartPayload(BaseModel):
    """Schema for starting a new proctoring session."""
    student_id: str = Field(..., description="Unique student identifier")
    student_name: Optional[str] = Field(None, description="Student display name")
    exam_name: Optional[str] = Field(None, description="Name of the exam")
    course: Optional[str] = Field(None, description="Course name")


# ── Response Models ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "2.0.0"
    timestamp: str
    uptime_seconds: float
    session_status: str


class StatsResponse(BaseModel):
    """Dashboard statistics response."""
    total_violations: int
    by_type: dict[str, int]
    cloud_alerts_count: int
    session_status: str
    session_duration: Optional[str] = None
    last_updated: str


class AlertEntry(BaseModel):
    """A single alert entry."""
    violation_type: str
    timestamp: str
    message: str = ""
    metadata: dict = {}
    student_id: Optional[str] = None
    severity: Optional[int] = None


class ReportEntry(BaseModel):
    """A single report file listing."""
    name: str
    url: str
    size_kb: float
    created: str


class SessionResponse(BaseModel):
    """Session status response."""
    session_id: Optional[str] = None
    status: str
    student_id: Optional[str] = None
    started_at: Optional[str] = None
    violation_count: int = 0
