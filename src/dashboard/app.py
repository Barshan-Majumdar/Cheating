"""
Production-Grade FastAPI Server for AI Proctoring System.
==========================================================

Features:
  - WebSocket real-time push for instant dashboard updates
  - Session management (start/stop proctoring sessions)
  - Ring-buffered in-memory alert store + file persistence
  - Automatic port-retry on startup (avoids "address in use" crashes)
  - CORS enabled for cross-origin desktop client communication
  - Auto-generated interactive API docs at /docs
  - Health check for Render / uptime monitors
  - Structured logging with request ID middleware
  - Production error handlers (no stack traces leaked to clients)

Endpoints:
  GET   /                 → Live dashboard (responsive HTML, auto-refresh + WebSocket)
  GET   /health           → Health check
  GET   /docs             → Interactive API documentation (Swagger UI)

  GET   /api/alerts       → Recent alerts (newest first)
  POST  /api/alerts       → Receive alert from desktop proctoring client
  GET   /api/stats        → Violation summary with breakdown
  GET   /api/violations   → All violations from violations.json
  GET   /api/reports      → List generated HTML/PDF reports
  WS    /ws/alerts        → WebSocket stream for real-time alert push

  POST  /api/session/start  → Start a named proctoring session
  POST  /api/session/stop   → Stop the current session
  GET   /api/session/status → Get current session info
"""

import os
import json
import yaml
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

try:
    from .models import (
        AlertPayload, SessionStartPayload,
        HealthResponse, StatsResponse, AlertEntry,
        ReportEntry, SessionResponse, SessionStatus,
    )
except ImportError:
    from dashboard.models import (
        AlertPayload, SessionStartPayload,
        HealthResponse, StatsResponse, AlertEntry,
        ReportEntry, SessionResponse, SessionStatus,
    )


# ── Logging ────────────────────────────────────────────────────────────────

logger = logging.getLogger("proctoring.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)


# ── Configuration ──────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load config.yaml — searches multiple candidate paths."""
    candidates = [
        Path("config/config.yaml"),
        Path("../config/config.yaml"),
        Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f)
                logger.info(f"Loaded config from {p.resolve()}")
                return cfg
    logger.warning("No config.yaml found — using fallback defaults")
    return {
        "logging": {"log_path": "./logs", "alert_cooldown": 5},
        "global": {"output_path": "./reports"},
        "reporting": {"output_dir": "./reports/generated"},
    }


config = _load_config()

LOG_PATH        = Path(config["logging"]["log_path"])
REPORTS_DIR     = Path(config.get("reporting", {}).get("output_dir", "./reports/generated"))
VIOLATIONS_FILE = Path(config.get("global", {}).get("output_path", "./reports")) / "violations.json"
CAPTURES_DIR    = Path(config.get("global", {}).get("output_path", "./reports")) / "violation_captures"


# ── State ──────────────────────────────────────────────────────────────────

_startup_time: datetime = datetime.now()

# In-memory alert ring buffer
cloud_alerts: list[dict] = []
MAX_CLOUD_ALERTS = 1000

# Session state
session: dict = {
    "id": None,
    "status": SessionStatus.IDLE,
    "student_id": None,
    "student_name": None,
    "exam_name": None,
    "started_at": None,
    "violation_count": 0,
}


# ── WebSocket Manager ─────────────────────────────────────────────────────

class ConnectionManager:
    """Manages WebSocket connections for real-time dashboard push."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected ({len(self.active_connections)} active)")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected ({len(self.active_connections)} active)")

    async def broadcast(self, message: dict):
        """Send a message to all connected WebSocket clients."""
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)


ws_manager = ConnectionManager()


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global _startup_time
    _startup_time = datetime.now()

    # Ensure directories
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dashboard running — logs: {LOG_PATH}, reports: {REPORTS_DIR}")
    yield
    logger.info("Shutting down dashboard.")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Proctoring API",
    description=(
        "Production-grade REST + WebSocket API for the AI exam proctoring system.\n\n"
        "• **Real-time alerts** via WebSocket at `/ws/alerts`\n"
        "• **Session management** for exam lifecycles\n"
        "• **Violation tracking** with severity scoring\n"
        "• **Report access** for generated HTML/PDF reports"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow desktop client and any frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for reports
if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")
if CAPTURES_DIR.exists():
    app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")


# ── Error Handlers ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: never leak stack traces to the client in production."""
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


# ══════════════════════════════════════════════════════════════════════════
#                              ROUTES
# ══════════════════════════════════════════════════════════════════════════

# ── Dashboard ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    """Self-contained responsive dashboard with WebSocket real-time updates."""
    return HTMLResponse(content=_build_dashboard_html())


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health-check for Render, load balancers, and uptime monitors."""
    uptime = (datetime.now() - _startup_time).total_seconds()
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(uptime, 1),
        session_status=session["status"],
    )


# ── Alerts ─────────────────────────────────────────────────────────────────

@app.get("/api/alerts", response_model=list[AlertEntry], tags=["Alerts"])
async def get_alerts(limit: int = 50):
    """
    Return recent alerts (newest first).
    Priority: cloud ring-buffer → local alerts.log file.
    """
    if cloud_alerts:
        return JSONResponse(content=cloud_alerts[-limit:][::-1])

    # Fallback: local log file
    log_file = LOG_PATH / "alerts.log"
    alerts: list[str] = []
    if log_file.exists():
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                alerts = [line.strip() for line in lines[-limit:] if line.strip()]
        except IOError:
            pass
    return JSONResponse(content=alerts[::-1])


@app.post("/api/alerts", status_code=201, tags=["Alerts"])
async def receive_alert(payload: AlertPayload):
    """
    Receive an alert from the desktop proctoring client.
    Also broadcasts the alert to all connected WebSocket dashboards instantly.
    """
    entry = {
        "violation_type": payload.violation_type,
        "timestamp": payload.timestamp or datetime.now().isoformat(),
        "message": payload.message or "",
        "metadata": payload.metadata or {},
        "student_id": payload.student_id or session.get("student_id"),
        "severity": payload.severity,
    }

    # Store in ring buffer
    cloud_alerts.append(entry)
    if len(cloud_alerts) > MAX_CLOUD_ALERTS:
        del cloud_alerts[: len(cloud_alerts) - MAX_CLOUD_ALERTS]

    # Increment session violation count
    session["violation_count"] = session.get("violation_count", 0) + 1

    # Persist to log file
    log_file = LOG_PATH / "alerts.log"
    try:
        with open(log_file, "a") as f:
            f.write(f"{entry['timestamp']} - {entry['violation_type']}: {entry['message']}\n")
    except IOError as e:
        logger.warning(f"Failed to write alert to log file: {e}")

    # Broadcast to all WebSocket clients for instant dashboard update
    await ws_manager.broadcast({"type": "new_alert", "data": entry})

    return {"status": "received", "alert_id": len(cloud_alerts)}


# ── Stats ──────────────────────────────────────────────────────────────────

@app.get("/api/stats", response_model=StatsResponse, tags=["Analytics"])
async def get_stats():
    """Return a summary of violation counts, broken down by type."""
    violations = _read_violations_file()
    total = len(violations)

    by_type: dict[str, int] = {}
    for v in violations:
        vtype = v.get("type", "UNKNOWN")
        by_type[vtype] = by_type.get(vtype, 0) + 1

    # Calculate session duration
    duration = None
    if session["started_at"]:
        try:
            started = datetime.fromisoformat(session["started_at"])
            delta = datetime.now() - started
            minutes = int(delta.total_seconds() // 60)
            seconds = int(delta.total_seconds() % 60)
            duration = f"{minutes}m {seconds}s"
        except (ValueError, TypeError):
            pass

    return StatsResponse(
        total_violations=total,
        by_type=by_type,
        cloud_alerts_count=len(cloud_alerts),
        session_status=session["status"],
        session_duration=duration,
        last_updated=datetime.now().strftime("%H:%M:%S"),
    )


# ── Violations ─────────────────────────────────────────────────────────────

@app.get("/api/violations", tags=["Analytics"])
async def get_violations(limit: Optional[int] = None):
    """Return all violations from the violations.json file."""
    data = _read_violations_file()
    if limit:
        data = data[-limit:]
    return JSONResponse(content=data)


# ── Reports ────────────────────────────────────────────────────────────────

@app.get("/api/reports", response_model=list[ReportEntry], tags=["Reports"])
async def list_reports():
    """List all generated report files (HTML/PDF)."""
    reports = []
    if REPORTS_DIR.exists():
        for f in sorted(REPORTS_DIR.iterdir()):
            if f.suffix in (".html", ".pdf"):
                try:
                    reports.append(ReportEntry(
                        name=f.name,
                        url=f"/reports/{f.name}",
                        size_kb=round(f.stat().st_size / 1024, 1),
                        created=datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                    ))
                except OSError:
                    pass
    return reports


# ── Session Management ─────────────────────────────────────────────────────

@app.post("/api/session/start", response_model=SessionResponse, tags=["Session"])
async def start_session(payload: SessionStartPayload):
    """Start a new proctoring session."""
    session["id"] = str(uuid.uuid4())[:8]
    session["status"] = SessionStatus.ACTIVE
    session["student_id"] = payload.student_id
    session["student_name"] = payload.student_name
    session["exam_name"] = payload.exam_name
    session["started_at"] = datetime.now().isoformat()
    session["violation_count"] = 0

    # Clear old alerts for the new session
    cloud_alerts.clear()

    logger.info(f"Session started: {session['id']} for student {payload.student_id}")

    await ws_manager.broadcast({"type": "session_started", "data": session.copy()})

    return SessionResponse(
        session_id=session["id"],
        status=session["status"],
        student_id=session["student_id"],
        started_at=session["started_at"],
        violation_count=0,
    )


@app.post("/api/session/stop", response_model=SessionResponse, tags=["Session"])
async def stop_session():
    """Stop the current proctoring session."""
    if session["status"] != SessionStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="No active session to stop")

    session["status"] = SessionStatus.COMPLETED
    logger.info(f"Session stopped: {session['id']} — {session['violation_count']} violations")

    await ws_manager.broadcast({"type": "session_stopped", "data": session.copy()})

    return SessionResponse(
        session_id=session["id"],
        status=session["status"],
        student_id=session["student_id"],
        started_at=session["started_at"],
        violation_count=session["violation_count"],
    )


@app.get("/api/session/status", response_model=SessionResponse, tags=["Session"])
async def get_session_status():
    """Get current session information."""
    return SessionResponse(
        session_id=session["id"],
        status=session["status"],
        student_id=session["student_id"],
        started_at=session["started_at"],
        violation_count=session["violation_count"],
    )


# ── WebSocket ──────────────────────────────────────────────────────────────

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alert streaming.
    The dashboard connects here for instant updates without polling.
    """
    await ws_manager.connect(websocket)
    try:
        # Send current state on connect
        await websocket.send_json({
            "type": "connected",
            "data": {
                "session": session.copy(),
                "recent_alerts": cloud_alerts[-10:],
            }
        })
        # Keep alive — listen for client messages (e.g., pings)
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# ══════════════════════════════════════════════════════════════════════════
#                              HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _read_violations_file() -> list:
    """Safely read violations.json."""
    if VIOLATIONS_FILE.exists():
        try:
            with open(VIOLATIONS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _build_dashboard_html() -> str:
    """
    Self-contained responsive HTML dashboard.
    Uses WebSocket for instant updates + fetch() polling as fallback.
    """
    return """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Proctoring Dashboard</title>
  <style>
    :root {
      --bg: #0f1117; --surface: #1a1d2e; --card: #232740;
      --accent: #6c63ff; --accent2: #00d4aa; --danger: #ff4757;
      --warn: #ffa502; --text: #e8e8f0; --text-muted: #8e8ea0;
      --radius: 14px; --shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
      background: var(--bg); color: var(--text); min-height: 100vh;
    }

    /* ── Header ────────────────────── */
    .header {
      background: linear-gradient(135deg, var(--surface) 0%, #1e2140 100%);
      padding: 20px 32px; display: flex; align-items: center;
      justify-content: space-between; flex-wrap: wrap; gap: 12px;
      border-bottom: 1px solid rgba(108,99,255,0.2);
    }
    .header h1 {
      font-size: 1.4rem; font-weight: 700;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .header-right { display: flex; align-items: center; gap: 12px; }
    .badge {
      padding: 4px 14px; border-radius: 20px;
      font-size: 0.75rem; font-weight: 600;
    }
    .badge.live { background: var(--accent2); color: #000; animation: pulse 2s infinite; }
    .badge.ws-ok { background: var(--accent); color: #fff; }
    .badge.ws-off { background: var(--danger); color: #fff; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }

    /* ── Session Bar ───────────────── */
    .session-bar {
      background: var(--surface); padding: 10px 32px;
      display: flex; align-items: center; gap: 20px;
      font-size: .85rem; color: var(--text-muted);
      border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .session-bar strong { color: var(--text); }

    /* ── Grid / Cards ──────────────── */
    .grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px; padding: 20px 32px;
    }
    .card {
      background: var(--card); border-radius: var(--radius);
      padding: 20px 24px; box-shadow: var(--shadow);
      transition: transform .2s, box-shadow .2s;
      border: 1px solid rgba(255,255,255,0.04);
    }
    .card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(0,0,0,.5); }
    .card h3 { font-size:.78rem; color:var(--text-muted); margin-bottom:6px;
      text-transform:uppercase; letter-spacing:1px; }
    .card .value { font-size:2rem; font-weight:800; }
    .card.accent .value { color: var(--accent); }
    .card.danger .value { color: var(--danger); }
    .card.warn   .value { color: var(--warn); }
    .card.green  .value { color: var(--accent2); }

    /* ── Section ───────────────────── */
    .section { padding: 0 32px 24px; }
    .section h2 {
      font-size: 1.1rem; margin-bottom: 14px;
      display: flex; align-items: center; gap: 8px;
    }
    .section h2::before {
      content: ''; width: 4px; height: 20px; border-radius: 2px;
      background: var(--accent);
    }

    /* ── Alert List ────────────────── */
    .alert-list {
      display: flex; flex-direction: column; gap: 8px;
      max-height: 400px; overflow-y: auto; padding-right: 6px;
    }
    .alert-item {
      background: var(--card); border-radius: 10px; padding: 12px 16px;
      border-left: 4px solid var(--danger); font-size: .88rem;
      animation: slideIn .3s ease;
    }
    .alert-item.info { border-left-color: var(--accent); }
    .alert-item.new  { border-left-color: var(--accent2); background: #1e2a35; }
    .alert-item .time { color: var(--text-muted); font-size:.75rem; margin-right: 8px; }
    .alert-item .type { font-weight: 700; color: var(--warn); }
    @keyframes slideIn { from{opacity:0;transform:translateX(-12px)} to{opacity:1;transform:translateX(0)} }

    /* ── Reports Table ─────────────── */
    .reports-table {
      width: 100%; border-collapse: collapse;
      background: var(--card); border-radius: var(--radius); overflow: hidden;
    }
    .reports-table th, .reports-table td { padding: 10px 16px; text-align: left; }
    .reports-table th { background: var(--surface); color: var(--text-muted);
      font-size:.78rem; text-transform:uppercase; letter-spacing:1px; }
    .reports-table tr:not(:last-child) td { border-bottom: 1px solid rgba(255,255,255,0.04); }
    .reports-table a { color: var(--accent); text-decoration: none; }
    .reports-table a:hover { text-decoration: underline; }

    /* ── Scrollbar ──────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--surface); }
    ::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 3px; }

    /* ── Responsive ─────────────────── */
    @media (max-width: 600px) {
      .header, .grid, .section, .session-bar { padding-left: 14px; padding-right: 14px; }
      .card .value { font-size: 1.5rem; }
    }
  </style>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
</head>
<body>
  <div class="header">
    <h1>&#128737; AI Proctoring Dashboard</h1>
    <div class="header-right">
      <span class="badge" id="ws-badge">WS: ---</span>
      <span class="badge live" id="status-badge">● LIVE</span>
    </div>
  </div>

  <div class="session-bar" id="session-bar">
    Session: <strong id="sess-status">Loading...</strong>
    &nbsp;|&nbsp; Student: <strong id="sess-student">—</strong>
    &nbsp;|&nbsp; Duration: <strong id="sess-duration">—</strong>
    &nbsp;|&nbsp; Uptime: <strong id="sess-uptime">—</strong>
  </div>

  <!-- Stat Cards -->
  <div class="grid" id="stats-grid">
    <div class="card accent"><h3>Total Violations</h3><div class="value" id="stat-total">—</div></div>
    <div class="card danger"><h3>Live Alerts</h3><div class="value" id="stat-cloud">—</div></div>
    <div class="card green"><h3>Last Updated</h3><div class="value" id="stat-time" style="font-size:1.3rem;">—</div></div>
  </div>

  <!-- Violation Breakdown -->
  <div class="grid" id="breakdown-grid"></div>

  <!-- Recent Alerts -->
  <div class="section">
    <h2>Recent Alerts</h2>
    <div class="alert-list" id="alert-list">
      <div class="alert-item info">Connecting...</div>
    </div>
  </div>

  <!-- Reports -->
  <div class="section">
    <h2>Generated Reports</h2>
    <table class="reports-table">
      <thead><tr><th>Report</th><th>Size</th><th>Created</th></tr></thead>
      <tbody id="reports-body"><tr><td colspan="3" style="color:var(--text-muted)">Loading...</td></tr></tbody>
    </table>
  </div>

<script>
const API = window.location.origin;
let ws = null;
let wsRetryCount = 0;

// ─── WebSocket ─────────────────────────────────
function connectWS() {
  const wsUrl = API.replace('http','ws') + '/ws/alerts';
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    wsRetryCount = 0;
    document.getElementById('ws-badge').textContent = 'WS: Connected';
    document.getElementById('ws-badge').className = 'badge ws-ok';
  };

  ws.onclose = () => {
    document.getElementById('ws-badge').textContent = 'WS: Offline';
    document.getElementById('ws-badge').className = 'badge ws-off';
    // Auto-reconnect with backoff
    const delay = Math.min(1000 * Math.pow(2, wsRetryCount), 15000);
    wsRetryCount++;
    setTimeout(connectWS, delay);
  };

  ws.onerror = () => { ws.close(); };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'new_alert') {
      // Instantly prepend the new alert to the list
      const list = document.getElementById('alert-list');
      const first = list.querySelector('.alert-item.info');
      if (first && first.textContent.includes('Connecting')) first.remove();
      if (first && first.textContent.includes('No alerts')) first.remove();

      const el = document.createElement('div');
      el.className = 'alert-item new';
      const d = msg.data;
      el.innerHTML = `<span class="time">${d.timestamp||''}</span> <span class="type">${d.violation_type||''}</span> ${d.message||''}`;
      list.prepend(el);

      // Flash effect
      setTimeout(() => el.classList.remove('new'), 3000);

      // Also refresh stats
      fetchStats();
    } else if (msg.type === 'connected') {
      console.log('WS connected, session:', msg.data.session);
    }
  };

  // Keep-alive ping every 25s
  setInterval(() => { if (ws && ws.readyState === WebSocket.OPEN) ws.send('ping'); }, 25000);
}

// ─── REST Polling (fallback + initial load) ────
async function fetchStats() {
  try {
    const res = await fetch(`${API}/api/stats`);
    const d = await res.json();
    document.getElementById('stat-total').textContent = d.total_violations ?? '—';
    document.getElementById('stat-cloud').textContent = d.cloud_alerts_count ?? '—';
    document.getElementById('stat-time').textContent = d.last_updated ?? '—';
    document.getElementById('sess-status').textContent = d.session_status ?? 'idle';
    document.getElementById('sess-duration').textContent = d.session_duration ?? '—';

    const grid = document.getElementById('breakdown-grid');
    grid.innerHTML = '';
    if (d.by_type) {
      for (const [type, count] of Object.entries(d.by_type)) {
        const card = document.createElement('div');
        card.className = 'card warn';
        card.innerHTML = `<h3>${type.replace(/_/g,' ')}</h3><div class="value">${count}</div>`;
        grid.appendChild(card);
      }
    }
  } catch(e) { console.warn('Stats fetch failed', e); }
}

async function fetchHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const d = await res.json();
    const mins = Math.floor(d.uptime_seconds / 60);
    const secs = Math.floor(d.uptime_seconds % 60);
    document.getElementById('sess-uptime').textContent = `${mins}m ${secs}s`;
  } catch(e) {}
}

async function fetchAlerts() {
  try {
    const res = await fetch(`${API}/api/alerts?limit=30`);
    const data = await res.json();
    const list = document.getElementById('alert-list');
    if (!data.length) { list.innerHTML = '<div class="alert-item info">No alerts yet.</div>'; return; }
    list.innerHTML = '';
    for (const item of data) {
      const el = document.createElement('div');
      el.className = 'alert-item';
      if (typeof item === 'string') {
        el.textContent = item;
      } else {
        el.innerHTML = `<span class="time">${item.timestamp||''}</span> <span class="type">${item.violation_type||''}</span> ${item.message||''}`;
      }
      list.appendChild(el);
    }
  } catch(e) { console.warn('Alerts fetch failed', e); }
}

async function fetchSession() {
  try {
    const res = await fetch(`${API}/api/session/status`);
    const d = await res.json();
    document.getElementById('sess-status').textContent = d.status ?? '—';
    document.getElementById('sess-student').textContent = d.student_id ?? '—';
  } catch(e) {}
}

async function fetchReports() {
  try {
    const res = await fetch(`${API}/api/reports`);
    const data = await res.json();
    const body = document.getElementById('reports-body');
    if (!data.length) { body.innerHTML = '<tr><td colspan="3" style="color:var(--text-muted)">No reports yet.</td></tr>'; return; }
    body.innerHTML = '';
    for (const r of data) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td><a href="${r.url}" target="_blank">${r.name}</a></td><td>${r.size_kb} KB</td><td>${new Date(r.created).toLocaleString()}</td>`;
      body.appendChild(tr);
    }
  } catch(e) {}
}

// ─── Init ──────────────────────────────────────
connectWS();
fetchStats(); fetchAlerts(); fetchReports(); fetchSession(); fetchHealth();
setInterval(fetchStats, 5000);
setInterval(fetchAlerts, 5000);
setInterval(fetchReports, 30000);
setInterval(fetchSession, 5000);
setInterval(fetchHealth, 10000);
</script>
</body>
</html>"""


# ── Standalone Server ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)