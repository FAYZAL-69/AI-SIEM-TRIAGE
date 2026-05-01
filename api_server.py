from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import sqlite3
import os
import logging
from datetime import datetime

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    filename="siem.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ── API Key Auth ───────────────────────────────────────────────
API_KEY = os.getenv("SIEM_API_KEY", "changeme-before-deploy")  # Set via environment variable
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Depends(api_key_header)):
    if key != API_KEY:
        logger.warning("Unauthorized access attempt.")
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key

# ── Rate Limiter ───────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="SIEM Log Ingestion API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Database setup ─────────────────────────────────────────────
DB_FILE = "siem_logs.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            dur       REAL,
            proto     TEXT,
            service   TEXT,
            state     TEXT,
            spkts     INTEGER,
            dpkts     INTEGER,
            sbytes    INTEGER,
            dbytes    INTEGER,
            rate      REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Allowed values (whitelist) ─────────────────────────────────
ALLOWED_PROTO   = {"tcp", "udp", "icmp", "arp", "ospf", "sctp"}
ALLOWED_SERVICE = {"http", "ftp", "ftp-data", "dns", "ssh", "ssl",
                   "smtp", "pop3", "irc", "dhcp", "snmp", "radius", "-"}
ALLOWED_STATE   = {"CON", "FIN", "INT", "REQ", "RST", "CLO", "ACC"}

# ── Request model ──────────────────────────────────────────────
class LogEntry(BaseModel):
    dur:     float
    proto:   str
    service: str
    state:   str
    spkts:   int
    dpkts:   int
    sbytes:  int
    dbytes:  int
    rate:    float

    @validator("proto")
    def validate_proto(cls, v):
        if v not in ALLOWED_PROTO:
            raise ValueError(f"Invalid proto '{v}'. Allowed: {ALLOWED_PROTO}")
        return v

    @validator("service")
    def validate_service(cls, v):
        if v not in ALLOWED_SERVICE:
            raise ValueError(f"Invalid service '{v}'. Allowed: {ALLOWED_SERVICE}")
        return v

    @validator("state")
    def validate_state(cls, v):
        if v not in ALLOWED_STATE:
            raise ValueError(f"Invalid state '{v}'. Allowed: {ALLOWED_STATE}")
        return v

    @validator("dur", "rate")
    def must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Value must be non-negative.")
        return v

    @validator("spkts", "dpkts", "sbytes", "dbytes")
    def must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("Packet/byte counts must be non-negative.")
        return v

# ── Routes ─────────────────────────────────────────────────────

@app.post("/ingest_log", dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def ingest_log(request: Request, log: LogEntry):
    """Ingest a single network log entry into the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO logs (timestamp, dur, proto, service, state,
                              spkts, dpkts, sbytes, dbytes, rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            log.dur, log.proto, log.service, log.state,
            log.spkts, log.dpkts, log.sbytes, log.dbytes, log.rate
        ))
        conn.commit()
        total = cursor.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        conn.close()
        logger.info(f"Log ingested: proto={log.proto} service={log.service} state={log.state}")
        return {"status": "logged", "total": total}
    except Exception as e:
        logger.error(f"Failed to ingest log: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/logs", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def get_logs(request: Request, limit: int = 100, offset: int = 0):
    """Retrieve paginated logs from the database."""
    if limit > 500:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 500.")
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM logs ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to retrieve logs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/health")
async def health_check():
    """Public health check endpoint — no auth required."""
    return {"status": "ok", "db": DB_FILE}


# ── Run with: ──────────────────────────────────────────────────
# SIEM_API_KEY=your-secret-key uvicorn api_server:app --reload