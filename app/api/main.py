import hmac
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.agent.loop import recent_history, record_turn, run_turn
from app.api import identity
from app.config import env, hub_by_id, scoring_config
from app.data import store
from app.jobs.refresh import run_refresh
from app.llm.base import LLMError
from app.llm.factory import get_llm

logger = logging.getLogger(__name__)
MAX_MESSAGE_CHARS = 2000


def daily_refresh() -> None:
    conn = store.connect()
    try:
        run_refresh(conn)
    except Exception:
        logger.exception("Daily refresh failed")
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = None
    if env("ENABLE_SCHEDULER").lower() == "true":
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(daily_refresh, "cron", hour=6)
        scheduler.start()
    yield
    if scheduler:
        scheduler.shutdown()


app = FastAPI(title="Weather Risk Intelligence Agent", lifespan=lifespan)


def get_conn():
    conn = store.connect()
    try:
        yield conn
    finally:
        conn.close()


def get_llm_dependency():
    try:
        return get_llm()
    except Exception as error:
        raise HTTPException(503, f"LLM provider isn't configured: {error}") from error


def require_client(request: Request, x_client_id: str = Header(), conn=Depends(get_conn)) -> str:
    ip = identity.client_ip(request)
    identity.limiter.check(ip)
    client_hash = identity.hash_client_id(x_client_id)
    if not store.client_exists(conn, client_hash):
        identity.limiter.record(ip)
        raise HTTPException(401, "Unknown client ID")
    return client_hash


class ClientRequest(BaseModel):
    mode: Literal["generate", "custom"]
    client_id: str | None = None


class VerifyRequest(BaseModel):
    client_id: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: str | None = None


@app.get("/health")
def health(conn=Depends(get_conn)):
    scores = store.latest_snapshots(conn, scoring_config()["method_version"])
    return {"status": "ok", "hubs_scored": len(scores)}


@app.post("/clients", status_code=201)
def create_client(body: ClientRequest, request: Request, conn=Depends(get_conn)):
    ip = identity.client_ip(request)
    identity.limiter.check(ip)
    identity.new_id_limiter.check(ip)
    if body.mode == "generate":
        client_id = identity.generate_client_id()
    else:
        client_id = body.client_id or ""
        if not identity.valid_custom_id(client_id):
            raise HTTPException(422, f"Use at least {identity.MIN_CUSTOM_LENGTH} characters, no outer spaces.")
    if not store.create_client(conn, identity.hash_client_id(client_id)):
        identity.limiter.record(ip)
        raise HTTPException(409, "That ID isn't available.")
    identity.new_id_limiter.record(ip)
    return {"client_id": client_id}


@app.post("/clients/verify")
def verify_client(body: VerifyRequest, request: Request, conn=Depends(get_conn)):
    ip = identity.client_ip(request)
    identity.limiter.check(ip)
    if not store.client_exists(conn, identity.hash_client_id(body.client_id)):
        identity.limiter.record(ip)
        raise HTTPException(401, "Unknown client ID")
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatRequest, request: Request, client_hash: str = Depends(require_client),
         conn=Depends(get_conn), llm=Depends(get_llm_dependency)):
    identity.count_chat(client_hash, identity.client_ip(request))
    if body.conversation_id:
        if not store.owns_conversation(conn, client_hash, body.conversation_id):
            raise HTTPException(404, "Conversation not found")
        conversation_id = body.conversation_id
    else:
        conversation_id = store.create_conversation(conn, client_hash)
    history = recent_history(store.conversation_turns(conn, client_hash, conversation_id))
    try:
        result = run_turn(conn, llm, conversation_id, history, body.message)
    except LLMError as error:
        raise HTTPException(502, str(error)) from error
    except Exception as error:
        logger.exception("Chat turn failed")
        raise HTTPException(500, "Something went wrong answering that. Please try again.") from error
    record_turn(conn, result, body.message, identity.keyed_hash("ip", identity.client_ip(request)))
    return {
        "conversation_id": conversation_id,
        "answer": result["answer"],
        "verified": result["verified"],
        "flag": result["flag"],
        "tools_used": [
            {"name": call["name"], "result": call["result"], "is_error": call["is_error"]}
            for call in result["trace"]["tool_calls"]
        ],
    }


@app.get("/conversations")
def conversations(client_hash: str = Depends(require_client), conn=Depends(get_conn)):
    return store.list_conversations(conn, client_hash)


@app.get("/conversations/{conversation_id}")
def conversation(conversation_id: str, client_hash: str = Depends(require_client),
                 conn=Depends(get_conn)):
    if not store.owns_conversation(conn, client_hash, conversation_id):
        raise HTTPException(404, "Conversation not found")
    turns = store.conversation_turns(conn, client_hash, conversation_id)
    return [{k: turn[k] for k in ("user_message", "answer", "trace", "flag", "created_at")} for turn in turns]


@app.get("/hubs")
def hubs_scores(conn=Depends(get_conn)):
    scores = store.latest_snapshots(conn, scoring_config()["method_version"])
    return sorted(
        ({k: s[k] for k in ("hub_id", "name", "region", "edd", "tier", "as_of")} for s in scores),
        key=lambda s: -s["edd"],
    )


@app.get("/hubs/{hub_id}/score")
def hub_score(hub_id: str, conn=Depends(get_conn)):
    if not hub_by_id(hub_id):
        raise HTTPException(404, "Hub not found")
    scores = store.latest_snapshots(conn, scoring_config()["method_version"])
    score = next((s for s in scores if s["hub_id"] == hub_id), None)
    if not score:
        raise HTTPException(404, "No score yet")
    return score


@app.get("/alerts")
def alerts(conn=Depends(get_conn)):
    return store.list_alerts(conn)


@app.get("/admin/flags")
def admin_flags(flag: str | None = None, x_admin_token: str = Header(default=""), conn=Depends(get_conn)):
    token = env("ADMIN_TOKEN")
    if not token or not hmac.compare_digest(token, x_admin_token):
        raise HTTPException(403, "Forbidden")
    return store.flagged_turns(conn, flag)
