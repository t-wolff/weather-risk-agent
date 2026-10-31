"""Anonymous client IDs. There is no login: the ID works like a password."""
import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app.config import env

MIN_CUSTOM_LENGTH = 12
MAX_FAILURES = 10
FAILURE_WINDOW_SECONDS = 600
RATE_WINDOW_SECONDS = 3600
MAX_CHATS_PER_CLIENT = 30
# IDs are free, so a client limit alone could be dodged by making more of them.
MAX_CHATS_PER_IP = 60
MAX_NEW_IDS_PER_IP = 20


def _secret() -> bytes:
    return env("APP_SECRET", "local-dev-secret-change-me").encode()


def keyed_hash(kind: str, value: str) -> str:
    return hmac.new(_secret(), f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()


def hash_client_id(client_id: str) -> str:
    return keyed_hash("client", client_id)


def generate_client_id() -> str:
    return secrets.token_urlsafe(16)


def valid_custom_id(client_id: str) -> bool:
    return len(client_id) >= MIN_CUSTOM_LENGTH and client_id.strip() == client_id


def client_ip(request: Request) -> str:
    if env("TRUST_FORWARDED_FOR", "false").lower() == "true":
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class FailureLimiter:
    """Counts attempts per key, in memory, so limits reset when the API restarts."""

    def __init__(self, max_failures: int = MAX_FAILURES, window: int = FAILURE_WINDOW_SECONDS,
                 message: str = "Too many failed attempts. Try again later."):
        self.max_failures = max_failures
        self.window = window
        self.message = message
        self.failures: dict[str, deque] = defaultdict(deque)

    def _prune(self, key: str) -> deque:
        attempts = self.failures[key]
        cutoff = time.monotonic() - self.window
        while attempts and attempts[0] < cutoff:
            attempts.popleft()
        return attempts

    def check(self, key: str) -> None:
        if len(self._prune(key)) >= self.max_failures:
            raise HTTPException(429, self.message)

    def record(self, key: str) -> None:
        self._prune(key).append(time.monotonic())


CHAT_LIMIT_MESSAGE = "You've sent too many messages. Try again later."
limiter = FailureLimiter()
client_chat_limiter = FailureLimiter(MAX_CHATS_PER_CLIENT, RATE_WINDOW_SECONDS, CHAT_LIMIT_MESSAGE)
ip_chat_limiter = FailureLimiter(MAX_CHATS_PER_IP, RATE_WINDOW_SECONDS, CHAT_LIMIT_MESSAGE)
new_id_limiter = FailureLimiter(MAX_NEW_IDS_PER_IP, RATE_WINDOW_SECONDS,
                                "Too many new client IDs. Try again later.")
ALL_LIMITERS = (limiter, client_chat_limiter, ip_chat_limiter, new_id_limiter)


def count_chat(client_hash: str, ip: str) -> None:
    """Counted before the LLM call, because a failed turn can still spend requests."""
    client_chat_limiter.check(client_hash)
    ip_chat_limiter.check(ip)
    client_chat_limiter.record(client_hash)
    ip_chat_limiter.record(ip)
