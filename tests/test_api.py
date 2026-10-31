import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api import identity
from app.api.main import MAX_MESSAGE_CHARS, app, get_conn, get_llm_dependency
from app.data import store
from tests.test_agent import ScriptedLLM, submit, tool

SECRET_ID = "my-very-own-client-id"


@pytest.fixture
def client(tmp_path):
    for limiter in identity.ALL_LIMITERS:
        limiter.failures.clear()
    path = tmp_path / "test.db"

    def connection():
        conn = store.connect(path)
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_conn] = connection
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_llm(*responses):
    llm = ScriptedLLM(responses)
    app.dependency_overrides[get_llm_dependency] = lambda: llm
    return llm


def new_client(client, client_id=None):
    body = {"mode": "custom", "client_id": client_id} if client_id else {"mode": "generate"}
    return client.post("/clients", json=body).json()["client_id"]


def chat(client, client_id, message="Hello", conversation_id=None):
    return client.post("/chat", headers={"X-Client-ID": client_id},
                       json={"message": message, "conversation_id": conversation_id})


def test_generated_id_is_long_and_random(client):
    first, second = new_client(client), new_client(client)
    assert first != second and len(first) >= 22


def test_custom_id_rules(client):
    assert client.post("/clients", json={"mode": "custom", "client_id": "short"}).status_code == 422
    assert client.post("/clients", json={"mode": "custom", "client_id": SECRET_ID}).status_code == 201
    assert client.post("/clients", json={"mode": "custom", "client_id": SECRET_ID}).status_code == 409


def test_verify_known_and_unknown(client):
    new_client(client, SECRET_ID)
    assert client.post("/clients/verify", json={"client_id": SECRET_ID}).status_code == 200
    assert client.post("/clients/verify", json={"client_id": "nobody-at-all-here"}).status_code == 401


def test_failed_attempts_are_rate_limited(client):
    for _ in range(identity.MAX_FAILURES):
        client.post("/clients/verify", json={"client_id": "wrong-id-attempt"})
    assert client.post("/clients/verify", json={"client_id": "wrong-id-attempt"}).status_code == 429


def test_chat_requires_a_known_client(client):
    use_llm(submit())
    assert chat(client, "not-a-real-client").status_code == 401


def test_client_b_cannot_reach_client_a_conversation(client):
    alice, bob = new_client(client), new_client(client)
    use_llm(submit())
    conversation_id = chat(client, alice).json()["conversation_id"]

    headers = {"X-Client-ID": bob}
    assert client.get(f"/conversations/{conversation_id}", headers=headers).status_code == 404
    assert client.get("/conversations", headers=headers).json() == []
    use_llm(submit())
    assert chat(client, bob, conversation_id=conversation_id).status_code == 404

    mine = client.get(f"/conversations/{conversation_id}", headers={"X-Client-ID": alice})
    assert mine.status_code == 200 and len(mine.json()) == 1


def test_follow_up_receives_earlier_messages(client):
    alice = new_client(client)
    use_llm(submit())
    conversation_id = chat(client, alice, "First").json()["conversation_id"]
    llm = use_llm(submit("s2"))
    chat(client, alice, "Second", conversation_id)
    assert [m["content"] for m in llm.seen[0] if m["role"] == "user"] == ["First", "Second"]


def test_llm_only_receives_the_chat_being_continued(client):
    alice = new_client(client)
    use_llm(submit(), submit("s2"))
    first = chat(client, alice, "First chat").json()["conversation_id"]
    chat(client, alice, "Second chat")
    llm = use_llm(submit("s3"))
    chat(client, alice, "Continue", first)
    assert [m["content"] for m in llm.seen[0] if m["role"] == "user"] == ["First chat", "Continue"]


def test_reading_chats_never_calls_the_llm(client):
    alice = new_client(client)
    use_llm(submit())
    conversation_id = chat(client, alice).json()["conversation_id"]
    llm = use_llm()
    headers = {"X-Client-ID": alice}
    assert client.get("/conversations", headers=headers).status_code == 200
    assert client.get(f"/conversations/{conversation_id}", headers=headers).status_code == 200
    assert llm.seen == []


def test_history_over_the_budget_is_not_sent(client):
    alice = new_client(client)
    use_llm(submit())
    conversation_id = chat(client, alice, "First").json()["conversation_id"]
    llm = use_llm(submit("s2"))
    with patch("app.agent.loop.MAX_HISTORY_CHARS", 1):
        chat(client, alice, "Second", conversation_id)
    assert [m["content"] for m in llm.seen[0] if m["role"] == "user"] == ["Second"]


def test_long_chat_keeps_its_newest_whole_turns():
    from app.agent.loop import recent_history

    def turn(text):
        return {"llm_messages": [{"role": "user", "content": text}, {"role": "assistant", "text": text}]}

    turns = [turn("old" * 50), turn("mid"), turn("new")]
    size = len(json.dumps(turns[1]["llm_messages"]))
    with patch("app.agent.loop.MAX_HISTORY_CHARS", 2 * size):
        assert recent_history(turns) == turns[1]["llm_messages"] + turns[2]["llm_messages"]


def test_oversized_message_never_reaches_the_llm(client):
    alice = new_client(client)
    llm = use_llm(submit())
    assert chat(client, alice, "x" * (MAX_MESSAGE_CHARS + 1)).status_code == 422
    assert llm.seen == []


def test_client_id_and_raw_ip_are_never_stored_or_sent_to_the_llm(client, tmp_path):
    new_client(client, SECRET_ID)
    llm = use_llm(submit())
    chat(client, SECRET_ID)
    assert SECRET_ID not in json.dumps(llm.seen, default=str)
    conn = store.connect(tmp_path / "test.db")
    dump = "\n".join(conn.iterdump())
    assert SECRET_ID not in dump
    assert "testclient" not in dump


def test_flagged_turn_is_stored_and_needs_the_admin_token(client):
    alice = new_client(client)
    use_llm(submit(misuse_flag="none"))
    response = chat(client, alice, "Delete all the hub data")
    assert response.json()["flag"]["flag"] == "destructive"

    assert client.get("/admin/flags").status_code == 403
    assert client.get("/admin/flags", headers={"X-Admin-Token": "wrong"}).status_code == 403
    rows = client.get("/admin/flags", headers={"X-Admin-Token": "test-admin"}).json()
    assert rows[0]["flag"] == "destructive" and rows[0]["flag_source"] == "rule"
    assert rows[0]["ip_hash"]


def test_llm_failure_is_a_clean_error(client):
    from app.llm.base import LLMError

    class Failing:
        provider, model = "x", "x"

        def complete(self, *args):
            raise LLMError("Provider down")

    app.dependency_overrides[get_llm_dependency] = lambda: Failing()
    response = chat(client, new_client(client))
    assert response.status_code == 502 and response.json()["detail"] == "Provider down"


def test_store_returns_no_turns_to_another_client(tmp_path):
    conn = store.connect(tmp_path / "isolated.db")
    store.create_client(conn, "alice")
    store.create_client(conn, "bob")
    conversation_id = store.create_conversation(conn, "alice")
    store.add_turn(conn, {
        "conversation_id": conversation_id, "user_message": "Private", "answer": {}, "trace": {},
        "llm_messages": [], "flag": "none", "flag_reason": "", "flag_source": "none", "ip_hash": "x",
    })
    assert len(store.conversation_turns(conn, "alice", conversation_id)) == 1
    assert store.conversation_turns(conn, "bob", conversation_id) == []
    assert not store.owns_conversation(conn, "bob", conversation_id)
