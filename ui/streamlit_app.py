import os

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
TIMEOUT = 180
EXAMPLES = [
    "Which hubs in the Midwest are most exposed to winter disruption?",
    "Compare Miami and Houston in terms of hurricane and flood exposure.",
    "What percentage of days in Denver last year had snowfall?",
    "Why is the Dallas hub's weather disruption risk high?",
]

st.set_page_config(page_title="Weather Risk Agent", layout="wide")


def request_headers() -> dict:
    headers = {}
    if st.session_state.get("client_id"):
        headers["X-Client-ID"] = st.session_state.client_id
    ip = getattr(st.context, "ip_address", None)
    if isinstance(ip, str) and ip:
        headers["X-Forwarded-For"] = ip
    return headers


def api(method: str, path: str, **kwargs) -> httpx.Response:
    return httpx.request(method, f"{API_URL}{path}", headers=request_headers(), timeout=TIMEOUT, **kwargs)


def error_text(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail"))
    except ValueError:
        return f"Request failed with status {response.status_code}"


def start_screen() -> None:
    st.title("Weather Risk Intelligence Agent")
    st.write("Chats are kept under a client ID. There is no login, so anyone with your ID can see your chats.")
    choice = st.radio(
        "How would you like to start?",
        ["Generate an ID for me", "Create my own ID", "I already have an ID"],
    )
    typed = ""
    if choice != "Generate an ID for me":
        typed = st.text_input("Client ID", help="At least 12 characters")
    if not st.button("Continue", type="primary"):
        return
    if choice == "I already have an ID":
        response = api("POST", "/clients/verify", json={"client_id": typed})
        if response.status_code != 200:
            st.error(error_text(response))
            return
        st.session_state.client_id = typed
    else:
        body = {"mode": "generate"} if choice.startswith("Generate") else {"mode": "custom", "client_id": typed}
        response = api("POST", "/clients", json=body)
        if response.status_code != 201:
            st.error(error_text(response))
            return
        st.session_state.client_id = response.json()["client_id"]
        st.session_state.show_id = True
    st.rerun()


def render_yearly(result: dict) -> None:
    if not result.get("years"):
        return
    frame = pd.DataFrame(result["years"]).set_index("year")
    st.caption(f"{result['hub']}: {result['metric'].replace('_', ' ')} per year")
    st.bar_chart(frame["count"])


def render_answer(payload: dict) -> None:
    answer = payload["answer"]
    if not payload["verified"]:
        st.warning("Unverified: some numbers in this answer couldn't be matched to tool results.")
    st.markdown(answer["answer_markdown"])
    if answer["ranked_hubs"]:
        st.dataframe(pd.DataFrame(answer["ranked_hubs"]), hide_index=True)
    for tool in payload.get("tools_used", []):
        if tool["name"] == "yearly_history" and not tool["is_error"]:
            render_yearly(tool["result"])
    st.caption(f"Confidence: {answer['confidence']}. {answer['confidence_reason']}")
    with st.expander("Assumptions, sources and reasoning"):
        for item in answer["assumptions"]:
            st.markdown(f"- {item}")
        if answer["data_sources"]:
            st.markdown("**Sources:** " + ", ".join(answer["data_sources"]))
        st.markdown(f"**Reasoning:** {answer['reasoning_summary']}")
        names = [tool["name"] for tool in payload.get("tools_used", [])]
        if names:
            st.markdown("**Tools:** " + ", ".join(names))


def load_conversation(conversation_id: str) -> None:
    response = api("GET", f"/conversations/{conversation_id}")
    if response.status_code != 200:
        st.error(error_text(response))
        return
    st.session_state.conversation_id = conversation_id
    st.session_state.chat = [
        {
            "question": turn["user_message"],
            "payload": {
                "answer": turn["answer"],
                "verified": turn["answer"].get("verified", True),
                "tools_used": turn["trace"]["tool_calls"],
            },
        }
        for turn in response.json()
    ]


def sidebar() -> None:
    with st.sidebar:
        if st.session_state.pop("show_id", False):
            st.info("Save your client ID. You need it to return to these chats.")
            st.code(st.session_state.client_id)
        if st.button("New chat"):
            st.session_state.conversation_id = None
            st.session_state.chat = []
        response = api("GET", "/conversations")
        if response.status_code == 200:
            for item in response.json():
                label = (item["first_message"] or "Empty chat")[:40]
                if st.button(label, key=item["id"]):
                    load_conversation(item["id"])
        st.divider()
        hubs = api("GET", "/hubs")
        if hubs.status_code == 200 and hubs.json():
            st.subheader("All hubs")
            frame = pd.DataFrame(hubs.json())[["name", "edd", "tier"]].round(1)
            st.dataframe(frame, hide_index=True)
            st.caption("Expected disruption days per year")


def chat_screen() -> None:
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("conversation_id", None)
    sidebar()
    st.title("Weather Risk Intelligence Agent")
    for item in st.session_state.chat:
        st.chat_message("user").write(item["question"])
        with st.chat_message("assistant"):
            render_answer(item["payload"])

    question = st.chat_input("Ask about hub weather risk", max_chars=2000)
    if not st.session_state.chat:
        for example in EXAMPLES:
            if st.button(example):
                question = example
    if not question:
        return
    st.chat_message("user").write(question)
    with st.chat_message("assistant"), st.spinner("Checking the data..."):
        response = api("POST", "/chat", json={
            "message": question, "conversation_id": st.session_state.conversation_id,
        })
        if response.status_code != 200:
            st.error(error_text(response))
            return
        payload = response.json()
    st.session_state.conversation_id = payload["conversation_id"]
    st.session_state.chat.append({"question": question, "payload": payload})
    st.rerun()


try:
    if st.session_state.get("client_id"):
        chat_screen()
    else:
        start_screen()
except httpx.HTTPError:
    st.error(f"The API isn't reachable at {API_URL}. Start it with `make api`.")
