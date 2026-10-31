CREATE TABLE IF NOT EXISTS nri_hazards (
    hub_id TEXT NOT NULL,
    nri_field TEXT NOT NULL,
    annual_frequency REAL,
    loss_ratio_rating TEXT,
    PRIMARY KEY (hub_id, nri_field)
);

CREATE TABLE IF NOT EXISTS weather_cache (
    hub_id TEXT NOT NULL,
    year INTEGER NOT NULL,
    days_json TEXT NOT NULL,
    complete INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (hub_id, year)
);

CREATE TABLE IF NOT EXISTS declarations (
    hub_id TEXT NOT NULL,
    disaster_number INTEGER NOT NULL,
    declaration_type TEXT NOT NULL,
    incident_type TEXT NOT NULL,
    title TEXT NOT NULL,
    declaration_date TEXT NOT NULL,
    PRIMARY KEY (hub_id, disaster_number)
);

CREATE TABLE IF NOT EXISTS score_snapshots (
    hub_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    method_version TEXT NOT NULL,
    edd REAL NOT NULL,
    tier TEXT NOT NULL,
    breakdown_json TEXT NOT NULL,
    PRIMARY KEY (hub_id, as_of, method_version)
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hub_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    old_edd REAL NOT NULL,
    new_edd REAL NOT NULL,
    old_tier TEXT NOT NULL,
    new_tier TEXT NOT NULL,
    reason TEXT NOT NULL,
    delivery_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (hub_id, as_of)
);

CREATE TABLE IF NOT EXISTS clients (
    client_hash TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    client_hash TEXT NOT NULL REFERENCES clients (client_hash),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_client ON conversations (client_hash);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations (id),
    user_message TEXT NOT NULL,
    answer_json TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    llm_messages_json TEXT NOT NULL,
    flag TEXT NOT NULL,
    flag_reason TEXT NOT NULL,
    flag_source TEXT NOT NULL,
    ip_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns (conversation_id);
CREATE INDEX IF NOT EXISTS idx_turns_flag ON turns (flag);
