CREATE TABLE IF NOT EXISTS user_goals (
    metric_key TEXT PRIMARY KEY,
    current_target NUMERIC NOT NULL,
    max_achieved NUMERIC NOT NULL DEFAULT 0
);

-- Seed defaults (idempotent: only insert missing keys from app on first run)
