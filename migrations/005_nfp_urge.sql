-- Log moments an urge was felt and overcome (separate from streak resets).
CREATE TABLE IF NOT EXISTS nfp_urge (
  id SERIAL PRIMARY KEY,
  logged_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nfp_urge_at ON nfp_urge (logged_at);
