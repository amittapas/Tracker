-- No-fap / no-porn streak: fixed epoch (first 4pm anchor) + relapse events.
CREATE TABLE IF NOT EXISTS nfp_streak (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  epoch_started_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS nfp_relapse (
  id SERIAL PRIMARY KEY,
  relapsed_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nfp_relapse_at ON nfp_relapse (relapsed_at);
