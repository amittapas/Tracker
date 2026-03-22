-- Run once against your Tracker database (psql, Neon, etc.)
ALTER TABLE health ADD COLUMN IF NOT EXISTS steps integer NOT NULL DEFAULT 0;
