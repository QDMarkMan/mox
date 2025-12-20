-- PostgreSQL schema for conversation storage
-- Run this to initialize the database: psql -d molx -f schema.sql

CREATE TABLE IF NOT EXISTS conversations (
    session_id UUID PRIMARY KEY,
    messages JSONB NOT NULL DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for efficient cleanup of expired sessions
CREATE INDEX IF NOT EXISTS idx_conversations_activity 
    ON conversations(last_activity);

-- Index for efficient listing
CREATE INDEX IF NOT EXISTS idx_conversations_created 
    ON conversations(created_at DESC);

-- Comments for documentation
COMMENT ON TABLE conversations IS 'Stores conversation sessions for MolX Agent';
COMMENT ON COLUMN conversations.session_id IS 'Unique session identifier (UUID)';
COMMENT ON COLUMN conversations.messages IS 'Array of conversation messages in JSONB format';
COMMENT ON COLUMN conversations.metadata IS 'Additional session metadata in JSONB format';
COMMENT ON COLUMN conversations.created_at IS 'Session creation timestamp';
COMMENT ON COLUMN conversations.last_activity IS 'Last activity timestamp for TTL expiration';
