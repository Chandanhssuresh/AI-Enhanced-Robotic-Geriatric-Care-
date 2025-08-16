-- Create the Signuppage table in Supabase
-- Run this script in your Supabase SQL Editor

CREATE TABLE IF NOT EXISTS signuppage (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    phone_number VARCHAR(20) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create an index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_signuppage_email ON signuppage(email);

-- Create an index on username for faster lookups
CREATE INDEX IF NOT EXISTS idx_signuppage_username ON signuppage(username);

-- Add a trigger to automatically update the updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_signuppage_updated_at 
    BEFORE UPDATE ON signuppage 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();
