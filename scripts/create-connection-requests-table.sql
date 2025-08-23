-- Create connection_requests table for managing caretaker-elderly connections
CREATE TABLE connection_requests (
  id SERIAL PRIMARY KEY,
  caretaker_id INTEGER NOT NULL REFERENCES caretakers(id) ON DELETE CASCADE,
  elderly_id INTEGER NOT NULL REFERENCES signuppage(id) ON DELETE CASCADE,
  caretaker_name TEXT NOT NULL,
  caretaker_facility TEXT NOT NULL,
  elderly_name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Create indexes for better performance
CREATE INDEX idx_connection_requests_caretaker_id ON connection_requests(caretaker_id);
CREATE INDEX idx_connection_requests_elderly_id ON connection_requests(elderly_id);
CREATE INDEX idx_connection_requests_status ON connection_requests(status);

-- Create unique constraint to prevent duplicate requests
CREATE UNIQUE INDEX idx_unique_connection_request ON connection_requests(caretaker_id, elderly_id) 
WHERE status IN ('pending', 'approved');

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_connection_requests_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = TIMEZONE('utc'::text, NOW());
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_connection_requests_updated_at
    BEFORE UPDATE ON connection_requests
    FOR EACH ROW
    EXECUTE FUNCTION update_connection_requests_updated_at();
