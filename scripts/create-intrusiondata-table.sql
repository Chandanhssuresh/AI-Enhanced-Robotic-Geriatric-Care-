-- Create Intrusion Detection Data table
CREATE TABLE intrusiondata (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES signuppage(id) ON DELETE CASCADE,
  intrusion_detected BOOLEAN NOT NULL,
  no_intrusion BOOLEAN NOT NULL,
  predicted_status VARCHAR(30) NOT NULL CHECK (predicted_status IN ('intrusion_detected', 'no_intrusion')),
  confidence_score INTEGER NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 100),
  risk_level VARCHAR(10) NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Create indexes for better performance
CREATE INDEX idx_intrusiondata_user_id ON intrusiondata(user_id);
CREATE INDEX idx_intrusiondata_created_at ON intrusiondata(created_at);
CREATE INDEX idx_intrusiondata_predicted_status ON intrusiondata(predicted_status);
CREATE INDEX idx_intrusiondata_risk_level ON intrusiondata(risk_level);

-- Add comments for documentation
COMMENT ON TABLE intrusiondata IS 'Stores real-time intrusion detection data from AI robot security sensors';
COMMENT ON COLUMN intrusiondata.intrusion_detected IS 'Whether an intrusion was detected (true/false)';
COMMENT ON COLUMN intrusiondata.no_intrusion IS 'Whether no intrusion was detected (true/false)';
COMMENT ON COLUMN intrusiondata.predicted_status IS 'ML model predicted security status';
COMMENT ON COLUMN intrusiondata.confidence_score IS 'ML model confidence level (0-100%)';
COMMENT ON COLUMN intrusiondata.risk_level IS 'Assessed risk level (low/medium/high)';
