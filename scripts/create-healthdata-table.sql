-- Create the healthdata table for storing real-time health sensor data
CREATE TABLE healthdata (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES signuppage(id) ON DELETE CASCADE,
  heart_rate INTEGER NOT NULL,
  spo2 INTEGER NOT NULL,
  body_temperature DECIMAL(4,1) NOT NULL,
  predicted_condition TEXT NOT NULL,
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Create indexes for better performance
CREATE INDEX idx_healthdata_user_id ON healthdata(user_id);
CREATE INDEX idx_healthdata_timestamp ON healthdata(timestamp);
CREATE INDEX idx_healthdata_user_timestamp ON healthdata(user_id, timestamp);

-- Add comments for documentation
COMMENT ON TABLE healthdata IS 'Stores real-time health sensor data from AI robotic care system';
COMMENT ON COLUMN healthdata.heart_rate IS 'Heart rate in beats per minute (BPM)';
COMMENT ON COLUMN healthdata.spo2 IS 'Oxygen saturation percentage (SpO2)';
COMMENT ON COLUMN healthdata.body_temperature IS 'Body temperature in Celsius';
COMMENT ON COLUMN healthdata.predicted_condition IS 'ML model predicted health condition';
