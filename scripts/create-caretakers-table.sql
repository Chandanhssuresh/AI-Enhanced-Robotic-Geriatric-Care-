-- Create caretakers table for healthcare professionals
CREATE TABLE caretakers (
  id SERIAL PRIMARY KEY,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  phone_number TEXT NOT NULL,
  facility_name TEXT NOT NULL,
  license_number TEXT NOT NULL,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Create indexes for better performance
CREATE INDEX idx_caretakers_email ON caretakers(email);
CREATE INDEX idx_caretakers_username ON caretakers(username);
CREATE INDEX idx_caretakers_license ON caretakers(license_number);
