-- Safe migration to align Signuppage schema with app column names
-- Run this in the Supabase SQL Editor

-- 1) Ensure table name is lowercase `signuppage`
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'Signuppage'
  ) THEN
    EXECUTE 'ALTER TABLE "Signuppage" RENAME TO signuppage';
  END IF;
END $$;

-- 2) Rename camelCase columns to snake_case if they exist
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='signuppage' AND column_name='firstName'
  ) THEN
    EXECUTE 'ALTER TABLE signuppage RENAME COLUMN "firstName" TO first_name';
  END IF;

  -- handle unquoted camelCase that became all lowercase
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='signuppage' AND column_name='firstname'
  ) THEN
    EXECUTE 'ALTER TABLE signuppage RENAME COLUMN firstname TO first_name';
  END IF;

  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='signuppage' AND column_name='lastName'
  ) THEN
    EXECUTE 'ALTER TABLE signuppage RENAME COLUMN "lastName" TO last_name';
  END IF;

  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='signuppage' AND column_name='lastname'
  ) THEN
    EXECUTE 'ALTER TABLE signuppage RENAME COLUMN lastname TO last_name';
  END IF;

  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='signuppage' AND column_name='phoneNumber'
  ) THEN
    EXECUTE 'ALTER TABLE signuppage RENAME COLUMN "phoneNumber" TO phone_number';
  END IF;

  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='signuppage' AND column_name='phonenumber'
  ) THEN
    EXECUTE 'ALTER TABLE signuppage RENAME COLUMN phonenumber TO phone_number';
  END IF;
END $$;

-- 3) Ensure updated_at trigger exists
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_signuppage_updated_at ON signuppage;
CREATE TRIGGER update_signuppage_updated_at
    BEFORE UPDATE ON signuppage
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

