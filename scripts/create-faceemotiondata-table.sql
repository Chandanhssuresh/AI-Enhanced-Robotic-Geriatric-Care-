-- Create Face Emotion Data table
CREATE TABLE faceemotiondata (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES signuppage(id) ON DELETE CASCADE,
  happy_score INTEGER NOT NULL CHECK (happy_score >= 0 AND happy_score <= 100),
  sad_score INTEGER NOT NULL CHECK (sad_score >= 0 AND sad_score <= 100),
  angry_score INTEGER NOT NULL CHECK (angry_score >= 0 AND angry_score <= 100),
  neutral_score INTEGER NOT NULL CHECK (neutral_score >= 0 AND neutral_score <= 100),
  predicted_emotion VARCHAR(20) NOT NULL CHECK (predicted_emotion IN ('happy', 'sad', 'angry', 'neutral')),
  confidence_score INTEGER NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 100),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Create indexes for better performance
CREATE INDEX idx_faceemotiondata_user_id ON faceemotiondata(user_id);
CREATE INDEX idx_faceemotiondata_created_at ON faceemotiondata(created_at);
CREATE INDEX idx_faceemotiondata_predicted_emotion ON faceemotiondata(predicted_emotion);

-- Add comments for documentation
COMMENT ON TABLE faceemotiondata IS 'Stores real-time face emotion detection data from AI robot sensors';
COMMENT ON COLUMN faceemotiondata.happy_score IS 'Happy emotion detection score (0-100%)';
COMMENT ON COLUMN faceemotiondata.sad_score IS 'Sad emotion detection score (0-100%)';
COMMENT ON COLUMN faceemotiondata.angry_score IS 'Angry emotion detection score (0-100%)';
COMMENT ON COLUMN faceemotiondata.neutral_score IS 'Neutral emotion detection score (0-100%)';
COMMENT ON COLUMN faceemotiondata.predicted_emotion IS 'ML model predicted primary emotion';
COMMENT ON COLUMN faceemotiondata.confidence_score IS 'ML model confidence level (0-100%)';
