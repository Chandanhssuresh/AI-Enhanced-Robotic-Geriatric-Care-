-- Create Speech Emotion Data table
CREATE TABLE speechemotiondata (
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
CREATE INDEX idx_speechemotiondata_user_id ON speechemotiondata(user_id);
CREATE INDEX idx_speechemotiondata_created_at ON speechemotiondata(created_at);
CREATE INDEX idx_speechemotiondata_predicted_emotion ON speechemotiondata(predicted_emotion);

-- Add comments for documentation
COMMENT ON TABLE speechemotiondata IS 'Stores real-time speech emotion detection data from AI robot sensors';
COMMENT ON COLUMN speechemotiondata.happy_score IS 'Happy speech emotion detection score (0-100%)';
COMMENT ON COLUMN speechemotiondata.sad_score IS 'Sad speech emotion detection score (0-100%)';
COMMENT ON COLUMN speechemotiondata.angry_score IS 'Angry speech emotion detection score (0-100%)';
COMMENT ON COLUMN speechemotiondata.neutral_score IS 'Neutral speech emotion detection score (0-100%)';
COMMENT ON COLUMN speechemotiondata.predicted_emotion IS 'ML model predicted primary speech emotion';
COMMENT ON COLUMN speechemotiondata.confidence_score IS 'ML model confidence level (0-100%)';
