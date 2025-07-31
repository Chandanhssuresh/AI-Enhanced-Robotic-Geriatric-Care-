-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table (extends Supabase auth.users)
CREATE TABLE public.profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone TEXT,
    role TEXT CHECK (role IN ('family_member', 'caregiver', 'admin')) DEFAULT 'family_member',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create care recipients table
CREATE TABLE public.care_recipients (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    date_of_birth DATE,
    gender TEXT CHECK (gender IN ('male', 'female', 'other')),
    address TEXT,
    emergency_contact_name TEXT,
    emergency_contact_phone TEXT,
    medical_conditions TEXT[],
    medications JSONB,
    care_plan_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create care plans table
CREATE TABLE public.care_plans (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    plan_type TEXT CHECK (plan_type IN ('essential', 'complete', 'premium')) NOT NULL,
    monthly_cost DECIMAL(10,2),
    features JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create family_care_recipients junction table
CREATE TABLE public.family_care_recipients (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    family_member_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    care_recipient_id UUID REFERENCES public.care_recipients(id) ON DELETE CASCADE,
    relationship TEXT,
    is_primary_contact BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(family_member_id, care_recipient_id)
);

-- Create health_metrics table
CREATE TABLE public.health_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    care_recipient_id UUID REFERENCES public.care_recipients(id) ON DELETE CASCADE,
    metric_type TEXT NOT NULL, -- 'heart_rate', 'blood_pressure', 'temperature', 'weight', etc.
    value JSONB NOT NULL, -- Store metric values as JSON for flexibility
    unit TEXT,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    recorded_by TEXT DEFAULT 'ai_robot', -- 'ai_robot', 'manual', 'device'
    notes TEXT
);

-- Create activities table
CREATE TABLE public.activities (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    care_recipient_id UUID REFERENCES public.care_recipients(id) ON DELETE CASCADE,
    activity_type TEXT NOT NULL, -- 'medication', 'exercise', 'social', 'emergency', etc.
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (status IN ('scheduled', 'completed', 'missed', 'cancelled')) DEFAULT 'scheduled',
    scheduled_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create alerts table
CREATE TABLE public.alerts (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    care_recipient_id UUID REFERENCES public.care_recipients(id) ON DELETE CASCADE,
    alert_type TEXT NOT NULL, -- 'emergency', 'medication', 'health', 'system'
    severity TEXT CHECK (severity IN ('low', 'medium', 'high', 'critical')) DEFAULT 'medium',
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    is_acknowledged BOOLEAN DEFAULT false,
    acknowledged_by UUID REFERENCES public.profiles(id),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create robot_interactions table
CREATE TABLE public.robot_interactions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    care_recipient_id UUID REFERENCES public.care_recipients(id) ON DELETE CASCADE,
    interaction_type TEXT NOT NULL, -- 'conversation', 'health_check', 'emergency_response', etc.
    content JSONB, -- Store conversation or interaction data
    sentiment_score DECIMAL(3,2), -- AI-analyzed sentiment (-1 to 1)
    duration_seconds INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create demo_requests table
CREATE TABLE public.demo_requests (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT NOT NULL,
    relationship TEXT,
    care_needs TEXT,
    preferred_time TEXT,
    questions TEXT,
    status TEXT CHECK (status IN ('pending', 'scheduled', 'completed', 'cancelled')) DEFAULT 'pending',
    scheduled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.care_recipients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.care_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.family_care_recipients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.health_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.activities ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.robot_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.demo_requests ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
-- Profiles: Users can only see and edit their own profile
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

-- Family members can see care recipients they're associated with
CREATE POLICY "Family members can view their care recipients" ON public.care_recipients
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.family_care_recipients fcr
            WHERE fcr.care_recipient_id = id AND fcr.family_member_id = auth.uid()
        )
    );

-- Similar policies for other tables...
CREATE POLICY "Family members can view health metrics" ON public.health_metrics
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.family_care_recipients fcr
            WHERE fcr.care_recipient_id = health_metrics.care_recipient_id 
            AND fcr.family_member_id = auth.uid()
        )
    );

-- Create indexes for better performance
CREATE INDEX idx_care_recipients_created_at ON public.care_recipients(created_at);
CREATE INDEX idx_health_metrics_care_recipient ON public.health_metrics(care_recipient_id);
CREATE INDEX idx_health_metrics_recorded_at ON public.health_metrics(recorded_at);
CREATE INDEX idx_activities_care_recipient ON public.activities(care_recipient_id);
CREATE INDEX idx_activities_scheduled_at ON public.activities(scheduled_at);
CREATE INDEX idx_alerts_care_recipient ON public.alerts(care_recipient_id);
CREATE INDEX idx_alerts_created_at ON public.alerts(created_at);
CREATE INDEX idx_robot_interactions_care_recipient ON public.robot_interactions(care_recipient_id);
CREATE INDEX idx_robot_interactions_created_at ON public.robot_interactions(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_profiles_updated_at BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_care_recipients_updated_at BEFORE UPDATE ON public.care_recipients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_care_plans_updated_at BEFORE UPDATE ON public.care_plans
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
