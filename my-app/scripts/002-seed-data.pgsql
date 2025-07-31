-- PostgreSQL script for seeding sample data
-- Insert sample care plans
INSERT INTO public.care_plans (name, description, plan_type, monthly_cost, features) VALUES
(
    'Essential Care',
    'Basic monitoring and companionship for independent seniors',
    'essential',
    299.00,
    '{
        "daily_check_ins": true,
        "medication_reminders": true,
        "basic_companionship": true,
        "family_updates": true,
        "emergency_response": false,
        "telehealth_integration": false,
        "24_7_monitoring": false
    }'::jsonb
),
(
    'Complete Care',
    'Comprehensive monitoring and support with emergency response',
    'complete',
    599.00,
    '{
        "daily_check_ins": true,
        "medication_reminders": true,
        "basic_companionship": true,
        "family_updates": true,
        "emergency_response": true,
        "telehealth_integration": true,
        "24_7_monitoring": true,
        "ai_companionship": true,
        "family_dashboard": true
    }'::jsonb
),
(
    'Premium Care',
    'Advanced care with specialized support and custom programs',
    'premium',
    899.00,
    '{
        "daily_check_ins": true,
        "medication_reminders": true,
        "basic_companionship": true,
        "family_updates": true,
        "emergency_response": true,
        "telehealth_integration": true,
        "24_7_monitoring": true,
        "ai_companionship": true,
        "family_dashboard": true,
        "specialized_programs": true,
        "priority_support": true,
        "custom_care_plans": true
    }'::jsonb
);

-- Insert sample demo request (for testing)
INSERT INTO public.demo_requests (
    first_name, 
    last_name, 
    email, 
    phone, 
    relationship, 
    care_needs, 
    preferred_time, 
    questions
) VALUES (
    'John',
    'Smith',
    'john.smith@example.com',
    '(555) 123-4567',
    'child',
    'comprehensive',
    'afternoon',
    'I would like to know more about the emergency response features and how the AI companion works.'
);
