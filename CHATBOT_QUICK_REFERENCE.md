# Chatbot Quick Start Guide

## 🎯 What's Ready

✅ **Chatbot is ONLY on Elderly Portal** (`/elderly-portal`)
- Floating bot icon in bottom-right corner
- Supports both text chat and voice input
- Auto voice output for responses

## 🚀 How to Use

### Text Chat
1. Click the floating bot icon
2. Type your message in the input field
3. Press Enter or click "Send"
4. Hear the response spoken aloud

### Voice Chat
1. Click the floating bot icon
2. Click the green **Mic button** to start listening
3. Speak your message
4. Click red **Stop button** when done
5. Message is sent automatically
6. Hear the response spoken aloud

## 🔧 Configuration

All required in `.env`:
```
GEMINI_API_KEY=AIzaSyA7Vgo8xBnQ38gD7oJhlv01MHdhWruMO44
NEXT_PUBLIC_SUPABASE_URL=https://evkkpbbotsrdiuawfqrf.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 🧠 Sample Queries

- "How's my heart rate?"
- "What's my mood?"
- "Any security alerts?"
- "How am I feeling today?"
- "Help me with my medications"

## 📍 Files Modified

| File | Change |
|------|--------|
| `app/elderly-portal/page.tsx` | Added Chatbot component import and render |
| `app/dashboard/page.tsx` | No chatbot icon (as requested) |
| `components/ui/chatbot.tsx` | Existing - handles chat & voice |
| `app/api/chat/route.ts` | Existing - processes messages with Gemini |

## ✨ Features

✅ Real-time chat
✅ Voice input (speech recognition)
✅ Voice output (text-to-speech)
✅ Context-aware responses (from database)
✅ Error handling
✅ Loading states
✅ Auto-speak responses

## 🌐 Access

**Dev Server**: http://localhost:3002
**Elderly Portal**: http://localhost:3002/elderly-portal

## 🐛 If Something Goes Wrong

1. Check `.env` has `GEMINI_API_KEY`
2. Visit `http://localhost:3002/api/chat` - should show JSON response
3. Check browser console for errors
4. Restart dev server: `npm run dev`
5. Clear browser cache and reload

---

**Status**: ✅ Working and ready!
