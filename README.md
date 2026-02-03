---
title: dtSearch Query Builder
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.27.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# NLQ to dtSearch Application

An application that converts search requests written in Natural Language to a dtSearch syntax by making a call to OpenAI's Chat GPT. The app also captures conversation history and allows the user to save the interaction.

## Restricting Access (Hugging Face Spaces)

The app supports username/password authentication for private deployment on Hugging Face Spaces.

### Setup

1. **Set your Space to Private**  
   In your Space: Settings → Visibility → Private.

2. **Add the `BASIC_AUTH_USERS` secret**  
   Settings → Repository secrets → New secret:
   - Name: `BASIC_AUTH_USERS` (Hugging Face standard)
   - Value: comma-separated `username:password` pairs, e.g.  
     `admin:YourSecurePassword,user1:pass1,user2:pass2`
   
   Note: `AUTH_CREDENTIALS` is also supported as an alternative name.

3. Add yourself as admin, e.g. `jkhancock:YourAdminPassword`.

4. When approving new users, add a new `username:password` pair and share the credentials securely.

### Token limits (Hugging Face Spaces)

If follow-up responses are empty or cut off, raise the completion limit via a **Repository secret** (Settings → Repository secrets → New secret):

| Name | Example value | Purpose |
|------|----------------|--------|
| `MAX_COMPLETION_TOKENS` | `2048` or `4096` | Max tokens per model reply (default 2048). |
| `MAX_INPUT_TOKENS` | `12000` or `16000` | Max context size (system + history + message). |

These are optional; the app uses the defaults in code if not set.

### Local Development

If `AUTH_CREDENTIALS` is not set, the app runs without authentication (no login required).


