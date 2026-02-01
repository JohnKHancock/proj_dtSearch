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

2. **Add the `AUTH_CREDENTIALS` secret**  
   Settings → Repository secrets → New secret:
   - Name: `AUTH_CREDENTIALS`
   - Value: comma-separated `username:password` pairs, e.g.  
     `admin:YourSecurePassword,user1:pass1,user2:pass2`

3. Add yourself as admin, e.g. `jkhancock:YourAdminPassword`.

4. When approving new users, add a new `username:password` pair and share the credentials securely.

### Local Development

If `AUTH_CREDENTIALS` is not set, the app runs without authentication (no login required).


