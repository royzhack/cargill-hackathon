# Chatbot Setup Guide

## Quick Setup

### Option 1: Use Config File (Recommended)

1. **Open `chatbot_config.txt`** in your editor
2. **Update the API keys:**
   ```
   TEAM_API_KEY=your-actual-team-api-key
   SHARED_OPENAI_KEY=your-actual-shared-openai-key
   ```
   (Just replace the placeholder values, no quotes needed)
3. **Save the file**
4. **Run the chatbot:**
   ```bash
   python interactive_chatbot.py
   ```

That's it! The chatbot will automatically use the keys from `chatbot_config.txt`.

### Option 2: Use .env File

1. **Create a `.env` file** in the project root (or copy from `.env.example` if it exists)
2. **Add your keys:**
   ```
   TEAM_API_KEY=your-actual-team-api-key
   SHARED_OPENAI_KEY=your-actual-shared-openai-key
   ```
3. **Run the chatbot:**
   ```bash
   python interactive_chatbot.py
   ```

### Option 3: Use Environment Variables

**On macOS/Linux:**
```bash
export TEAM_API_KEY="your-actual-team-api-key"
export SHARED_OPENAI_KEY="your-actual-shared-openai-key"
python interactive_chatbot.py
```

**On Windows (Command Prompt):**
```cmd
set TEAM_API_KEY=your-actual-team-api-key
set SHARED_OPENAI_KEY=your-actual-shared-openai-key
python interactive_chatbot.py
```

**On Windows (PowerShell):**
```powershell
$env:TEAM_API_KEY="your-actual-team-api-key"
$env:SHARED_OPENAI_KEY="your-actual-shared-openai-key"
python interactive_chatbot.py
```

## Priority Order

The chatbot checks for API keys in this order:
1. **`chatbot_config.txt`** (highest priority)
2. **`.env` file**
3. **Environment variables** (lowest priority)

## Verification

To verify your setup is working:

```bash
python load_env.py
```

This will show you which configuration method is being used.

## Files

- **`chatbot_config.txt`** - Main config file (edit this!)
- **`.env`** - Alternative config file (if you prefer .env format)
- **`load_env.py`** - Utility to load config files

## Troubleshooting

**"API keys not found" error:**
- Make sure you've updated `chatbot_config.txt` with actual keys (not the placeholder text)
- Check that the file is saved
- Make sure there are no quotes around the values
- Try using environment variables as a fallback

**"Module not found" error:**
- Install dependencies: `pip install openai`

**Still having issues?**
- Check that your API keys are correct
- Verify the keys don't have extra spaces or quotes
- Make sure you're running from the project root directory
