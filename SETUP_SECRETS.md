# 🔐 Secrets Configuration Guide

This guide helps you configure API credentials for the NIFTY Option Screener v6.0.

## 📋 Quick Start

The secrets configuration file has been created at `.streamlit/secrets.toml`. You need to replace the placeholder values with your actual credentials.

## 🔑 Required: Dhan API Credentials

### Step 1: Get Dhan API Credentials

1. Visit [DhanHQ](https://dhanhq.co/)
2. Sign up or log in to your account
3. Navigate to API settings
4. Generate your API credentials:
   - **Client ID**: Your unique client identifier
   - **Access Token**: Your authentication token

### Step 2: Update secrets.toml

Open `.streamlit/secrets.toml` and replace these values:

```toml
# Replace these with your actual Dhan credentials:
DHAN_CLIENT_ID = "your_actual_client_id_here"
DHAN_ACCESS_TOKEN = "your_actual_access_token_here"

[DHAN]
CLIENT_ID = "your_actual_client_id_here"
ACCESS_TOKEN = "your_actual_access_token_here"
```

**Note**: You need to update BOTH the flat format AND the nested `[DHAN]` section with the same values.

## ✅ Testing Your Configuration

After adding your credentials, test the setup:

```bash
streamlit run app.py
```

If configured correctly, you should see:
- ✅ **No error messages** about missing credentials
- ✅ The app loads successfully
- ✅ Live data fetching works

## 🎯 Optional Features

### Telegram Notifications (Optional)

Get trade alerts via Telegram:

1. Create a bot using [@BotFather](https://t.me/BotFather) on Telegram
2. Get your Chat ID using [@userinfobot](https://t.me/userinfobot)
3. Update in `secrets.toml`:

```toml
TELEGRAM_BOT_TOKEN = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
TELEGRAM_CHAT_ID = "987654321"
```

### AI Analysis (Optional)

Enable AI-powered market insights:

1. **Perplexity AI**: Get API key from [Perplexity](https://www.perplexity.ai/)
   ```toml
   PERPLEXITY_API_KEY = "pplx-xxxxxxxxxxxxx"
   ENABLE_AI_ANALYSIS = "true"
   ```

2. **Perplexity AI**: Get API key from [Perplexity](https://www.perplexity.ai/settings/api)
   - Pro subscribers get $5 monthly credits automatically
   - Supports real-time web research and market analysis
   ```toml
   [PERPLEXITY]
   API_KEY = "pplx-xxxxxxxxxxxxx"
   MODEL = "sonar"  # Options: sonar, sonar-pro
   SEARCH_DEPTH = "medium"  # Options: low, medium, high
   ```

3. **NewsData.io**: Get API key from [NewsData.io](https://newsdata.io/)
   ```toml
   [NEWSDATA]
   API_KEY = "pub_xxxxxxxxxxxxx"
   ```

### Database Storage (Optional)

For persistent data storage using Supabase:

1. Create a project at [Supabase](https://supabase.com/)
2. Get your project URL and anon key
3. Update in `secrets.toml`:

```toml
SUPABASE_URL = "https://xxxxxxxxxxxxx.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## 🚀 Deploying to Streamlit Cloud

When deploying to Streamlit Cloud:

1. **DO NOT** upload `secrets.toml` to your repository (it's already in `.gitignore`)
2. Instead, add secrets in Streamlit Cloud:
   - Go to your app dashboard
   - Click **Settings** → **Secrets**
   - Copy the entire content of your local `secrets.toml`
   - Paste it in the secrets editor
   - Save

## 🔒 Security Best Practices

- ✅ **Never commit** `secrets.toml` to git (already in `.gitignore`)
- ✅ **Never share** your API credentials publicly
- ✅ **Rotate keys** periodically for security
- ✅ **Use read-only** API keys when possible
- ✅ **Monitor usage** to detect unauthorized access

## 🐛 Troubleshooting

### Error: "Dhan API credentials not found"

**Solution**: Check that you've replaced ALL placeholder values in both formats:
```toml
DHAN_CLIENT_ID = "your_dhan_client_id"  # ❌ Replace this!
DHAN_CLIENT_ID = "1234567890"           # ✅ Actual value

[DHAN]
CLIENT_ID = "1234567890"                # ✅ Same value here
```

### Error: "Connection failed" or "API Error"

**Possible causes**:
1. Invalid credentials - Double-check your Client ID and Access Token
2. Expired token - Generate a new Access Token from Dhan
3. API rate limit - Wait a few minutes and try again
4. Network issues - Check your internet connection

### Error: "Module not found"

**Solution**: Install required packages:
```bash
pip install -r requirements.txt
```

## 📚 Next Steps

Once configured:

1. ✅ Run the app: `streamlit run app.py`
2. ✅ Verify data is loading correctly
3. ✅ Test optional features (Telegram, AI) if configured
4. ✅ Review the [QUICKSTART.md](QUICKSTART.md) for usage guide
5. ✅ Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if you encounter issues

## 🆘 Getting Help

If you're still having issues:

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review Dhan API documentation
3. Verify all credentials are correct
4. Check the application logs for specific error messages

---

**Remember**: Keep your credentials secure and never share them publicly! 🔐
