# 🚀 Quick Start - Testing API Capabilities

## Step 1: Set up your environment

```bash
# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure API Keys

1. Copy the environment template:
```bash
cp env.template .env
```

2. Edit `.env` and add your API keys:
```bash
# Open .env in your editor and add the keys you have
nano .env  # or use your preferred editor
```

At minimum, add one of these:
- `RUNWAY_API_KEY` - to test Runway's capabilities
- `LUMA_API_KEY` - to test Luma Dream Machine
- `PIKA_API_KEY` - to test Pika

## Step 3: Run the API Capability Test

```bash
python test_api_capabilities.py
```

This script will:
1. Create a test maze image
2. Try to submit it with a text prompt to each API
3. Report which APIs accept both text and image inputs
4. Save results to `api_test_results.json`

## What We're Testing

For each API, we're checking:
- ✅ Can it accept an image input?
- ✅ Can it accept a text prompt?
- ✅ Can it accept BOTH simultaneously?
- ✅ What are the actual parameter names?
- ✅ What error messages do we get if not supported?

## Expected Outcomes

### If text+image IS supported:
The API will accept both inputs and return a generation ID or start processing.

### If text+image is NOT supported:
The API will return an error like:
- "Invalid parameter: prompt"
- "Cannot specify both text and image"
- "Missing required field"

## Providing API Keys to Test

Please provide any API keys you have access to, and I'll help you:
1. Test the actual capabilities
2. Update our documentation with verified information
3. Implement working integrations for compatible APIs

---

**Note**: Without actual testing, we cannot be certain about any API's capabilities. The documentation might not tell the full story!
