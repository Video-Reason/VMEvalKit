# WaveSpeed Wan 2.2 Analysis for VMEvalKit

Based on [WaveSpeed.ai's Wan 2.2 collection](https://wavespeed.ai/collections/wan-2-2)

## 📊 Model Overview

WaveSpeed's Wan 2.2 is described as a "new generation multimodal generative model" with:
- MoE (Mixture of Experts) architecture
- Multiple generation modes including **text-to-video** and **image-to-video**
- "Precise semantic compliance" for complex scenes

## 🔍 Key Models to Test

### Standard Models
1. **wan-2.2/i2v-480p** ($0.15/5sec)
   - Image-to-video generation
   - Question: Does it accept text prompts for guidance?

2. **wan-2.2/t2v-480p** ($0.15/5sec)
   - Text-to-video generation
   - Question: Does it accept reference images?

### Interesting Special Models

3. **wan-2.2/video-edit** ($0.2/5sec)
   - "Edit videos with text prompt easily"
   - Example: "Change her clothing to bikini"
   - This suggests text-based control over visual content

4. **wan-2.2/fun-control** ($0.2/5sec)
   - "Multi-modal conditional inputs"
   - "Control Codes mechanism"
   - Explicitly mentions multi-modal inputs

5. **wan-2.2/animate** ($0.2/5sec)
   - Character animation with "holistic movement and expression replication"
   - May support text guidance with image input

## 🎯 Why This Might Work for VMEvalKit

The documentation mentions:
- **"Multi-modal conditional inputs"** (fun-control model)
- **"Precise semantic compliance"** - understanding complex instructions
- **"Better restoring users' creative intentions"** - following text guidance

## ❓ Critical Questions to Answer

1. **Do i2v models accept text parameters?**
   - Test parameters: `prompt`, `text`, `description`, `guidance`

2. **Do t2v models accept image parameters?**
   - Test parameters: `reference_image`, `input_image`, `style_reference`

3. **What about the special models?**
   - `video-edit`: Seems designed for text-guided editing
   - `fun-control`: Explicitly mentions multi-modal inputs

## 📋 Testing Strategy

```python
# Test 1: Basic i2v with text
payload = {
    "image": maze_image,
    "prompt": "Solve the maze from start to end"
}

# Test 2: fun-control with multi-modal
payload = {
    "image": maze_image,
    "control_codes": "navigate_path",
    "text": "Show solution path"
}

# Test 3: video-edit approach
payload = {
    "input": maze_image,
    "edit_prompt": "Draw the solution path through the maze"
}
```

## 💰 Pricing Comparison

| Model | Resolution | Price |
|-------|------------|-------|
| Wan 2.2 i2v | 480p | $0.15/5sec |
| Wan 2.2 i2v | 720p | $0.30/5sec |
| Wan 2.2 i2v | 1080p | $0.80/5sec |
| Runway gen4_turbo | 720p | $0.05/sec ($0.25/5sec) |
| Runway veo3 | 720p | $0.40/sec ($2.00/5sec) |

WaveSpeed appears competitively priced.

## 🚀 Next Steps

1. Get WaveSpeed API key
2. Test each model variant with text+image inputs
3. Document actual parameter names and capabilities
4. Implement working integration if supported

---

**Note**: All of this needs verification with actual API testing. The documentation suggests promising capabilities, but we need to confirm with real API calls.
