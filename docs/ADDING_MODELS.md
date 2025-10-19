# Adding Models to VMEvalKit

Quick guide for integrating video generation models into VMEvalKit's **modular inference system** with dynamic loading.

## ⚡ Requirements

✅ **MUST support: Image + Text → Video** (essential for reasoning evaluation)  
✅ **Inherit from ModelWrapper**: Use abstract base class for consistency  
✅ **Unified interface**: `generate(image_path, text_prompt, duration, output_filename, **kwargs)`  
✅ **Parameter separation**: Constructor for config, generate() for runtime inputs

## 🎯 Integration Steps

### API Models (3 steps)
1. Create `vmevalkit/models/{provider}_inference.py` with Service + Wrapper
2. Register in `vmevalkit/runner/MODEL_CATALOG.py` (pure data registry)
3. Update `vmevalkit/models/__init__.py`

### Open-Source Models (4 steps)  
1. Add submodule: `git submodule add {repo_url} submodules/{ModelName}`
2. Create wrapper with subprocess/direct calls
3. Register in MODEL_CATALOG.py
4. Update __init__.py

---

## 📝 Implementation Guide

### API Models

**File**: `vmevalkit/models/{provider}_inference.py`

```python
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import time
from .base import ModelWrapper

class {Provider}Service:
    def __init__(self, model_id: str, api_key: Optional[str] = None, **kwargs):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("{PROVIDER}_API_KEY")
        
    async def generate_video(self, prompt: str, image_path: str, **kwargs):
        # Your API implementation:
        # 1. Prepare request with image + text
        # 2. Make API call  
        # 3. Return (video_bytes, metadata)
        video_bytes = b""  # API response
        metadata = {"request_id": "id", "model": self.model_id}
        return video_bytes, metadata

class {Provider}Wrapper(ModelWrapper):
    """VMEvalKit wrapper implementing standardized interface"""
    def __init__(self, model: str, output_dir: str = "./data/outputs", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, output_dir, **kwargs)
        self.service = {Provider}Service(model_id=model, api_key=api_key, **kwargs)
    
    def generate(self, image_path: Union[str, Path], text_prompt: str, 
                 duration: float = 8.0, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate video - implements ModelWrapper.generate()"""
        start_time = time.time()
        
        if not Path(image_path).exists():
            return {
                "success": False, 
                "error": f"Image not found: {image_path}", 
                "video_path": None,
                "duration_seconds": 0,
                "generation_id": "error",
                "model": self.model,
                "status": "failed",
                "metadata": {"prompt": text_prompt, "image_path": str(image_path)}
            }
        
        video_bytes, metadata = asyncio.run(
            self.service.generate_video(prompt=text_prompt, image_path=str(image_path), **kwargs)
        )
        
        if video_bytes:
            if not output_filename:
                output_filename = f"{self.model.replace('/', '_')}_{int(time.time())}.mp4"
            video_path = self.output_dir / output_filename
            video_path.write_bytes(video_bytes)
            
            return {
                "success": True, 
                "video_path": str(video_path),
                "error": None,
                "duration_seconds": time.time() - start_time,
                "generation_id": metadata.get("request_id", "unknown"),
                "model": self.model,
                "status": "success",
                "metadata": {"prompt": text_prompt, "image_path": str(image_path), **metadata}
            }
        
        return {
            "success": False, 
            "error": "No video generated", 
            "video_path": None,
            "duration_seconds": time.time() - start_time,
            "generation_id": "failed",
            "model": self.model,
            "status": "failed",
            "metadata": {"prompt": text_prompt, "image_path": str(image_path)}
        }
```

### Open-Source Models

**File**: `vmevalkit/models/{model}_inference.py`

```python
import sys, subprocess, time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .base import ModelWrapper

# Add submodule to path
{MODEL}_PATH = Path(__file__).parent.parent.parent / "submodules" / "{ModelName}"
sys.path.insert(0, str({MODEL}_PATH))

class {Model}Service:
    def __init__(self, model_id: str = "default", output_dir: str = "./data/outputs", **kwargs):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if not {MODEL}_PATH.exists():
            raise FileNotFoundError(f"Run: git submodule update --init {ModelName}")

    def _run_subprocess_inference(self, image_path: str, text_prompt: str, 
                                  output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """For models requiring subprocess execution"""
        start_time = time.time()
        output_path = self.output_dir / (output_filename or f"{self.model_id}_{int(time.time())}.mp4")
        
        cmd = [sys.executable, str({MODEL}_PATH / "inference.py"),
               "--prompt", text_prompt, "--image", str(image_path), "--output", str(output_path)]
        
        try:
            result = subprocess.run(cmd, cwd=str({MODEL}_PATH), 
                                  capture_output=True, text=True, timeout=300)
            success = result.returncode == 0 and output_path.exists()
            return {
                "success": success, 
                "video_path": str(output_path) if success else None,
                "error": result.stderr if not success else None,
                "duration_seconds": time.time() - start_time,
                "generation_id": f"{self.model_id}_{int(time.time())}",
                "model": self.model_id,
                "status": "success" if success else "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": image_path,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False, 
                "error": "Timeout", 
                "video_path": None,
                "duration_seconds": time.time() - start_time,
                "generation_id": "timeout",
                "model": self.model_id,
                "status": "failed",
                "metadata": {"text_prompt": text_prompt, "image_path": image_path}
            }
    
    def generate(self, image_path: Union[str, Path], text_prompt: str, 
                 duration: float = 8.0, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if not Path(image_path).exists():
            return {
                "success": False, 
                "error": f"Image not found: {image_path}", 
                "video_path": None,
                "duration_seconds": 0,
                "generation_id": "error",
                "model": self.model_id,
                "status": "failed",
                "metadata": {"text_prompt": text_prompt, "image_path": str(image_path)}
            }
        
        return self._run_subprocess_inference(str(image_path), text_prompt, 
                                             output_filename=output_filename, **kwargs)

class {Model}Wrapper(ModelWrapper):
    def __init__(self, model: str, output_dir: str = "./data/outputs", **kwargs):
        super().__init__(model, output_dir, **kwargs)
        self.service = {Model}Service(model_id=model, output_dir=output_dir, **kwargs)
    
    def generate(self, image_path: Union[str, Path], text_prompt: str, 
                 duration: float = 8.0, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        return self.service.generate(image_path, text_prompt, duration, output_filename, **kwargs)
```

## 🔌 Registration

### 1. Register in `vmevalkit/runner/MODEL_CATALOG.py`

```python
# Add to appropriate family section
{PROVIDER}_MODELS = {
    "{provider}-model-name": {
        "wrapper_module": "vmevalkit.models.{provider}_inference",
        "wrapper_class": "{Provider}Wrapper",
        "service_class": "{Provider}Service",
        "model": "actual-model-id",
        "description": "Brief description", 
        "family": "{Provider}"
    }
}

# Add to combined registries
AVAILABLE_MODELS = {
    **EXISTING_MODELS,
    **{PROVIDER}_MODELS
}

MODEL_FAMILIES = {
    **EXISTING_FAMILIES,
    "{Provider}": {PROVIDER}_MODELS
}
```

**Key Features:**
- **String Module Paths**: Enable dynamic loading without circular imports
- **Family Organization**: Models grouped by provider for easy management
- **Pure Data**: No imports or logic in catalog, just model definitions
- **Flexible Loading**: Models loaded on-demand via dynamic imports

### 2. Update `vmevalkit/models/__init__.py`

```python
from .{provider}_inference import {Provider}Service, {Provider}Wrapper

__all__ = [
    ...,
    "{Provider}Service", "{Provider}Wrapper"
]
```

## ✅ Testing

### Quick Test
```python
from vmevalkit.runner.inference import InferenceRunner, run_inference

# Test direct function
result = run_inference(
    model_name="your-model-name", 
    image_path="test.png",
    text_prompt="test prompt",
    output_dir="./data/outputs"
)
print(f"Success: {result.get('success')}")
print(f"Video: {result.get('video_path')}")

# Test via runner (recommended)
runner = InferenceRunner(output_dir="./test_output")
result = runner.run(
    model_name="your-model-name",
    image_path="test.png", 
    text_prompt="test prompt"
)
```

### Validate Dynamic Loading
```python
from vmevalkit.runner.inference import _load_model_wrapper

# Test that your wrapper loads correctly
wrapper_class = _load_model_wrapper("your-model-name")
print(f"Loaded: {wrapper_class.__name__}")

# Check catalog registration
from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS
print(f"Model registered: {'your-model-name' in AVAILABLE_MODELS}")

# Verify family organization
from vmevalkit.runner.MODEL_CATALOG import get_models_by_family
family_models = get_models_by_family("YourProvider")
print(f"Family has {len(family_models)} models")
```

## 💡 Key Rules

**DO:**
- **Inherit from ModelWrapper**: `class YourWrapper(ModelWrapper)`
- **Call super().__init__()**: In wrapper constructor
- **Use string paths in catalog**: Enable dynamic loading
- **Return standardized format**: Follow the exact dict structure with all required keys
- **Validate inputs**: Check image exists, handle errors gracefully
- **Follow naming**: `{Provider}Wrapper`, `{provider}_inference.py`
- **Separate concerns**: Service (API logic) + Wrapper (VMEvalKit interface)

**DON'T:**
- **Skip base class**: All wrappers must inherit from `ModelWrapper`
- **Use direct imports in catalog**: Use string module paths instead
- **Mix service logic in wrapper**: Keep wrapper thin, delegate to service
- **Pass runtime args to constructor**: Constructor for config, generate() for inputs
- **Forget required return fields**: Must include success, video_path, error, duration_seconds, etc.

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Check registration in `MODEL_CATALOG.py` |
| Import error | Verify `wrapper_module` path in catalog |
| Dynamic loading fails | Check `wrapper_class` name matches actual class |
| Wrapper inheritance error | Ensure wrapper inherits from `ModelWrapper` |
| Missing return fields | Follow exact return format from `ModelWrapper.generate()` |
| Submodule missing | `git submodule update --init --recursive` |
| API auth fails | Set environment variable: `{PROVIDER}_API_KEY` |

## 🏗️ Architecture Overview

The new modular architecture separates concerns cleanly:

```
MODEL_CATALOG.py    →  Pure data registry (string module paths)
inference.py        →  Orchestration with dynamic loading  
models/*.py         →  Service + Wrapper pairs (self-contained)
models/base.py      →  Abstract ModelWrapper interface
```

**Benefits:**
- **Dynamic Loading**: Models loaded on-demand, no circular imports
- **Easy Extension**: Add to catalog, no core file changes needed
- **Consistent Interface**: ModelWrapper ensures all models work the same way
- **Family Organization**: Models grouped by provider for management
- **Maintainable**: Each model file is self-contained (~300-500 lines)

---

Ready to add your model? Follow the steps above and you'll be integrated with VMEvalKit's modular architecture! 🎯