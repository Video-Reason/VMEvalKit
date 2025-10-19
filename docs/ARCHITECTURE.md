# VMEvalKit Architecture

VMEvalKit uses a **clean modular architecture** with dynamic loading, designed for scalability and maintainability.

## 🏗️ Core Architecture

### Overview

```
vmevalkit/
├── runner/
│   ├── MODEL_CATALOG.py    # 📋 Pure model registry (40 models, 11 families)
│   └── inference.py        # 🎭 Orchestration with dynamic loading  
├── models/
│   ├── base.py            # 🔧 Abstract ModelWrapper interface
│   ├── luma_inference.py  # LumaInference + LumaWrapper
│   ├── veo_inference.py   # VeoService + VeoWrapper
│   └── ...                # Each provider: Service + Wrapper
```

### Key Principles

1. **Separation of Concerns**: Registry, orchestration, and implementations are separate
2. **Dynamic Loading**: Models loaded on-demand using string module paths  
3. **Family Organization**: Models grouped by provider families
4. **Consistent Interface**: All wrappers inherit from `ModelWrapper` abstract base
5. **No Circular Imports**: String paths eliminate import dependencies

## 📋 Model Catalog System

### Pure Data Registry
`runner/MODEL_CATALOG.py` contains only model definitions - no imports or logic:

```python
LUMA_MODELS = {
    "luma-ray-2": {
        "wrapper_module": "vmevalkit.models.luma_inference",  # String path
        "wrapper_class": "LumaWrapper",                       # Class name  
        "service_class": "LumaInference",                     # Service class
        "model": "ray-2",                                     # Actual model ID
        "description": "Luma Ray 2 - Latest model",
        "family": "Luma Dream Machine"
    }
}
```

### Family Organization
Models are organized into logical families:

```python
MODEL_FAMILIES = {
    "Luma Dream Machine": LUMA_MODELS,
    "Google Veo": VEO_MODELS,
    "WaveSpeed WAN 2.2": WAVESPEED_WAN_22_MODELS,
    # ... 11 families total
}

AVAILABLE_MODELS = {**LUMA_MODELS, **VEO_MODELS, ...}  # Combined registry
```

### Benefits
- **Pure Data**: No imports, logic, or side effects
- **Flexible Loading**: Dynamic imports based on string paths
- **Easy Extension**: Add models without touching other files
- **Family Queries**: `get_models_by_family("Luma Dream Machine")`

## 🎭 Dynamic Loading System

### How It Works

```python
def _load_model_wrapper(model_name: str) -> Type[ModelWrapper]:
    # 1. Look up model in catalog
    config = AVAILABLE_MODELS[model_name] 
    
    # 2. Dynamic import using string path
    module = importlib.import_module(config["wrapper_module"])
    wrapper_class = getattr(module, config["wrapper_class"])
    
    return wrapper_class

# Usage
wrapper_class = _load_model_wrapper("luma-ray-2")
wrapper = wrapper_class(model="ray-2", output_dir="./output")
```

### Benefits
- **On-Demand Loading**: Models only loaded when needed
- **No Circular Imports**: String paths eliminate dependency cycles  
- **Conditional Loading**: Can handle optional dependencies gracefully
- **Runtime Flexibility**: Can load models based on availability

## 🔧 Base Interface System

### ModelWrapper Abstract Base Class

All wrappers inherit from `ModelWrapper` to ensure consistency:

```python
class ModelWrapper(ABC):
    def __init__(self, model: str, output_dir: str = "./data/outputs", **kwargs):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    @abstractmethod
    def generate(self, image_path, text_prompt, duration=8.0, 
                 output_filename=None, **kwargs) -> Dict[str, Any]:
        """Standardized interface for all video generation models."""
        pass
```

### Standardized Return Format

All wrappers must return a dictionary with these exact keys:

```python
{
    "success": bool,           # Whether generation succeeded
    "video_path": str | None,  # Path to generated video file  
    "error": str | None,       # Error message if failed
    "duration_seconds": float, # Time taken for generation
    "generation_id": str,      # Unique identifier
    "model": str,             # Model name used
    "status": str,            # "success", "failed", etc.
    "metadata": Dict[str, Any] # Additional metadata
}
```

### Benefits
- **Type Safety**: Abstract base ensures method signatures
- **Consistency**: All models work identically from user perspective
- **Polymorphism**: Can treat all models uniformly
- **IDE Support**: Better autocomplete and error detection

## 🔄 Inference Flow

### High-Level Flow

```python
1. User calls: runner.run("luma-ray-2", image_path, prompt)
2. InferenceRunner → run_inference()
3. run_inference() → _load_model_wrapper("luma-ray-2")  
4. Dynamic loading: imports vmevalkit.models.luma_inference.LumaWrapper
5. Instantiate wrapper: LumaWrapper(model="ray-2", output_dir=...)
6. Call wrapper.generate() → delegates to LumaInference service
7. Return standardized result dictionary
```

### File Organization Per Model

Each model file is self-contained with:

```python
# vmevalkit/models/provider_inference.py

class ProviderService:
    """Core service implementation for API calls/inference logic"""
    async def generate_video(self, prompt, image_path, **kwargs):
        # Provider-specific implementation
        pass

class ProviderWrapper(ModelWrapper):  
    """VMEvalKit interface wrapper"""
    def __init__(self, model, output_dir, **kwargs):
        super().__init__(model, output_dir, **kwargs)
        self.service = ProviderService(...)
    
    def generate(self, image_path, text_prompt, **kwargs):
        # Standardized interface, delegates to service
        return self.service.generate(...)
```

## 📊 Architecture Benefits

### Before vs After

| **Aspect** | **Before (Monolithic)** | **After (Modular)** |
|------------|-------------------------|---------------------|
| **File Size** | 1,364 lines | 322 lines (inference.py) |
| **Model Registry** | Hardcoded in inference.py | Pure data in MODEL_CATALOG.py |
| **Loading** | Static imports, all at once | Dynamic loading, on-demand |
| **Interface** | No standardization | Abstract ModelWrapper base class |
| **Coupling** | Tightly coupled | Loosely coupled |
| **Extension** | Modify core files | Add to catalog only |
| **Testing** | Hard to mock/isolate | Easy to test components |

### Development Benefits

1. **Easy Model Addition**: Add to catalog, create wrapper, done
2. **Better Maintainability**: Each model file is self-contained (~300-500 lines)
3. **Flexible Dependencies**: Optional models can fail gracefully  
4. **Consistent Interface**: All models work the same way
5. **Type Safety**: Abstract base class catches interface violations
6. **Family Management**: Logical grouping for bulk operations

### Runtime Benefits

1. **Faster Startup**: Models loaded only when needed
2. **Memory Efficiency**: Only active models consume memory
3. **Error Isolation**: Model loading failures don't crash the system
4. **Conditional Loading**: Can handle missing dependencies gracefully

## 🎯 Usage Patterns

### Simple Generation
```python
from vmevalkit.runner.inference import run_inference

result = run_inference(
    model_name="luma-ray-2",
    image_path="maze.png", 
    text_prompt="Solve this maze"
)
```

### Structured Output
```python
from vmevalkit.runner.inference import InferenceRunner

runner = InferenceRunner(output_dir="./experiments")
result = runner.run(
    model_name="veo-3.0-generate",
    image_path="chess.png",
    text_prompt="Show the winning move"
)
# Creates: ./experiments/{domain}_task/{question_id}/{run_id}/
```

### Family Operations
```python
from vmevalkit.runner.MODEL_CATALOG import get_models_by_family

luma_models = get_models_by_family("Luma Dream Machine")
for model_name, config in luma_models.items():
    print(f"{model_name}: {config['description']}")
```

### Model Discovery
```python
runner = InferenceRunner()

# List all models
all_models = runner.list_models()
print(f"Available: {len(all_models)} models")

# List by family  
by_family = runner.list_models_by_family()
for family, models in by_family.items():
    print(f"{family}: {len(models)} models")

# Family statistics
stats = runner.get_model_families() 
print(f"Families: {stats}")
```

This architecture provides a solid foundation for video reasoning evaluation that scales cleanly with new models and providers! 🚀
