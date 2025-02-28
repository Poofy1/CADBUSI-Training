import importlib
import os

def get_model(model_type, **kwargs):
    """
    Factory function to load the appropriate model class.
    
    Args:
        model_type: Complete module name (e.g., 'model_MIL', 'model_ABMIL', 'model_GenSCL')
        **kwargs: Arguments to pass to the model constructor
    
    Returns:
        Instantiated model
    """
    # Ensure the name starts with 'model_'
    if not model_type.startswith('model_'):
        raise ValueError(f"Model name must start with 'model_', got '{model_type}'")
    
    try:
        module = importlib.import_module(f'archs.{model_type}')
        model_class = module.Embeddingmodel
        return model_class(**kwargs)
    except (ImportError, AttributeError) as e:
        # Dynamically get available models by listing files in the archs directory
        archs_dir = os.path.dirname(os.path.abspath(__file__))
        available_models = []
        
        for filename in os.listdir(archs_dir):
            if filename.startswith('model_') and filename.endswith('.py'):
                available_models.append(filename[:-3])  # Remove .py extension
        
        available_models.sort()
        models_str = ", ".join(f"'{model}'" for model in available_models)
        raise ValueError(f"Model '{model_type}' not found or has no Embeddingmodel class. Available models: {models_str}") from e