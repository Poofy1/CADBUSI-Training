import importlib

def get_model(model_type, **kwargs):
    """
    Factory function to load the appropriate model class.
    
    Args:
        model_type: String identifier for model (e.g., 'MIL', 'ABMIL', 'GenSCL')
        **kwargs: Arguments to pass to the model constructor
    
    Returns:
        Instantiated model
    """
    try:
        module = importlib.import_module(f'archs.model_{model_type}')
        model_class = module.Embeddingmodel
        return model_class(**kwargs)
    except (ImportError, AttributeError) as e:
        available_models = ["MIL", "MIL_ins", "MIL_sali", "ABMIL", 
                            "instances",
                          "GenSCL", "FanoGan", "INS", "baggett_transformer"]
        raise ValueError(f"Model type '{model_type}' not found. Available models: {available_models}") from e