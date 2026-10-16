
from typing import Tuple
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor


class ModelLoader:
    """Handles loading and management of LFM2 audio models."""
    
    def __init__(self, repo_id: str = "LiquidAI/LFM2-Audio-1.5B"):
        """
        Initialize the ModelLoader.
        
        Args:
            repo_id: Hugging Face repository ID for the model
        """
        self.repo_id = repo_id
        self.processor = None
        self.model = None
        
    def load_models(self) -> Tuple[LFM2AudioProcessor, LFM2AudioModel]:
        """
        Load the LFM2 audio processor and model.
        
        Returns:
            Tuple of (processor, model) both in eval mode
        """
        print(f"Loading models from {self.repo_id}...")
        
        self.processor = LFM2AudioProcessor.from_pretrained(self.repo_id).eval()
        self.model = LFM2AudioModel.from_pretrained(self.repo_id).eval()
        
        print("Models loaded successfully")
        return self.processor, self.model
    
    def get_models(self) -> Tuple[LFM2AudioProcessor, LFM2AudioModel]:
        """
        Get the loaded models. Load them if not already loaded.
        
        Returns:
            Tuple of (processor, model)
        """
        if self.processor is None or self.model is None:
            return self.load_models()
        return self.processor, self.model