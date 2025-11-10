from typing import List, Tuple
from PIL import Image
import datasets

from .config import EvaluationConfig


def create_batches(dataset: datasets.Dataset, config: EvaluationConfig) -> List[Tuple[List[Image.Image], List[str]]]:
    """
    Create batches of images and labels from dataset.
    
    Args:
        dataset: HuggingFace dataset
        config: Evaluation configuration containing batch_size, column names and label mapping
    
    Returns:
        List of tuples, where each tuple contains (batch_images, batch_labels)
    """
    batches = []
    current_batch_images = []
    current_batch_labels = []
    
    for sample in dataset:
        image = sample[config.image_column]
        
        if config.label_mapping is not None:
            label = config.label_mapping[sample[config.label_column]]
        else:
            label = sample[config.label_column]
        
        current_batch_images.append(image)
        current_batch_labels.append(label)
        
        if len(current_batch_images) == config.batch_size:
            batches.append((current_batch_images, current_batch_labels))
            current_batch_images = []
            current_batch_labels = []
    
    # Add remaining samples as last batch
    if current_batch_images:
        batches.append((current_batch_images, current_batch_labels))
    
    return batches