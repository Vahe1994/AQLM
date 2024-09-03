from block import WrappedFalconBlock
from config import DistributedFalconConfig
from model import (
    DistributedFalconForCausalLM,
    DistributedFalconForSequenceClassification,
    DistributedFalconModel,
)
from src.from_pretrained import register_model_classes

register_model_classes(
    config=DistributedFalconConfig,
    model=DistributedFalconModel,
    model_for_causal_lm=DistributedFalconForCausalLM,
    model_for_sequence_classification=DistributedFalconForSequenceClassification,
)
