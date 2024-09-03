from block import WrappedBloomBlock
from config import DistributedBloomConfig
from model import (
    DistributedBloomForCausalLM,
    DistributedBloomForSequenceClassification,
    DistributedBloomModel,
)
from src.from_pretrained import register_model_classes

register_model_classes(
    config=DistributedBloomConfig,
    model=DistributedBloomModel,
    model_for_causal_lm=DistributedBloomForCausalLM,
    model_for_sequence_classification=DistributedBloomForSequenceClassification,
)
