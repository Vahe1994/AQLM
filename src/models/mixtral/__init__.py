from block import WrappedMixtralBlock
from config import DistributedMixtralConfig
from model import (
    DistributedMixtralForCausalLM,
    DistributedMixtralForSequenceClassification,
    DistributedMixtralModel,
)
from src.from_pretrained import register_model_classes

register_model_classes(
    config=DistributedMixtralConfig,
    model=DistributedMixtralModel,
    model_for_causal_lm=DistributedMixtralForCausalLM,
    model_for_sequence_classification=DistributedMixtralForSequenceClassification,
)
