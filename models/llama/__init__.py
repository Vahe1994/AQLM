from block import WrappedLlamaBlock
from config import DistributedLlamaConfig
from model import (
    DistributedLlamaForCausalLM,
    DistributedLlamaForSequenceClassification,
    DistributedLlamaModel,
)
from speculative_model import DistributedLlamaForSpeculativeGeneration
from src.from_pretrained import register_model_classes

register_model_classes(
    config=DistributedLlamaConfig,
    model=DistributedLlamaModel,
    model_for_causal_lm=DistributedLlamaForCausalLM,
    model_for_speculative=DistributedLlamaForSpeculativeGeneration,
    model_for_sequence_classification=DistributedLlamaForSequenceClassification,
)
