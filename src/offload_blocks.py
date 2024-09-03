import sys

sys.path.append("..")
import torch.nn as nn

class ModelWithOffloadedBlocks(base):
    def __init__(self, config, _from_pretrained_kwargs, **_duplicate_kwargs):
        num_layers, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent block initialization
        super().__init__(config)
        assert hasattr(self.model, "layers") and isinstance(self.model.layers, (list, tuple, nn.ModuleList))
        self.model.layers = nn.ModuleList([
            OffloadedBlock(config._name_or_path, block_index, **_from_pretrained_kwargs)
            for block_index in range(num_layers)
        ])
        config.num_hidden_layers = num_layers

    @classmethod
    def from_pretrained(cls, name_or_path, **kwargs):
        return super().from_pretrained(name_or_path, **kwargs, _from_pretrained_kwargs=kwargs)

class OffloadedBlock(nn.Module):
    def __init__(self, model_name: str, block_index: int, **kwargs):
        super().__init__()
        self.model_name, self.block_index, self.kwargs = model_name, block_index, kwargs

    def load_from_disk(self):
        return load_pretrained_block(self.model_name, self.block_index, **self.kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"Layer of type {type(self)} cannot be used directly. Call new_layer = offloaded_layer.load_from_disk()")
