from transformers import MistralConfig as OrigMistralConfig


class MistralConfig(OrigMistralConfig):
    model_type = "mistral_aqlm"

    def __init__(
        self,
        aqlm: dict[str, int] = {
            "nbits_per_codebook": 16,
            "num_codebooks": 1,
            "out_group_size": 8,
            "in_group_size": 1,
        },
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aqlm = aqlm
