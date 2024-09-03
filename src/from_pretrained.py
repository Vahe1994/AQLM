"""
The file was taken from Petals project https://github.com/bigscience-workshop/petals/blob/main/src/petals/server/from_pretrained.py
This was done to minimized dependences from 3-rd party repos.
"""
import fcntl
import json
import os
import shutil
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type, Union

import huggingface_hub
import safetensors
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoConfig, MixtralConfig, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.utils import get_file_from_repo

import utils

logger = utils.eval_logger
DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")
DEFAULT_CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", os.environ.get(
                "HF_HOME", f"{os.environ['HOME']}/.cache/huggingface") + "/hub")

def always_needs_auth(model_name: Union[str, os.PathLike, None]) -> bool:
    loading_from_repo = model_name is not None and not os.path.isdir(model_name)
    return loading_from_repo and model_name.startswith("meta-llama/Llama-2-")

def always_needs_auth(model_name: Union[str, os.PathLike, None]) -> bool:
    loading_from_repo = model_name is not None and not os.path.isdir(model_name)
    return loading_from_repo and model_name.startswith("meta-llama/Llama-2-")

def get_model_block(config, layer_idx: int = 0):
    """
    The function to create a model block based on the block class
    kwargs argument **only** is necessary for specific classes, like Mixtral.
    They will not be passed to other block constructors.

    """
    # if config.block_class == WrappedMixtralBlock:
    #     config = PreTrainedModel._autoset_attn_implementation(config)
    #     return config.block_class(config, layer_idx)
    return config.block_class(config)

def resolve_block_dtype(config: PretrainedConfig, dtype: Union[str, torch.dtype]) -> torch.dtype:
    """If dtype is "auto", resolves it using BloomConfig. Returns `dtype` intact otherwise."""
    if dtype not in ("auto", None):
        return dtype
    if config.torch_dtype not in ("auto", None, torch.float32):
        # If config specifies float32, we override it to the default dtype below
        return config.torch_dtype
    return torch.bfloat16

def load_pretrained_block(
    model_name: str,
    block_index: int,
    *,
    config: Optional[PretrainedConfig] = None,
    torch_dtype: Union[torch.dtype, str] = "auto",
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: Optional[str] = None,
    max_disk_space: Optional[int] = None,
) -> nn.Module:
    if config is None:
        config = AutoDistributedConfig.from_pretrained(model_name, use_auth_token=token)
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"
    torch_dtype = resolve_block_dtype(config, torch_dtype)

    with init_empty_weights():
        block = get_model_block(config, layer_idx=block_index)

    block_prefix = f"{config.block_prefix}.{block_index}."
    state_dict = _load_state_dict_from_repo(
        model_name,
        block_prefix,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        max_disk_space=max_disk_space,
    )

    for param_name, _ in block.named_parameters():
        assert param_name in state_dict, f"{param_name} not in state dict"
        param = state_dict[param_name]
        if not str(param.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            param = param.to(torch_dtype)
        set_module_tensor_to_device(block, param_name, "cpu", value=param, dtype=param.dtype)

    logger.info(f"Loaded {model_name} block {block_index}")
    return block


StateDict = Dict[str, torch.Tensor]


def _load_state_dict_from_repo(
    model_name: str,
    block_prefix: str,
    *,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
) -> StateDict:
    if always_needs_auth(model_name) and token is None:
        token = True

    index_file = _find_index_file(model_name, revision=revision, token=token, cache_dir=cache_dir)
    if index_file.endswith(".index.json"):  # Sharded model
        path = get_file_from_repo(model_name, filename=index_file, use_auth_token=token, cache_dir=cache_dir)
        if path is None:
            # _find_index_file() told that a file exists but we can't get it (e.g., it just disappeared)
            raise ValueError(f"Failed to get file {index_file}")

        with open(path) as f:
            index = json.load(f)
        filenames = {
            filename for param_name, filename in index["weight_map"].items() if param_name.startswith(block_prefix)
        }
        if not filenames:
            raise RuntimeError(f"Block {block_prefix}* not found in the index: {index['weight_map']}")
    else:  # Non-sharded model
        filenames = {index_file}
    logger.debug(f"Loading {block_prefix}* from {filenames}")

    state_dict = {}
    for filename in filenames:
        shard_state_dict = _load_state_dict_from_repo_file(
            model_name,
            filename,
            block_prefix=block_prefix,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            max_disk_space=max_disk_space,
        )
        shard_state_dict = {
            param_name[len(block_prefix) :]: param
            for param_name, param in shard_state_dict.items()
            if param_name.startswith(block_prefix)
        }  # Remove unused parameters from memory
        state_dict.update(shard_state_dict)
    return state_dict


INDEX_FILES = ["model.safetensors.index.json", "model.safetensors", "pytorch_model.bin.index.json", "pytorch_model.bin"]


def _find_index_file(
    model_name: str, *, revision: Optional[str] = None, token: Optional[Union[str, bool]] = None, cache_dir: str
) -> str:
    # If we have cached weights (e.g., Pickle from older Petals versions), reuse them
    for filename in INDEX_FILES:
        path = get_file_from_repo(
            model_name,
            filename,
            revision=revision,
            use_auth_token=token,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        if path is not None:
            return filename

    # If we don't, prefer Safetensors when possible
    # (we don't download files here since we can't account for max_disk_space in case of large files)
    for filename in INDEX_FILES:
        with suppress(EntryNotFoundError):
            get_hf_file_metadata(hf_hub_url(model_name, filename, revision=revision), token=token)
            return filename

    raise ValueError(
        f"Repo {model_name} does not contain weights in a supported format: files {INDEX_FILES} do not exist"
    )


def _load_state_dict_from_repo_file(
    model_name: str,
    filename: str,
    *,
    block_prefix: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
    delay: float = 30,
) -> StateDict:
    # First, try to find the weights locally
    try:
        with allow_cache_reads(cache_dir):
            path = get_file_from_repo(
                model_name,
                filename,
                revision=revision,
                use_auth_token=token,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            if path is not None:
                return _load_state_dict_from_local_file(path, block_prefix=block_prefix)
    except Exception:
        logger.warning(f"Cache for file {filename} is corrupted, it will be downloaded again", exc_info=True)

    # If not found, ensure that we have enough disk space to download them (maybe remove something)
    while True:
        try:
            with allow_cache_writes(cache_dir):
                url = hf_hub_url(model_name, filename, revision=revision)
                file_size = get_hf_file_metadata(url, token=token).size
                if file_size is not None:
                    free_disk_space_for(file_size, cache_dir=cache_dir, max_disk_space=max_disk_space)
                else:
                    logger.warning(f"Failed to fetch size of file {filename} from repo {model_name}")

                path = get_file_from_repo(
                    model_name,
                    filename,
                    revision=revision,
                    use_auth_token=token,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
                if path is None:
                    raise RuntimeError(f"File {filename} does not exist in repo {model_name}")
                return _load_state_dict_from_local_file(path, block_prefix=block_prefix)
        except Exception as e:
            logger.warning(f"Failed to load file {filename} from HF Hub (retry in {delay:.0f} sec)", exc_info=True)
            time.sleep(delay)


def _load_state_dict_from_local_file(path: str, *, block_prefix: Optional[str] = None) -> StateDict:
    if path.endswith(".bin"):
        return torch.load(path, map_location="cpu")

    if path.endswith(".safetensors"):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys() if block_prefix is None or key.startswith(block_prefix)}

    raise ValueError(f"Unknown weight format: {path}")


@dataclass
class _ModelClasses:
    config: Type[PretrainedConfig]
    model: Optional[Type[PreTrainedModel]] = None
    model_for_causal_lm: Optional[Type[PreTrainedModel]] = None
    model_for_speculative: Optional[Type[PreTrainedModel]] = None
    model_for_sequence_classification: Optional[Type[PreTrainedModel]] = None


_CLASS_MAPPING = {}  # Populated by petals.models.* subpackages with register_model_classes()


def register_model_classes(*, config: Type[PretrainedConfig], **kwargs):
    assert issubclass(config, PretrainedConfig)
    assert config.model_type not in _CLASS_MAPPING, f"Model type {config.model_type} is already registered"

    _CLASS_MAPPING[config.model_type] = _ModelClasses(config=config, **kwargs)


class _AutoDistributedBase:
    _mapping_field = None  # Should be defined in child classes

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike, None], *args, **kwargs) -> PretrainedConfig:
        if (
            always_needs_auth(model_name_or_path)
            and kwargs.get("token") is None
            and kwargs.get("use_auth_token") is None
        ):
            kwargs["use_auth_token"] = True

        config = AutoConfig.from_pretrained(model_name_or_path, *args, **kwargs)
        if config.model_type not in _CLASS_MAPPING:
            raise ValueError(f"Petals does not support model type {config.model_type}")

        proper_cls = getattr(_CLASS_MAPPING[config.model_type], cls._mapping_field)
        if proper_cls is None:
            raise ValueError(f"Petals does not have {cls.__name__} for model type {config.model_type}")

        return proper_cls.from_pretrained(model_name_or_path, *args, **kwargs)


class DefaultRevisionMixin:
    """
    Petals only supports Falcon loaded in the new in-library format (transformers.FalconModel).
    TII models were recently converted to this format but then reverted back due to compatibility issues.
    We chose to support only the new format since HF staff promised to eventually convert these models
    to the new format again, see https://huggingface.co/tiiuae/falcon-40b/discussions/90#64b4d23bf44fd957492f7602
    Until it happens, we override the default `main` revision for the TII repos with the commit
    pointing out to the model in the in-library format.
    """

    DEFAULT_REVISIONS = {
        "tiiuae/falcon-40b": "f1ba7d328c06aa6fbb4a8afd3c756f46d7e6b232",
        "tiiuae/falcon-40b-instruct": "7475ff8cfc36ed9a962b658ae3c33391566a85a5",
        "tiiuae/falcon-7b": "4e2d06f0a7c6370ebabbc30c6f59377ae8f73d76",
        "tiiuae/falcon-7b-instruct": "f8dac3fff96d5debd43edf56fb4e1abcfffbef28",
    }

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, revision: Optional[str] = None, **kwargs
    ):
        if revision is None and model_name_or_path in cls.DEFAULT_REVISIONS:
            revision = cls.DEFAULT_REVISIONS[model_name_or_path]
            logger.info(f"Loading {model_name_or_path}, revision {revision}")
        return super().from_pretrained(model_name_or_path, *args, revision=revision, **kwargs)


class AutoDistributedConfig(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "config"


class AutoDistributedModel(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "model"


class AutoDistributedModelForCausalLM(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "model_for_causal_lm"


class AutoDistributedSpeculativeModel(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "model_for_speculative"


class AutoDistributedModelForSequenceClassification(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "model_for_sequence_classification"

DEFAULT_CACHE_DIR = os.getenv("PETALS_CACHE", Path(Path.home(), ".cache", "petals"))

BLOCKS_LOCK_FILE = "blocks.lock"


@contextmanager
def _blocks_lock(cache_dir: Optional[str], mode: int):
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    lock_path = Path(cache_dir, BLOCKS_LOCK_FILE)

    os.makedirs(lock_path.parent, exist_ok=True)
    with open(lock_path, "wb+") as lock_fd:
        fcntl.flock(lock_fd.fileno(), mode)
        # The OS will release the lock when lock_fd is closed or the process is killed
        yield


def allow_cache_reads(cache_dir: Optional[str]):
    """Allows simultaneous reads, guarantees that blocks won't be removed along the way (shared lock)"""
    return _blocks_lock(cache_dir, fcntl.LOCK_SH)


def allow_cache_writes(cache_dir: Optional[str]):
    """Allows saving new blocks and removing the old ones (exclusive lock)"""
    return _blocks_lock(cache_dir, fcntl.LOCK_EX)


def free_disk_space_for(
    size: int,
    *,
    cache_dir: Optional[str],
    max_disk_space: Optional[int],
    os_quota: int = 1024**3,  # Minimal space we should leave to keep OS function normally
):
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_info = huggingface_hub.scan_cache_dir(cache_dir)

    available_space = shutil.disk_usage(cache_dir).free - os_quota
    if max_disk_space is not None:
        available_space = min(available_space, max_disk_space - cache_info.size_on_disk)

    gib = 1024**3
    logger.debug(f"Disk space: required {size / gib:.1f} GiB, available {available_space / gib:.1f} GiB")
    if size <= available_space:
        return

    cached_files = [file for repo in cache_info.repos for revision in repo.revisions for file in revision.files]

    # Remove as few least recently used files as possible
    removed_files = []
    freed_space = 0
    extra_space_needed = size - available_space
    for file in sorted(cached_files, key=lambda file: file.blob_last_accessed):
        os.remove(file.file_path)  # Remove symlink
        os.remove(file.blob_path)  # Remove contents

        removed_files.append(file)
        freed_space += file.size_on_disk
        if freed_space >= extra_space_needed:
            break
    if removed_files:
        logger.info(f"Removed {len(removed_files)} files to free {freed_space / gib:.1f} GiB of disk space")
        logger.debug(f"Removed paths: {[str(file.file_path) for file in removed_files]}")

    if freed_space < extra_space_needed:
        raise RuntimeError(
            f"Insufficient disk space to load a block. Please free {(extra_space_needed - freed_space) / gib:.1f} GiB "
            f"on the volume for {cache_dir} or increase --max_disk_space if you set it manually"
        )



class WrappedMixtralBlock(MixtralDecoderLayer):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self._attn_implementation = config._attn_implementation
        self.sliding_window = config.sliding_window
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        batch_size, seq_length, _ = hidden_states.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past

        if past_key_value is not None:
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            _past_key_value = self._reorder_cache_from_bloom(past_key_value, batch_size, past_key_values_length)
            past_key_value = DynamicCache()
            past_key_value.key_cache = [torch.empty(0) for _ in range(self.layer_idx)] + [_past_key_value[0]]
            past_key_value.value_cache = [torch.empty(0) for _ in range(self.layer_idx)] + [_past_key_value[1]]
            past_key_value._seen_tokens = past_key_values_length

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa":
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
                sliding_window=self.sliding_window,
            )

        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=hidden_states.device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs
        )

        if use_cache:
            present_key_value = outputs[-1]
            present_key_value = present_key_value[self.layer_idx]
            present_key_value = self._reorder_cache_to_bloom(present_key_value, batch_size, seq_length_with_past)
            outputs = outputs[:-1] + (present_key_value,)

        return outputs

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        # TODO: Move to mixin
        key_states, value_states = key_value
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.view(*key_states.shape)
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        # TODO: Move to mixin
        key_states, value_states = key_value
        value_states = value_states.view(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.view(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)
