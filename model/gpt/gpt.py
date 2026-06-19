"""
Core GaussianGPT transformer class that operates on tokens.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from utils.gaussian_vqvae_utils import int_cube_root

from .engine import Engine, SamplingParamsResolver
from .nanochat_gpt import GPT, GPTConfig


class NanoGPTGaussianModel(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        codebook_size: int,
        tokens_per_latent: int,
        num_feature_tokens: int,
    ):
        super(NanoGPTGaussianModel, self).__init__()

        self.feature_vocab_size = int(codebook_size)
        self.position_vocab_size = int(
            getattr(config, "position_vocab_size", self.feature_vocab_size)
        )
        side_length = int_cube_root(self.position_vocab_size)
        if side_length**3 != self.position_vocab_size:
            raise ValueError("position_vocab_size must be a perfect cube (k^3).")
        self.shared_vocab = bool(getattr(config, "shared_vocab", False))
        if self.shared_vocab:
            self.feature_token_offset = 0
            base = max(self.position_vocab_size, self.feature_vocab_size)
            self.eos_token_id = base
            self.pad_token_id = base + 1
            self.vocab_size = base + 2
        else:
            self.feature_token_offset = self.position_vocab_size
            self.eos_token_id = self.feature_token_offset + self.feature_vocab_size
            self.pad_token_id = self.eos_token_id + 1
            self.vocab_size = self.pad_token_id + 1

        self.config = config

        chunk_shape = getattr(config, "chunk_shape", None)
        if isinstance(chunk_shape, int):
            chunk_shape = [chunk_shape] * 3
        self.gpt_config = GPTConfig(
            sequence_len=config.n_ctx,
            vocab_size=self.vocab_size,
            n_layer=config.gpt_size.n_layer,
            n_head=config.gpt_size.n_head,
            n_kv_head=getattr(config.gpt_size, "n_kv_head", config.gpt_size.n_head),
            n_embd=config.gpt_size.n_embd,
            rope_basis=getattr(config, "rope_basis"),
            rope_bos_coord=getattr(config, "rope_bos_coord", (-1, -1, -1)),
            learned_pos_embed=bool(getattr(config, "learned_pos_embed", False)),
            dense_chunks=getattr(config, "dense_chunks", False),
            dense_chunk_shape=chunk_shape,
            dense_chunk_order=getattr(config, "chunk_order", "xyz"),
            dense_num_features=int(num_feature_tokens),
            sparse_tokens_per_latent=int(tokens_per_latent),
            sparse_num_position_tokens=(
                0
                if getattr(config, "dense_chunks", False)
                else int(getattr(config, "num_position_tokens", 1))
            ),
            sparse_position_hard_constraints=bool(
                getattr(config, "sparse_position_hard_constraints", False)
            ),
            value_embed_every_n_layers=getattr(config, "value_embed_every_n_layers", 2),
            sparse_position_vocab_size=self.position_vocab_size,
            sparse_feature_vocab_size=self.feature_vocab_size,
            sparse_feature_token_offset=self.feature_token_offset,
            sparse_shared_vocab=self.shared_vocab,
        )

        self.transformer = GPT(self.gpt_config)
        self.transformer.init_weights()

        self._engine = Engine(self.transformer, self.eos_token_id)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        return_logits=False,
    ):
        assert input_ids.dtype == torch.long
        assert int(input_ids.min()) >= 0 and int(input_ids.max()) < self.vocab_size

        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape")
            attention_mask = attention_mask.to(
                device=input_ids.device, dtype=torch.bool
            )

        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids shape")

        max_len = (
            self.gpt_config.sequence_len + 1
            if labels is not None
            else self.gpt_config.sequence_len
        )
        if input_ids.shape[1] > max_len:
            if labels is not None:
                if attention_mask is not None:
                    lengths = attention_mask.long().sum(dim=1)
                    max_start = torch.clamp(lengths.min() - max_len, min=0)
                else:
                    max_start = input_ids.shape[1] - max_len
                start = (
                    torch.randint(
                        0, max_start + 1, (1,), device=input_ids.device
                    ).item()
                    if self.training and max_start > 0
                    else 0
                )
                input_ids = input_ids[:, start : start + max_len]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, start : start + max_len]
                if labels is not None:
                    labels = labels[:, start : start + max_len]
            else:
                input_ids = input_ids[:, -max_len:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -max_len:]
                if labels is not None:
                    labels = labels[:, -max_len:]

        if labels is None:
            logits = self.transformer(input_ids, targets=None)
            return logits if return_logits else logits

        if input_ids.shape[1] < 2:
            raise ValueError("input_ids must have length >= 2 for next-token loss")

        inp = input_ids[:, :-1]
        target = labels[:, 1:].contiguous()
        # target[j] corresponds to labels position j+1, so use the shifted mask
        # to avoid supervising padded targets after EOS in variable-length batches.
        attn = attention_mask[:, 1:] if attention_mask is not None else None
        if attn is not None:
            target = target.masked_fill(~attn, -1)

        if return_logits:
            logits = self.transformer(inp, targets=None)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=-1,
            )
            return loss, logits

        loss = self.transformer(inp, targets=target)

        return loss

    @torch.no_grad()
    def sample_sequence(
        self,
        seq_len,
        batch_size=1,
        temperature=1.0,
        top_k=None,
        top_p=None,
        partials=None,
        stop_on_eos=True,
        seed: Optional[int] = None,
        sampling_params_resolver: Optional[SamplingParamsResolver] = None,
    ):
        assert partials is None, "Partials not implemented with NanoChat yet"

        device = next(self.parameters()).device
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        prompt = [self.eos_token_id]
        results = [[self.eos_token_id] for _ in range(batch_size)]
        for token_column, _ in self._engine.generate(
            prompt,
            num_samples=batch_size,
            max_tokens=seq_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_on_eos=stop_on_eos,
            seed=seed,
            sampling_params_resolver=sampling_params_resolver,
        ):
            for i, token in enumerate(token_column):
                results[i].append(token)
        tokens = torch.tensor(results, dtype=torch.long, device=device)
        return tokens[:, 1:]

    @torch.no_grad()
    def sample_sequence_with_prompt(
        self,
        prompt_tokens,
        max_new_tokens,
        num_samples=1,
        temperature=1.0,
        top_k=None,
        top_p=None,
        stop_on_eos=True,
        seed: Optional[int] = None,
        logits_processor: Optional[Callable[[torch.Tensor, list, int], None]] = None,
        sampling_params_resolver: Optional[SamplingParamsResolver] = None,
    ):
        if torch.is_tensor(prompt_tokens):
            prompt_tokens = prompt_tokens.tolist()
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        prompt = [self.eos_token_id]
        if prompt_tokens:
            prompt.extend(int(token) for token in prompt_tokens)

        results, _ = self._engine.generate_batch(
            prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_on_eos=stop_on_eos,
            seed=seed,
            logits_processor=logits_processor,
            sampling_params_resolver=sampling_params_resolver,
        )
        if results and results[0] and results[0][0] == self.eos_token_id:
            results = [seq[1:] for seq in results]
        device = next(self.parameters()).device
        return torch.tensor(results, dtype=torch.long, device=device)
