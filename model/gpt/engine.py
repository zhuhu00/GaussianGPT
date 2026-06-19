from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

SamplingParams = Tuple[float, Optional[int], Optional[float]]
SamplingParamsResolver = Callable[[int], SamplingParams]


class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(
        self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype
    ):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.token_cache = torch.full(
            (batch_size, seq_len),
            -1,
            device=device,
            dtype=torch.long,
        )
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.token_cache.fill_(-1)

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def store_tokens(self, tokens, start_pos):
        """Store tokens at absolute cache positions [start_pos, start_pos + T)."""
        if tokens.dim() != 2:
            raise ValueError(
                f"tokens must have shape (B, T), got {tuple(tokens.shape)}"
            )
        if tokens.shape[0] != self.batch_size:
            raise ValueError(
                f"tokens batch size mismatch: {tokens.shape[0]} != {self.batch_size}"
            )
        end_pos = start_pos + tokens.shape[1]
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Token cache overflow: end_pos={end_pos} > max_seq_len={self.max_seq_len}"
            )
        self.token_cache[:, start_pos:end_pos] = tokens

    def get_tokens(self, length):
        """Return cached tokens up to `length`."""
        if length < 0 or length > self.max_seq_len:
            raise ValueError(
                f"Requested token length {length} outside [0, {self.max_seq_len}]"
            )
        return self.token_cache[:, :length]

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert (
            self.n_layers == other.n_layers
            and self.n_heads == other.n_heads
            and self.head_dim == other.head_dim
        )
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.token_cache[:, :other_pos] = other.token_cache[:, :other_pos]
        self.cache_seqlens.fill_(other_pos)


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None, top_p=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if top_p is not None:
        assert 0.0 < top_p <= 1.0, "top_p must be in (0, 1]"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_values = torch.topk(logits, k, dim=-1).values[:, [-1]]
        logits = logits.masked_fill(logits < kth_values, float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(
            sorted_indices_to_remove, float("-inf")
        )
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(1, sorted_indices, sorted_logits)
        logits = filtered_logits

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = (
            current_tokens or []
        )  # Current token sequence for this row
        self.completed = False  # Whether this row has completed generation


class Engine:

    def __init__(self, model, eos_token_id):
        self.model = model
        self.eos_token_id = eos_token_id

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        seed=42,
        stop_on_eos: bool = True,
        logits_processor: Optional[Callable[[torch.Tensor, list, int], None]] = None,
        sampling_params_resolver: Optional[SamplingParamsResolver] = None,
    ):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(
            tokens[0], int
        ), "expecting list of ints"
        device = self.model.get_device()
        dtype = next(self.model.parameters()).dtype
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        assistant_end = self.eos_token_id
        bos = self.eos_token_id

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {
            "num_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (
            (len(tokens) + max_tokens)
            if max_tokens is not None
            else self.model.config.sequence_len
        )
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill  # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        import time

        start_wall = time.perf_counter()
        last_wall = start_wall
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            if logits_processor is not None:
                # The initial prefill logits are expanded from a batch=1 tensor.
                # Clone before optional in-place masking to avoid stride-0 aliasing.
                logits = logits.clone()
                logits_processor(logits, row_states, num_generated)

            current_temperature = temperature
            current_top_k = top_k
            current_top_p = top_p
            if sampling_params_resolver is not None:
                (
                    current_temperature,
                    current_top_k,
                    current_top_p,
                ) = sampling_params_resolver(num_generated)

            # Sample the next token for each row
            next_ids = sample_next_token(
                logits,
                rng,
                current_temperature,
                current_top_k,
                current_top_p,
            )  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state
            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                next_token = sampled_tokens[i]
                token_column.append(next_token)
                token_masks.append(1)
                state.current_tokens.append(next_token)
                if stop_on_eos and (next_token == assistant_end or next_token == bos):
                    state.completed = True

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1
            if max_tokens is not None and num_generated >= max_tokens:
                break

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(
                1
            )
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[
                :, -1, :
            ]  # (B, vocab_size)

            if num_generated % 256 == 0:
                now = time.perf_counter()
                if now - last_wall >= 60.0:
                    elapsed = now - start_wall
                    tokens_per_s = num_generated / max(elapsed, 1e-6)
                    if max_tokens is not None:
                        remaining = max_tokens - num_generated
                        eta = remaining / max(tokens_per_s, 1e-6)
                        eta_str = f", ETA ~{eta:.1f}s"
                        total_str = f", est total ~{elapsed + eta:.1f}s"
                    else:
                        eta_str = ""
                        total_str = ""
                    print(
                        f"Sampling progress: {num_generated}"
                        + (f"/{max_tokens}" if max_tokens is not None else "")
                        + f" tokens ({tokens_per_s:.2f} tok/s, "
                        + f"elapsed {elapsed:.1f}s{eta_str}{total_str}).",
                        flush=True,
                    )
                    last_wall = now

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.eos_token_id
        bos = self.eos_token_id
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks
