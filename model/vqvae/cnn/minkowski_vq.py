from typing import Dict, Tuple

import MinkowskiEngine as ME
import torch
from torch import nn
from vector_quantize_pytorch import ResidualLFQ, VectorQuantize
from vector_quantize_pytorch.lookup_free_quantization import LFQ


class SparseLatentVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        num_tokens: int = 1,
        kind: str = "lfq",
    ):
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.num_tokens = num_tokens
        self.kind = kind.lower()

        # TODO just always use the residual wrapper to simplify code
        if self.num_tokens > 1:
            if self.kind != "lfq":
                raise NotImplementedError(
                    f"Residual quantization only wired for kind='lfq', got {kind!r}."
                )
            self.vq = ResidualLFQ(
                dim=self.dim,
                num_quantizers=self.num_tokens,
                codebook_size=self.codebook_size,
                frac_per_sample_entropy=1.0,
                soft_clamp_input_value=10.0,
                experimental_softplus_entropy_loss=True,
                projection_has_bias=False,
            )
        elif self.kind == "lfq":
            self.vq = LFQ(
                dim=self.dim,
                codebook_size=self.codebook_size,
                frac_per_sample_entropy=1.0,
                soft_clamp_input_value=10.0,
                experimental_softplus_entropy_loss=True,
                projection_has_bias=False,
            )
        elif self.kind == "vq":
            self.vq = VectorQuantize(
                dim=self.dim,
                codebook_size=self.codebook_size,
                decay=0.99,
            )
        else:
            raise ValueError(f"Unknown VQ kind {kind!r}; expected 'lfq' or 'vq'.")

    def forward(self, x: ME.SparseTensor) -> Tuple[ME.SparseTensor, Dict]:
        input_dtype = x.F.dtype
        x_F_quantized, idxs, commitment_loss = self.vq(x.F.unsqueeze(0))
        x = ME.SparseTensor(
            features=x_F_quantized.squeeze(0).to(input_dtype),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        return x, {"loss_commit": commitment_loss.sum(), "idxs": idxs}

    def get_output_from_idxs(self, idxs: torch.Tensor) -> torch.Tensor:
        if self.num_tokens > 1:
            return self.vq.get_output_from_indices(idxs)
        if self.kind == "vq":
            return self.vq.get_output_from_indices(idxs.squeeze(0))
        return self.vq.indices_to_codes(idxs.squeeze(0))
