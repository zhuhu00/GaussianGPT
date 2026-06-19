from dataclasses import dataclass


@dataclass(frozen=True)
class RopeLayout:
    basis: str
    dense_chunks: bool
    dim: int

    @property
    def is_sequence(self) -> bool:
        return self.basis == "sequence"

    @property
    def is_position(self) -> bool:
        return self.basis == "position"

    @property
    def is_dense(self) -> bool:
        return self.dense_chunks

    @property
    def is_sparse(self) -> bool:
        return not self.dense_chunks


def resolve_rope_layout(*, rope_basis: str, dense_chunks: bool) -> RopeLayout:
    basis = str(rope_basis).lower()
    if basis not in {"sequence", "sequence_1d", "position"}:
        raise ValueError(
            f"Unknown rope_basis '{rope_basis}'. "
            "Expected 'sequence', 'sequence_1d', or 'position'."
        )

    if basis == "sequence_1d":
        # Linear-sequence-position RoPE,
        # available for both dense and sparse data.
        dim = 1
    elif basis == "sequence":
        dim = 1 if dense_chunks else 2
    else:
        dim = 3 if dense_chunks else 4

    return RopeLayout(basis=basis, dense_chunks=bool(dense_chunks), dim=dim)
