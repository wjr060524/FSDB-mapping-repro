"""Minimal public surface for the extracted FSDB mapping/QAT backend."""

from .quantizer import (
    LEGACY_CODEBOOK,
    FSDBEncoding,
    fsdb_decode,
    fsdb_encode_1to1,
    legacy_codebook_to_canonical_fields,
    legacy_project_codes,
)
__all__ = [
    "LEGACY_CODEBOOK",
    "FSDBEncoding",
    "fsdb_decode",
    "fsdb_encode_1to1",
    "legacy_codebook_to_canonical_fields",
    "legacy_project_codes",
]
