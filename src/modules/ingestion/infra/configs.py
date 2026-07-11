from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceResolverConfig:
    max_file_size_bytes: int


@dataclass(frozen=True, slots=True)
class ByTitleChunkingConfig:
    max_characters: int
    combine_text_under_n_chars: int
    new_after_n_chars: int


@dataclass(frozen=True, slots=True)
class UnstructuredProcessorConfig:
    chunking: ByTitleChunkingConfig
    extraction_timeout_seconds: float
    pdf_strategy: str