from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import AliasChoices, BaseModel, Field, model_validator


class BenchmarkEnvironment(BaseModel):
    model: str
    checkpoint_path: Optional[Path]
    workspace: Path


class InferenceRequest(BaseModel):
    task_id: int
    prompt: Optional[str] = None
    output_tokens: int
    input_ids: Optional[List[int]] = Field(
        alias=AliasChoices("input_ids", "logits"))

    @model_validator(mode="after")
    def verify_prompt_and_logits(self) -> InferenceRequest:
        if self.prompt is None and self.input_ids is None:
            raise ValueError(
                f"Both prompt and input_ids for {self.task_id} are both None.")
        return self


class DatasetMetadata(BaseModel):
    avg_isl: int
    avg_osl: int
    max_isl: int
    max_osl: int
    avg_sequence_length: int
    max_sequence_length: int
    num_requests: int

    def get_summary_for_print(self) -> str:
        return (
            "\n===========================================================\n"
            "= DATASET DETAILS\n"
            "===========================================================\n"
            f"Max Input Sequence Length:\t{self.max_isl}\n"
            f"Max Output Sequence Length:\t{self.max_osl}\n"
            f"Max Sequence Length:\t{self.max_sequence_length}\n"
            f"Target (Average) Input Sequence Length:\t{self.avg_isl}\n"
            f"Target (Average) Output Sequence Length:\t{self.avg_osl}\n"
            f"Number of Sequences:\t{self.num_requests}\n"
            "===========================================================\n"
            f"\n")
