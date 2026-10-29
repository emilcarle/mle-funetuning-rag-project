import re
from pathlib import Path
from typing import Annotated
import pydantic
from . import config

# Source IDs are the knowledge-base filenames without ".md", e.g. "account-security".
SourceId = Annotated[str, pydantic.StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
NonEmptyStr = Annotated[str, pydantic.StringConstraints(strip_whitespace=True, min_length=1)]

CITATION_PATTERN = re.compile(r"\[([a-z0-9]+(?:-[a-z0-9]+)*)\]")
CONTEXT_TAG_PATTERN = re.compile(r"^\[([a-z0-9]+(?:-[a-z0-9]+)*)\]", re.MULTILINE)

# Shared settings: reject unknown fields and make records read-only after validation.
RECORD_CONFIG = pydantic.ConfigDict(extra="forbid", frozen=True)


class instruction_case(pydantic.BaseModel):
    """One supervised training example from data/training/instruction_examples.jsonl."""

    model_config = RECORD_CONFIG

    id: NonEmptyStr
    # Empty only for an abstention example without evidence (checked below)
    reference_ids: list[SourceId]
    context: str
    question: NonEmptyStr
    answer: NonEmptyStr

    @pydantic.field_validator("reference_ids")
    @classmethod
    def reference_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError(f"reference_ids contains duplicates: {value}")
        return value

    @pydantic.model_validator(mode="after")
    def context_and_citations_match_references(self) -> "instruction_case":
        references = set(self.reference_ids)

        if not references and self.answer != config.ABSTAIN_MESSAGE:
            raise ValueError("a case without evidence must answer with the abstention message")

        context_tags = set(CONTEXT_TAG_PATTERN.findall(self.context))
        if context_tags != references:
            raise ValueError(
                f"context source tags {sorted(context_tags)} do not match "
                f"reference_ids {sorted(references)}"
            )

        cited = set(CITATION_PATTERN.findall(self.answer))
        if not cited <= references:
            raise ValueError(
                f"answer cites {sorted(cited - references)}, which are not in reference_ids"
            )
        return self


class evaluation_case(pydantic.BaseModel):
    """One held-out benchmark question from data/evaluation/cases.json."""

    model_config = RECORD_CONFIG

    id: NonEmptyStr
    question: NonEmptyStr
    expected_sources: list[SourceId]
    required_terms: list[NonEmptyStr]
    forbidden_terms: list[NonEmptyStr]
    should_abstain: bool

    @pydantic.model_validator(mode="after")
    def labels_match_abstention(self) -> "evaluation_case":
        if self.should_abstain:
            if self.expected_sources or self.required_terms:
                raise ValueError(
                    "an abstention case cannot have expected_sources or required_terms"
                )
        elif not self.expected_sources:
            raise ValueError("an answerable case needs at least one expected source")
        return self


class knowledge_document(pydantic.BaseModel):
    """One approved policy source from data/knowledge_base/."""

    model_config = RECORD_CONFIG

    id: SourceId
    title: NonEmptyStr
    text: NonEmptyStr

    @classmethod
    def from_markdown(cls, path: Path) -> "knowledge_document":
        """Build a document from a Markdown file: ID from the filename, title from the '# ' heading."""
        raw = Path(path).read_text(encoding="utf-8").strip()
        first_line, _, body = raw.partition("\n")
        if not first_line.startswith("# "):
            raise ValueError(f"{path} must start with a '# ' title heading")
        return cls(id=Path(path).stem, title=first_line.removeprefix("# "), text=body)


class knowledge_chunk(pydantic.BaseModel):
    """One paragraph of a knowledge document, with metadata for retrieval."""

    model_config = RECORD_CONFIG

    text: NonEmptyStr
    document_title: NonEmptyStr
    document_name: SourceId
    chunk_id: int
    doc_chunk_part: int
    

class embedded_chunk(knowledge_chunk):
    embedding: list[float]


class retrieved_chunk(knowledge_chunk):
    """A chunk returned by retrieval, with its similarity score to the query."""

    score: float
