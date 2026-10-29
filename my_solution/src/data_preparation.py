from pathlib import Path
import json
from . import config
from .data_classes import CITATION_PATTERN, evaluation_case, instruction_case, knowledge_chunk, knowledge_document


def load_document_chunks(directory: Path) -> list[knowledge_chunk]:
    """Split every .md file in `directory` into one chunk per paragraph, with metadata."""

    chunks: list[knowledge_chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        # Validated document: ID from the file name, title from the "# Title" heading, text without the heading
        document = knowledge_document.from_markdown(path)
        blocks = [block.strip() for block in document.text.split("\n\n")]
        # Keep text paragraphs only: drop empty blocks and any sub-headings
        body = [block for block in blocks if block and not block.startswith("#")]
        for doc_chunk_part, paragraph in enumerate(body):
            chunks.append(knowledge_chunk(
                # Join the wrapped lines of the paragraph into one line of text
                text=paragraph.replace("\n", " "),
                document_title=document.title,
                document_name=document.id,
                # Running number across all documents: 0, 1, 2, ...
                chunk_id=len(chunks),
                # Position of this chunk within its own document, restarting at 0 for each document
                doc_chunk_part=doc_chunk_part,
            ))
    return chunks


def load_evaluation_cases(path: Path) -> list[evaluation_case]:
    """Load the evaluation cases from a JSON file."""
    # Load the JSON list of dicts, then validate each dict as an evaluation_case
    file = json.loads(path.read_text(encoding="utf-8"))
    return [evaluation_case.model_validate(case) for case in file]


def remove_response_style(record: dict) -> dict:
    """Drop the [response-style] passage from a training record's context and reference_ids."""
    style_id = config.RESPONSE_STYLE_FILE.stem
    lines = record["context"].split("\n")
    record["context"] = "\n".join(line for line in lines if not line.startswith(f"[{style_id}]"))
    record["reference_ids"] = [source for source in record["reference_ids"] if source != style_id]
    return record


def load_training_cases(path: Path) -> list[instruction_case]:
    """Load the training cases from a JSONL file, without the response-style passages."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [remove_response_style(record) for record in records]
    # An answer that cites [response-style] has lost its evidence, so the case is dropped
    style_id = config.RESPONSE_STYLE_FILE.stem
    records = [record for record in records if style_id not in CITATION_PATTERN.findall(record["answer"])]
    return [instruction_case.model_validate(record) for record in records]


def normalise(text: str) -> str:
    """Lower-case and collapse whitespace, so small formatting differences do not hide a duplicate."""
    return " ".join(text.lower().split())


def assert_evaluation_instruction_difference(evaluation_cases: list[evaluation_case], instruction_cases: list[instruction_case]):
    """Raise an error if any evaluation case is also in the training set (same ID or same question)."""
    evaluation_ids = {case.id for case in evaluation_cases}
    instruction_ids = {case.id for case in instruction_cases}
    overlap = evaluation_ids & instruction_ids
    if overlap:
        raise ValueError(f"evaluation cases {sorted(overlap)} are also in the training set")

    instruction_questions = {normalise(case.question) for case in instruction_cases}
    leaked = [case.id for case in evaluation_cases if normalise(case.question) in instruction_questions]
    if leaked:
        raise ValueError(f"evaluation cases {leaked} have the same question as a training example")
    
    
def assert_cited_source_exists(evaluation_cases: list[evaluation_case], instruction_cases: list[instruction_case], knowledge_chunks: list[knowledge_chunk]):
    """Raise an error if any evaluation case cites a source that is not in the knowledge base."""
    # Extract a set of all unique source IDs from the knowledge base chunks (add the response-style since it is also a valid source)
    knowledge_source_ids = {chunk.document_name for chunk in knowledge_chunks}
    knowledge_source_ids.add(config.RESPONSE_STYLE_FILE.stem)
    for case in evaluation_cases:
        cited = set(case.expected_sources)
        if not cited.issubset(knowledge_source_ids):
            raise ValueError(
                f"evaluation case {case.id} cites {sorted(cited - knowledge_source_ids)}, which are not in the knowledge base"
            )
    for case in instruction_cases:
        cited = set(case.reference_ids)
        if not cited.issubset(knowledge_source_ids):
            raise ValueError(
                f"instruction case {case.id} cites {sorted(cited - knowledge_source_ids)}, which are not in the knowledge base"
            )
            
            
def ingest_data():
    """Load and validate all the data files, returning them as Python objects."""
    knowledge_chunks = load_document_chunks(config.KNOWLEDGE_BASE_DIR)
    evaluation_cases = load_evaluation_cases(config.EVALUATION_FILE)
    instruction_cases = load_training_cases(config.TRAINING_FILE)

    assert_cited_source_exists(evaluation_cases, instruction_cases, knowledge_chunks)
    assert_evaluation_instruction_difference(evaluation_cases, instruction_cases)

    return knowledge_chunks, evaluation_cases, instruction_cases