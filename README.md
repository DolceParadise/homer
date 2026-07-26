# Homer

Homer indexes literary works as both a semantic corpus and a knowledge graph, then
uses the relevant canon and stylistic passages to write grounded continuations.

The prototype is deliberately local: vectors are stored with Qdrant local mode,
the graph is persisted with NetworkX, and Cerebras is used only for structured
graph extraction, community summaries, and story generation.
GLM 4.7 is the preferred writer; production GPT-OSS 120B is used as a fallback
when the GLM preview queue remains unavailable after retries.

## Setup

The project uses the existing `nli` pyenv environment:

```bash
pyenv activate nli
python -m pip install -e ".[dev]"
```

Place `CEREBRAS_API_KEY` in `.env`. Process environment variables override values
loaded from that file. Local books and generated indexes are ignored by Git.

## Build the Sherlock corpus

```bash
homer ingest data/pg1661-images-3.epub data/pg2097-images-3.epub \
  --corpus sherlock
```

Indexing is resumable. Chunk extractions and community summaries are cached beneath
`.homer/sherlock/cache`, so rerunning the command does not repeat completed model
calls.

## Inspect and retrieve

```bash
homer inspect --corpus sherlock

homer retrieve --corpus sherlock \
  --prompt "Write a follow-up to The Adventure of the Blue Carbuncle"
```

## Write a grounded continuation

```bash
homer write --corpus sherlock --max-words 1200 \
  --prompt "What would have happened if Sherlock Holmes had caught Irene Adler?" \
  --output story.json
```

The JSON result contains the story and a grounding report identifying the graph
facts, chapters, and chunks supplied to the writer.

## Tests

```bash
pytest -m "not integration and not live"
pytest -m integration
pytest -m live
```

The default suite parses the two real EPUBs but uses deterministic local providers.
The integration suite loads Qwen3-Embedding-0.6B, and the live suite calls Cerebras
while indexing both books.
