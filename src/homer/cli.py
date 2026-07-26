from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from homer.embeddings import QwenEmbeddingProvider
from homer.llm import CerebrasProvider
from homer.pipeline import CorpusEngine


app = typer.Typer(
    name="homer",
    no_args_is_help=True,
    help="Create corpus-grounded literary continuations.",
)


def _engine(corpus: str, require_llm: bool = False) -> CorpusEngine:
    provider = CerebrasProvider() if require_llm else None
    return CorpusEngine(
        corpus=corpus,
        embedding_provider=QwenEmbeddingProvider(),
        graph_extractor=provider,
        community_summarizer=provider,
        story_writer=provider,
    )


@app.command()
def ingest(
    files: Annotated[list[Path], typer.Argument(help="PDF or EPUB files.")],
    corpus: Annotated[str, typer.Option(help="Corpus name.")] = "default",
    extraction_batch_size: Annotated[
        int,
        typer.Option(min=1, max=10, help="Chunks per graph extraction request."),
    ] = 4,
) -> None:
    """Parse, index, and extract a literary graph."""
    engine = _engine(corpus, require_llm=True)
    try:
        stats = engine.ingest(files, extraction_batch_size=extraction_batch_size)
        typer.echo(stats.model_dump_json(indent=2))
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from None
    finally:
        engine.close()


@app.command("inspect")
def inspect_corpus(
    corpus: Annotated[str, typer.Option(help="Corpus name.")] = "default",
) -> None:
    """Show persisted corpus statistics."""
    engine = _engine(corpus)
    try:
        typer.echo(engine.inspect().model_dump_json(indent=2))
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from None
    finally:
        engine.close()


@app.command()
def retrieve(
    prompt: Annotated[str, typer.Option(help="Retrieval query.")],
    corpus: Annotated[str, typer.Option(help="Corpus name.")] = "default",
    limit: Annotated[int, typer.Option(min=1, max=50)] = 14,
) -> None:
    """Print ranked graph facts and passages."""
    engine = _engine(corpus)
    try:
        context = engine.retrieve(prompt, limit=limit)
        typer.echo(context.model_dump_json(indent=2))
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from None
    finally:
        engine.close()


@app.command()
def write(
    prompt: Annotated[str, typer.Option(help="Story request.")],
    corpus: Annotated[str, typer.Option(help="Corpus name.")] = "default",
    output: Annotated[Path | None, typer.Option(help="Optional JSON output.")] = None,
    max_words: Annotated[int, typer.Option(min=200, max=5000)] = 1800,
) -> None:
    """Retrieve grounded context and generate a story."""
    engine = _engine(corpus, require_llm=True)
    try:
        result = engine.write(prompt, max_words=max_words)
        rendered = result.model_dump_json(indent=2)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
            typer.echo(str(output))
        else:
            typer.echo(rendered)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from None
    finally:
        engine.close()


@app.command()
def models() -> None:
    """Print configured model defaults without exposing credentials."""
    typer.echo(
        json.dumps(
            {
                "embedding": "Qwen/Qwen3-Embedding-0.6B",
                "graph": "gpt-oss-120b",
                "writer": "zai-glm-4.7",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
