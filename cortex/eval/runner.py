"""RAGAS and structural evaluation runner."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cortex.eval.dataset import GoldenQuestion, load_golden_questions
from cortex.retrieval.format import format_retrieval_results
from cortex.settings import Settings
from cortex.synthesis.synthesizer import KBSynthesizer

log = logging.getLogger(__name__)


@dataclass
class QuestionEvalResult:
    id: str
    question: str
    answer: str
    grade_passed: bool
    path_hit: bool
    expected_paths: list[str]
    source_paths: list[str]
    contexts: list[str] = field(default_factory=list)
    trace_id: str = ""
    ragas_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    run_id: str
    timestamp: str
    question_count: int
    path_hit_rate: float
    grade_pass_rate: float
    ragas_averages: dict[str, float]
    results: list[QuestionEvalResult]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "question_count": self.question_count,
            "path_hit_rate": self.path_hit_rate,
            "grade_pass_rate": self.grade_pass_rate,
            "ragas_averages": self.ragas_averages,
            "results": [
                {
                    "id": r.id,
                    "question": r.question,
                    "answer": r.answer,
                    "grade_passed": r.grade_passed,
                    "path_hit": r.path_hit,
                    "expected_paths": r.expected_paths,
                    "source_paths": r.source_paths,
                    "trace_id": r.trace_id,
                    "ragas_scores": r.ragas_scores,
                }
                for r in self.results
            ],
        }


def _path_hit(expected: list[str], source_paths: list[str]) -> bool:
    if not expected:
        return True
    normalized = [p.replace("\\", "/").lower() for p in source_paths]
    for target in expected:
        target_norm = target.replace("\\", "/").lower()
        if any(target_norm in path for path in normalized):
            return True
    return False


def _chunk_contexts(chunks: list, settings: Settings) -> list[str]:
    if not chunks:
        return []
    text = format_retrieval_results(
        chunks,
        include_parent=True,
        max_content_chars=settings.synthesis_context_chars,
    )
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def _run_ragas_batch(
    results: list[QuestionEvalResult],
    golden: list[GoldenQuestion],
    settings: Settings,
) -> dict[str, float]:
    try:
        from datasets import Dataset
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError(
            f"RAGAS eval imports failed ({exc}). Run: uv sync --extra eval"
        ) from exc

    golden_by_id = {g.id: g for g in golden}
    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for result in results:
        g = golden_by_id[result.id]
        rows["question"].append(result.question)
        rows["answer"].append(result.answer)
        rows["contexts"].append(result.contexts or ["no context retrieved"])
        rows["ground_truth"].append(g.ground_truth or result.answer)

    judge_model = settings.eval_llm_model or settings.llm_model
    log.info(
        "ragas_judge",
        extra={"model": judge_model, "timeout_s": settings.eval_llm_timeout_seconds},
    )

    llm = LangchainLLMWrapper(
        ChatOllama(
            base_url=settings.ollama_base_url,
            model=judge_model,
            temperature=0.0,
            timeout=settings.eval_llm_timeout_seconds,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.embed_model,
        )
    )

    dataset = Dataset.from_dict(rows)
    run_config = RunConfig(timeout=settings.eval_llm_timeout_seconds)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    df = scores.to_pandas()
    score_dict = df.mean(numeric_only=True).to_dict()
    for i, result in enumerate(results):
        for col in df.columns:
            if col in {"question", "answer", "contexts", "ground_truth"}:
                continue
            try:
                value = float(df.iloc[i][col])
                if value == value:
                    result.ragas_scores[col] = value
            except (TypeError, ValueError):
                continue

    return {k: float(v) for k, v in score_dict.items() if v == v}


def run_eval(
    settings: Settings | None = None,
    *,
    golden_path: Path | None = None,
    use_ragas: bool | None = None,
    limit: int | None = None,
) -> EvalReport:
    settings = settings or Settings()
    golden = load_golden_questions(golden_path)
    if limit is not None:
        golden = golden[:limit]

    synthesizer = KBSynthesizer(settings)
    results: list[QuestionEvalResult] = []

    for item in golden:
        log.info("eval_question", extra={"id": item.id, "question": item.question})
        synthesis = synthesizer.ask(item.question, limit=settings.rerank_top_k)
        source_paths = [s.relative_path for s in synthesis.sources]
        eval_result = QuestionEvalResult(
            id=item.id,
            question=item.question,
            answer=synthesis.answer,
            grade_passed=synthesis.grade_passed,
            path_hit=_path_hit(item.expected_paths, source_paths),
            expected_paths=item.expected_paths,
            source_paths=source_paths,
            contexts=_chunk_contexts(synthesis.chunks, settings),
            trace_id=synthesis.trace_id,
        )
        results.append(eval_result)

    run_ragas = settings.eval_ragas_enabled if use_ragas is None else use_ragas
    ragas_averages: dict[str, float] = {}
    if run_ragas and results:
        try:
            ragas_averages = _run_ragas_batch(results, golden, settings)
        except RuntimeError as exc:
            log.warning("ragas_skipped: %s", exc)
        except Exception as exc:
            log.warning("ragas_failed", extra={"error": str(exc)})

    path_hits = sum(1 for r in results if r.path_hit)
    grade_passes = sum(1 for r in results if r.grade_passed)
    count = len(results) or 1

    now = datetime.now(UTC)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    return EvalReport(
        run_id=run_id,
        timestamp=now.isoformat(),
        question_count=len(results),
        path_hit_rate=path_hits / count,
        grade_pass_rate=grade_passes / count,
        ragas_averages=ragas_averages,
        results=results,
    )


def save_report(report: EvalReport, settings: Settings | None = None) -> Path:
    settings = settings or Settings()
    out_dir = settings.eval_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.run_id}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path
