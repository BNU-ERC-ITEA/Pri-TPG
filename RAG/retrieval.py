import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .create_RAG_base import (
    DIAGRAMS_DIR,
    DATA_SPLITS,
    DEFAULT_SPLIT_FOR_RETRIEVAL,
    EMBEDDING_STORE_PATHS,
    IMAGE_REQUIRED_MODES,
    PROBLEMS_DIR,
    MultiModalEmbedding,
    build_problem_payload,
    cosine_similarity,
    load_embedding_store,
)


def rank_results(
    query_vector: List[float],
    entries: Dict[str, Dict[str, object]],
    top_k: int,
    exclude_id: str,
) -> List[Tuple[str, float, Dict[str, object]]]:
    scored: List[Tuple[str, float, Dict[str, object]]] = []
    for problem_id, entry in entries.items():
        if problem_id == exclude_id:
            continue
        if not isinstance(entry, dict):
            continue
        embedding = entry.get("embedding")
        if not embedding:
            continue
        score = cosine_similarity(query_vector, embedding)
        scored.append((problem_id, score, entry))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def _load_theorem_heads(problem_id: str) -> List[str]:
    json_path = PROBLEMS_DIR / f"{problem_id}.json"
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seqs = data.get("theorem_seqs") or []
    out: List[str] = []
    if isinstance(seqs, list):
        for item in seqs:
            if isinstance(item, str):
                head = item.split("(", 1)[0].strip()
                if head:
                    out.append(head)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve similar FormalGeo problems")
    parser.add_argument("--problem-id", type=int, default=13, help="Query problem id")
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar problems")
    parser.add_argument("--model", type=str, default=os.getenv("EMBEDDING_MODEL", "jina-embeddings-v4"), help="Embedding model name")
    parser.add_argument("--api-base", type=str, default=os.getenv("EMBEDDING_API_URL", "https://aihubmix.com/v1"), help="Embedding API base URL")
    parser.add_argument("--api-key", type=str, default=os.getenv("EMBEDDING_API_KEY", ""), help="Embedding API key")
    parser.add_argument("--instruction", type=str, help="Optional instruction prefix")
    parser.add_argument("--include-self", action="store_true", help="Include query problem in results")
    parser.add_argument(
        "--mode",
        choices=tuple(EMBEDDING_STORE_PATHS.keys()),
        default="text-image-en",
        help="Embedding retrieval mode",
    )
    parser.add_argument(
        "--split",
        choices=tuple(DATA_SPLITS.keys()),
        default=DEFAULT_SPLIT_FOR_RETRIEVAL,
        help="Split used for query id validation",
    )
    return parser.parse_args()


def load_problem_embedding_input(problem_id: str, mode: str) -> Tuple[List[str], List[str], List[str], str, str]:
    problem_path = PROBLEMS_DIR / f"{problem_id}.json"
    if not problem_path.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_path}")

    if mode == "layout":
        include_text = False
        include_layout_local = True
        layout_only = True
    elif mode == "image-only":
        include_text = False
        include_layout_local = False
        layout_only = False
    else:
        include_text = True
        include_layout_local = False
        layout_only = False

    payload = build_problem_payload(
        problem_path,
        DIAGRAMS_DIR,
        include_text=include_text,
        include_layout=include_layout_local,
        layout_only=layout_only,
        mode=mode,
    )
    if not payload:
        raise ValueError(f"Problem {problem_id} has no usable payload")
    return payload


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing embedding API key. Set EMBEDDING_API_KEY or pass --api-key")

    store_path = EMBEDDING_STORE_PATHS.get(args.mode)
    if store_path is None:
        raise ValueError(f"Unsupported embedding mode: {args.mode}")

    store = load_embedding_store(store_path)
    raw_entries = store.get("entries", {})
    if not isinstance(raw_entries, dict) or not raw_entries:
        print(f"Embedding store is empty: {store_path.name}")
        return

    requested_split_ids = DATA_SPLITS.get(args.split)
    if requested_split_ids is None:
        raise ValueError(f"Unsupported split: {args.split}")
    split_id_set = {str(pid) for pid in requested_split_ids}

    problem_id = str(args.problem_id)
    if problem_id not in split_id_set:
        raise ValueError(f"Problem {problem_id} is not in split {args.split}")

    entries: Dict[str, Dict[str, object]] = {
        str(pid): entry
        for pid, entry in raw_entries.items()
        if isinstance(entry, dict)
    }

    text_items, _shapes, _cleaned_cdl, image_path, _layout_desc = load_problem_embedding_input(problem_id, args.mode)
    if args.mode in IMAGE_REQUIRED_MODES and not image_path:
        raise ValueError(f"Problem {problem_id} lacks image required by mode {args.mode}")

    if args.mode == "text-en":
        query_text_items = text_items
        query_image_arg = None
    elif args.mode == "image-only":
        query_text_items = []
        query_image_arg = image_path
    else:
        query_text_items = text_items
        query_image_arg = image_path if image_path else None

    if not query_text_items and not query_image_arg:
        raise ValueError(f"Problem {problem_id} has no usable query payload in mode {args.mode}")

    embedder = MultiModalEmbedding(
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        instruction=args.instruction,
    )

    query_vector = embedder.embed(query_text_items, query_image_arg)
    top_results = rank_results(query_vector, entries, args.top_k, problem_id)
    if args.include_self and problem_id in entries:
        top_results.insert(0, (problem_id, 1.0, entries[problem_id]))

    if not top_results:
        print("No similar results found")
        return

    print(f"Similar problems for {problem_id} (mode={args.mode}):")
    for idx, (pid, score, _entry) in enumerate(top_results, start=1):
        theorem_heads = _load_theorem_heads(pid)
        theorem_preview = ", ".join(theorem_heads[:6]) if theorem_heads else "-"
        print(f"{idx}. problem={pid} score={score:.4f} theorems={theorem_preview}")


if __name__ == "__main__":
    main()
