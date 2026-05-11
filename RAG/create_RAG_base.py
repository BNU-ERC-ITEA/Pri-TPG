import argparse
import base64
import json
import mimetypes
import os
import time
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import openai


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "datasets" / "formalgeo7k_v2"
PROBLEMS_DIR = DATA_ROOT / "problems"
DIAGRAMS_DIR = DATA_ROOT / "diagrams"
EMBEDDING_STORE_DIR = Path(__file__).resolve().parent

EMBEDDING_STORE_FILENAMES = {
    "text-image": "embedding_store_text_image.json",
    "image-only": "embedding_store_image_only.json",
    "layout": "embedding_store_layout.json",
    "text-en": "embedding_store_text_en.json",
    "raw-cdl-image": "embedding_store_raw_cdl_image.json",
    "text-image-en": "embedding_store_text_image_en.json",
    "image+text": "embedding_store_image_plus_text.json",
}
EMBEDDING_STORE_PATHS = {
    mode: EMBEDDING_STORE_DIR / filename for mode, filename in EMBEDDING_STORE_FILENAMES.items()
}

PROBLEM_SPLIT_PATH = REPO_ROOT / "datasets" / "problem_split.json"


def load_data_splits() -> Dict[str, Tuple[int, ...]]:
    if not PROBLEM_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Problem split file not found: {PROBLEM_SPLIT_PATH}")
    with open(PROBLEM_SPLIT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "train": tuple(data.get("train", [])),
        "val": tuple(data.get("val", [])),
        "test": tuple(data.get("test", [])),
    }


DATA_SPLITS: Dict[str, Tuple[int, ...]] = load_data_splits()
DEFAULT_SPLIT_FOR_EMBEDDING = "train"
DEFAULT_SPLIT_FOR_RETRIEVAL = "test"

DEFAULT_API_KEY = os.getenv("EMBEDDING_API_KEY")
DEFAULT_API_BASE = os.getenv("EMBEDDING_API_URL")

IMAGE_REQUIRED_MODES = {"image-only", "layout", "raw-cdl-image", "text-image-en", "image+text"}


class RateLimiter:
    """Simple RPM/TPM limiter for embedding requests."""

    def __init__(self, rpm_limit: int = 500, tpm_limit: int = 1_000_000) -> None:
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.request_times: List[float] = []
        self.token_usage: List[Tuple[float, int]] = []

    def _cleanup(self, now: float) -> None:
        one_minute_ago = now - 60.0
        self.request_times = [t for t in self.request_times if t >= one_minute_ago]
        self.token_usage = [p for p in self.token_usage if p[0] >= one_minute_ago]

    def wait_if_needed(self, estimated_tokens: int = 2000) -> None:
        while True:
            now = time.time()
            self._cleanup(now)
            current_rpm = len(self.request_times)
            current_tpm = sum(tokens for _, tokens in self.token_usage)
            if current_rpm < self.rpm_limit and (current_tpm + estimated_tokens) < self.tpm_limit:
                self.request_times.append(now)
                self.token_usage.append((now, estimated_tokens))
                return

            wait_time = 1.0
            if current_rpm >= self.rpm_limit and self.request_times:
                wait_time = max(wait_time, self.request_times[0] + 60.0 - now + 0.1)
            if (current_tpm + estimated_tokens) >= self.tpm_limit and self.token_usage:
                wait_time = max(wait_time, self.token_usage[0][0] + 60.0 - now + 0.1)
            time.sleep(wait_time)


class MultiModalEmbedding:
    """OpenAI-compatible embedding client for text and optional image input."""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        instruction: str | None = None,
        task: str | None = "text-matching",
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self._model = model
        self._instruction = instruction
        self._task = task
        self._client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self._rate_limiter = rate_limiter

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, text_items: Sequence[str], image_path: Optional[str] = None, max_retries: int = 3) -> List[float]:
        text_payload = [x.strip() for x in text_items if isinstance(x, str) and x.strip()]
        if self._instruction:
            text_payload.insert(0, self._instruction.strip())

        inputs: List[Dict[str, str]] = []
        if text_payload:
            inputs.append({"text": "\n".join(text_payload)})

        if image_path:
            with open(image_path, "rb") as f:
                raw = f.read()
            encoded = base64.b64encode(raw).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(image_path)
            mime_type = mime_type or "image/png"
            inputs.append({"image": f"data:{mime_type};base64,{encoded}"})

        if not inputs:
            raise ValueError("Missing text/image content for embedding input")

        if self._rate_limiter:
            estimated_tokens = sum(len(i.get("text", "")) for i in inputs) // 4
            if any("image" in i for i in inputs):
                estimated_tokens += 800
            self._rate_limiter.wait_if_needed(estimated_tokens)

        kwargs: Dict[str, dict] = {}
        if self._task:
            kwargs["extra_body"] = {"task": self._task}

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = self._client.embeddings.create(input=inputs, model=self._model, **kwargs)
                embedding = resp.data[0].embedding
                if not embedding:
                    raise ValueError("Embedding API returned empty embedding")
                return embedding
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    msg = str(exc).lower()
                    time.sleep(15.0 if ("429" in msg or "rate limit" in msg) else 5.0)
                else:
                    break
        raise last_error or RuntimeError("Embedding request failed")


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _coerce_to_strings(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def clean_cdl_entries(entries: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for item in entries:
        text = str(item).strip()
        if not text:
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def extract_shape_tokens(problem_text_en: str) -> List[str]:
    if not problem_text_en:
        return []
    shape_words = [
        "triangle",
        "quadrilateral",
        "rectangle",
        "square",
        "circle",
        "line",
        "segment",
        "angle",
        "parallelogram",
        "trapezoid",
        "polygon",
        "diameter",
        "radius",
        "chord",
        "arc",
    ]
    low = problem_text_en.lower()
    found: List[str] = []
    for w in shape_words:
        if w in low:
            found.append(w)
    return found


def build_problem_payload(
    problem_path: Path,
    image_dir: Path,
    include_text: bool = True,
    include_layout: bool = False,
    layout_only: bool = False,
    mode: str = "text-image",
) -> Optional[Tuple[List[str], List[str], List[str], str, str]]:
    if not problem_path.exists():
        return None

    with open(problem_path, "r", encoding="utf-8") as f:
        problem_data = json.load(f)

    text_items: List[str] = []
    shapes: List[str] = []
    cleaned_cdl: List[str] = []
    layout_desc = ""

    if include_text:
        problem_text_en = str(problem_data.get("problem_text_en", "") or "").strip()

        if mode == "text-en":
            if problem_text_en:
                text_items.append(problem_text_en)
        elif mode == "raw-cdl-image":
            raw_cdl: List[str] = []
            for key in ("construction_cdl", "text_cdl", "image_cdl", "goal_cdl"):
                raw_cdl.extend(_coerce_to_strings(problem_data.get(key)))
            cleaned_cdl = raw_cdl
            text_items.extend(raw_cdl)
        elif mode == "text-image-en":
            shapes = extract_shape_tokens(problem_text_en)
            merged: List[str] = []
            for key in ("construction_cdl", "text_cdl", "image_cdl", "goal_cdl"):
                merged.extend(_coerce_to_strings(problem_data.get(key)))
            cleaned_cdl = clean_cdl_entries(merged)
            if problem_text_en:
                text_items.append(problem_text_en)
            text_items.extend(shapes)
            text_items.extend(cleaned_cdl)
        elif mode == "image+text":
            if problem_text_en:
                text_items.append(problem_text_en)
        else:
            shapes = extract_shape_tokens(problem_text_en)
            merged: List[str] = []
            for key in ("construction_cdl", "text_cdl", "image_cdl", "goal_cdl"):
                merged.extend(_coerce_to_strings(problem_data.get(key)))
            cleaned_cdl = clean_cdl_entries(merged)
            text_items.extend(shapes)
            text_items.extend(cleaned_cdl)

    image_name = problem_data.get("problem_img")
    image_path = ""
    if image_name:
        candidate = image_dir / str(image_name)
        if candidate.exists():
            image_path = str(candidate)

    if include_layout or layout_only:
        # Layout extraction is intentionally removed from solve-critical path.
        layout_desc = ""
        if layout_only:
            return None

    if mode in IMAGE_REQUIRED_MODES and not image_path:
        return None
    if not text_items and not image_path:
        return None
    return text_items, shapes, cleaned_cdl, image_path, layout_desc


def load_embedding_store(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {
            "model": "",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": None,
            "entries": {},
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_embedding_store(path: Path, data: Dict[str, object]) -> None:
    data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _validate_split_name(split: Optional[str]) -> Optional[str]:
    if split is None:
        return None
    name = split.strip().lower()
    if name not in DATA_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return name


def resolve_problem_files(
    numbers: Optional[List[int]],
    start: int,
    limit: Optional[int],
    split: Optional[str] = None,
) -> List[Path]:
    if not PROBLEMS_DIR.exists():
        raise FileNotFoundError(f"Problems directory not found: {PROBLEMS_DIR}")

    normalized_split = _validate_split_name(split)
    if numbers:
        wanted = {str(n) for n in numbers}
        all_files = sorted(PROBLEMS_DIR.glob("*.json"), key=lambda p: int(p.stem))
        return [p for p in all_files if p.stem in wanted]

    if normalized_split:
        split_ids = DATA_SPLITS[normalized_split]
        start_idx = max(start - 1, 0)
        selected = split_ids[start_idx:]
        if limit is not None:
            selected = selected[:limit]
        out: List[Path] = []
        for pid in selected:
            p = PROBLEMS_DIR / f"{pid}.json"
            if p.exists():
                out.append(p)
        return out

    files = sorted(PROBLEMS_DIR.glob("*.json"), key=lambda p: int(p.stem))
    if start > 1:
        files = files[start - 1 :]
    if limit is not None:
        files = files[:limit]
    return files


def build_embeddings(
    embedder: MultiModalEmbedding,
    problem_files: List[Path],
    force: bool = False,
    throttle_seconds: float = 0.5,
    mode: str = "text-image",
    split: str = DEFAULT_SPLIT_FOR_EMBEDDING,
    workers: int = 1,
) -> None:
    if mode not in EMBEDDING_STORE_PATHS:
        raise ValueError(f"Unsupported embedding mode: {mode}")
    if split not in DATA_SPLITS:
        raise ValueError(f"Unsupported split: {split}")

    store_path = EMBEDDING_STORE_PATHS[mode]
    store = load_embedding_store(store_path)
    entries = store.get("entries")
    if not isinstance(entries, dict):
        entries = {}
        store["entries"] = entries

    if not store.get("model"):
        store["model"] = embedder.model_name
    elif store.get("model") != embedder.model_name:
        store["model"] = embedder.model_name
        entries.clear()

    store["mode"] = mode
    store["split"] = split

    processed = 0
    skipped = 0
    errors = 0

    for problem_path in problem_files:
        problem_id = problem_path.stem
        existing = entries.get(problem_id)
        if (not force) and isinstance(existing, dict) and existing.get("mode") == mode and existing.get("embedding"):
            skipped += 1
            continue

        if mode == "layout":
            include_text = False
            include_layout = True
            layout_only = True
        elif mode == "image-only":
            include_text = False
            include_layout = False
            layout_only = False
        else:
            include_text = True
            include_layout = False
            layout_only = False

        payload = build_problem_payload(
            problem_path,
            DIAGRAMS_DIR,
            include_text=include_text,
            include_layout=include_layout,
            layout_only=layout_only,
            mode=mode,
        )
        if not payload:
            skipped += 1
            continue

        text_items, shapes, cleaned_cdl, image_path, layout_desc = payload
        image_arg = None if mode == "text-en" else (image_path if image_path else None)
        if not text_items and not image_arg:
            skipped += 1
            continue

        try:
            embedding = embedder.embed(text_items, image_arg)
        except Exception as exc:
            errors += 1
            print(f"[Error] {problem_id}: embedding failed: {exc}")
            continue

        entries[problem_id] = {
            "problem_id": problem_id,
            "mode": mode,
            "text_items": list(text_items),
            "embedding": embedding,
            "image_path": image_path or "",
            "shapes": shapes,
            "cdl": cleaned_cdl,
            "layout": layout_desc,
        }
        processed += 1
        save_embedding_store(store_path, store)
        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

    save_embedding_store(store_path, store)
    total_entries = len(entries)
    print(
        f"[Embedding Build {mode}/{split}] {store_path.name}: "
        f"processed={processed}, skipped={skipped}, errors={errors}, total_entries={total_entries}"
    )


def parse_bool_flag(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("--force must be true/false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-modal embedding store")
    parser.add_argument("--numbers", nargs="*", type=int, help="Specific problem ids")
    parser.add_argument("--start", type=int, default=1, help="Start index when --numbers is not set")
    parser.add_argument("--limit", type=int, default=5000, help="Max problems to process")
    parser.add_argument(
        "--force", type=parse_bool_flag, nargs="?", const=True, default=False, help="Overwrite existing embeddings"
    )
    parser.add_argument("--throttle", type=float, default=0.5, help="Sleep seconds between requests")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("EMBEDDING_MODEL", "jina-embeddings-v4"),
        help="Embedding model name",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("EMBEDDING_API_URL", DEFAULT_API_BASE),
        help="Embedding API base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("EMBEDDING_API_KEY", DEFAULT_API_KEY),
        help="Embedding API key",
    )
    parser.add_argument("--instruction", type=str, help="Optional instruction prefix")
    parser.add_argument(
        "--mode", choices=tuple(EMBEDDING_STORE_FILENAMES.keys()), default="image-only", help="Embedding mode"
    )
    parser.add_argument(
        "--split",
        choices=tuple(DATA_SPLITS.keys()),
        default=DEFAULT_SPLIT_FOR_EMBEDDING,
        help="Dataset split",
    )
    parser.add_argument("--workers", type=int, default=1, help="Reserved for compatibility")
    parser.add_argument("--rpm-limit", type=int, default=500, help="Requests per minute limit")
    parser.add_argument("--tpm-limit", type=int, default=1_000_000, help="Tokens per minute limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem_files = resolve_problem_files(args.numbers, args.start, args.limit, split=args.split)
    if not problem_files:
        print("No matching problem files found")
        return

    if not args.api_key:
        raise ValueError("Missing embedding API key. Set EMBEDDING_API_KEY or pass --api-key")

    limiter = RateLimiter(rpm_limit=args.rpm_limit, tpm_limit=args.tpm_limit)
    embedder = MultiModalEmbedding(
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        instruction=args.instruction,
        rate_limiter=limiter,
    )
    build_embeddings(
        embedder,
        problem_files,
        force=args.force,
        throttle_seconds=args.throttle,
        mode=args.mode,
        split=args.split,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()