from __future__ import annotations

import copy
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from openai import OpenAI

from RAG import retrieval as rag_retrieval
from RAG.create_RAG_base import (
    DATA_SPLITS,
    EMBEDDING_STORE_PATHS,
    IMAGE_REQUIRED_MODES,
    MultiModalEmbedding,
    cosine_similarity,
    load_embedding_store,
)


ONE_STEP_PROMPT_TEMPLATE = (
	"You should propose the MOST promising theorem to apply next for the current geometry problem. DO NOT output any object names or argument bindings.\n"
	"**Critical**: Candidate theorems are ranked by similarity and executability (readiness scores), but this does NOT mean you should pick the first one. Analyze the goal, current state, and choose the theorem that best advances toward solving the problem.\n"
	"Follow these rules:\n"
	"- Propose EXACTLY 1 theorem choice with its specific branch.\n"
	"- Only use theorem names from 'candidate_theorems'.\n"
	"- REQUIRED: The theorem MUST include branch index as theorem_name(branch_index), even if the theorem has only 1 branch.\n"
	"  Example: sine_theorem(1), parallel_judgment_corresponding_angle(2), vertical_angle(1)\n"
	"  ALWAYS write theorem_name(branch_index), NEVER write theorem_name alone.\n"
	"- Do NOT provide any geometry objects, wrapped entities, or argument lists; the solver will enumerate and apply valid object instances automatically.\n"
	"- **IMPORTANT: Use branch-level guidance fields to choose execution order:**\n"
	"  - 'branch_readiness': prioritize highly ready branches.\n"
	"  - 'tpg_guidance': check recommended next branches and unlocking effects.\n"
	"  - 'tpg_state_insights': follow high-value recommended paths/chains.\n"
	"- If your choice fails, the system has a recovery mechanism that will retry with alternative strategies.\n"
	"- Base your decision on 'problem', 'state', 'applied_history', 'branch_readiness', 'tpg_guidance', and 'tpg_state_insights'.\n"
	"- Pay special attention to 'state.delta_from_last_step' which shows what the previous theorem actually produced.\n"
	"- Do NOT repeat a call equivalent to one in 'applied_calls' or 'applied_history'.\n"
	"- In recovery mode, avoid the most recent failed calls in 'recent_failure'.\n"
	"Return STRICT JSON only: {\"calls\": [\"theorem_name(branch_index)\"]}. No commentary.\n"
	"Remember: Predict ONE theorem with branch index!\n"
	"Do your best to solve the problem!\n"
)

LLM_API_KEYS = [
    os.environ.get("OPENAI_API_KEY", "")
]
LLM_API_BASE = os.environ.get("OPENAI_BASE_URL", "https://aihubmix.com/v1")
_current_api_key_index = 0
THEOREM_CACHE: Dict[str, Any] = {}
THEOREM_TPG_CACHE: Dict[str, Any] = {}


def extract_theorem_name(call: str) -> str:
    if not call:
        return ""
    idx = call.find("(")
    return call[:idx].strip() if idx != -1 else call.strip()


def extract_theorem_name_and_branch(call: str) -> Optional[Tuple[str, int]]:
    if not call or "(" not in call:
        return None
    name, rest = call.split("(", 1)
    args = rest.rstrip(")")
    first = args.split(",", 1)[0].strip() if args else ""
    if first.isdigit():
        return (name.strip(), int(first))
    return (name.strip(), 1)


def _split_call_arguments(text: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in text:
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def is_wrapped_entity(token: str) -> bool:
    if not token or "(" not in token or not token.endswith(")"):
        return False
    prefix = token.split("(", 1)[0]
    return prefix and prefix[0].isupper() and prefix.isalpha()


def normalize_theorem_call(call: str) -> str:
    if not call:
        return ""
    return "".join(ch for ch in call if not ch.isspace())


def ensure_theorem_call_has_branch(call: str, default_branch: int = 1) -> str:
    text = normalize_theorem_call(call)
    if not text:
        return text
    if "(" not in text:
        return f"{text}({default_branch})"
    name, rest = text.split("(", 1)
    inner = rest[:-1] if rest.endswith(")") else rest
    first = inner.split(",", 1)[0].strip() if inner else ""
    if first.isdigit():
        return text
    return f"{name}({default_branch},{inner})" if inner else f"{name}({default_branch})"


def _split_signature(signature: str) -> Tuple[str, List[str]]:
    signature = signature.strip()
    if "(" in signature and signature.endswith(")"):
        head, remainder = signature.split("(", 1)
        arg_text = remainder[:-1]
        arguments = [arg.strip() for arg in arg_text.split(",") if arg.strip()]
        return head.strip(), arguments
    return signature, []


def compute_signature_details(name: str, info: Dict[str, Any]) -> Dict[str, Any]:
    signature = info.get("signature") or f"{name}(...)"
    _, argument_roles = _split_signature(signature)
    forms_list = info.get("forms") or []
    has_forms = bool(forms_list)

    if argument_roles:
        valid_call = f"{name}(branch_index, {', '.join(argument_roles)})"
    else:
        valid_call = f"{name}(branch_index)"

    return {
        "signature": signature,
        "argument_roles": argument_roles,
        "valid_call_example": valid_call,
        "has_forms": has_forms,
        "forms": forms_list,
    }


def build_theorem_prompt_entry(
    name: str,
    info: Dict[str, Any],
    valid_indices: Optional[Set[int]] = None,
    branch_scores: Optional[Dict[int, int]] = None,
    branch_details: Optional[Dict[int, List[str]]] = None,
) -> Dict[str, Any]:
    details = compute_signature_details(name, info)
    argument_roles = details["argument_roles"]
    valid_example = details["valid_call_example"]
    has_forms = details["has_forms"]

    if valid_indices is not None and has_forms:
        details["forms"] = [f for f in details["forms"] if f["index"] in valid_indices]

    if branch_scores and has_forms:
        readiness_labels = {3: "ready", 2: "partial", 1: "unknown", 0: "low"}
        for form in details["forms"]:
            idx = form["index"]
            if idx in branch_scores:
                score = branch_scores[idx]
                label = readiness_labels.get(score, "")
                if score == 2 and branch_details and idx in branch_details:
                    missing = branch_details[idx]
                    if missing:
                        label += f" (needs: {', '.join(missing)})"
                form["readiness"] = label
                form["readiness_score"] = score

    note_parts: List[str] = [f"Valid call example: {valid_example}."]
    if has_forms:
        note_parts.append("Replace 'branch_index' with a form id from 'forms'.")
    else:
        note_parts.append("This theorem has one branch; use branch_index=1.")
    if argument_roles:
        note_parts.append("Use raw tokens matching the argument_roles order.")
    note_parts.append("Never invent new theorem names or change argument order.")
    note = " ".join(note_parts)
    return {
        "name": name,
        "gdl_signature": details["signature"],
        "forms": details["forms"],
        "argument_roles": argument_roles,
        "valid_call_example": valid_example,
        "note": note,
    }


def _build_problem_dict(problem: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": problem.get("problem_id"),
        "text_en": problem.get("problem_text_en", ""),
        "construction_cdl": problem.get("construction_cdl", []),
        "text_cdl": problem.get("text_cdl", []),
        "image_cdl": problem.get("image_cdl", []),
        "goal_cdl": problem.get("goal_cdl", ""),
    }


def _reduce_theorem_call_for_llm(call: str) -> str:
    call = normalize_theorem_call(call)
    if "(" not in call:
        return call
    name, rest = call.split("(", 1)
    first = rest.rstrip(")").split(",", 1)[0].strip()
    return f"{name}({first})" if first.isdigit() else name


def _redact_text_remove_objects(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"([A-Z][A-Za-z0-9_]*)\(([^)]*)\)", r"\1()", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_redacted_problem_dict(problem: Dict[str, Any]) -> Dict[str, Any]:
    pd = _build_problem_dict(problem)
    return {
        "id": pd["id"],
        "text_en": _redact_text_remove_objects(pd["text_en"]),
        "construction_cdl": [_redact_text_remove_objects(str(x)) for x in pd["construction_cdl"]],
        "text_cdl": [_redact_text_remove_objects(str(x)) for x in pd["text_cdl"]],
        "image_cdl": [_redact_text_remove_objects(str(x)) for x in pd["image_cdl"]],
        "goal_cdl": _redact_text_remove_objects(pd["goal_cdl"]),
    }


def get_goal_type(problem: Any) -> str:
    try:
        goal = problem.get("parsed_cdl", {}).get("goal", {}) if isinstance(problem, dict) else {}
        item = goal.get("item")
        if isinstance(item, list) and item:
            head = str(item[0])
            if "Angle" in head:
                return "Angle"
            if any(x in head for x in ["Length", "Distance", "Perimeter"]):
                return "Length"
            if "Area" in head:
                return "Area"
    except Exception:
        pass
    return "Unknown"


def load_problem_embedding_input_adapter(problem_id: str, mode: str, problems_dir: Path, diagrams_dir: Path) -> Tuple[List[str], List[str], List[str], Optional[Path], str]:
    old_p = rag_retrieval.PROBLEMS_DIR
    old_d = rag_retrieval.DIAGRAMS_DIR
    try:
        rag_retrieval.PROBLEMS_DIR = problems_dir
        rag_retrieval.DIAGRAMS_DIR = diagrams_dir
        text_items, shapes, cleaned_cdl, image_path, layout_desc = rag_retrieval.load_problem_embedding_input(problem_id, mode)
        return text_items, shapes, cleaned_cdl, Path(image_path) if image_path else None, layout_desc
    finally:
        rag_retrieval.PROBLEMS_DIR = old_p
        rag_retrieval.DIAGRAMS_DIR = old_d


def load_problem(problem_id: str, problems_dir: Path) -> Dict[str, Any]:
    p = problems_dir / f"{problem_id}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_theorem_sequences(problem_id: str, problems_dir: Path) -> List[Tuple[str, int]]:
    key = f"{problem_id}_branch_seqs"
    if key in THEOREM_CACHE:
        return THEOREM_CACHE[key]
    data = load_problem(problem_id, problems_dir)
    seqs = data.get("theorem_seqs") or []
    out: List[Tuple[str, int]] = []
    for item in seqs:
        if not isinstance(item, str) or "(" not in item:
            continue
        name, rest = item.split("(", 1)
        first = rest.rstrip(")").split(",", 1)[0].strip()
        out.append((name, int(first) if first.isdigit() else 1))
    THEOREM_CACHE[key] = out
    return out


def load_theorem_tpg(problem_id: str, problems_dir: Path) -> Dict[str, List[str]]:
    data = load_problem(problem_id, problems_dir)
    tpg = data.get("theorem_seqs_dag") or {}
    out: Dict[str, List[str]] = {}
    for k, succ in tpg.items():
        nk = "START" if k == "START" else extract_theorem_name(k)
        vals = []
        for s in succ:
            if isinstance(s, str):
                h = extract_theorem_name(s)
                if h and h not in vals:
                    vals.append(h)
        out.setdefault(nk, [])
        for v in vals:
            if v not in out[nk]:
                out[nk].append(v)
    return out


def load_theorem_tpg_with_branches(problem_id: str, problems_dir: Path) -> Dict[Tuple[str, int], List[Tuple[str, int]]]:
    data = load_problem(problem_id, problems_dir)
    tpg = data.get("theorem_seqs_dag") or {}
    out: Dict[Any, List[Tuple[str, int]]] = {}
    for k, succ in tpg.items():
        nk: Any = "START" if k == "START" else extract_theorem_name_and_branch(k)
        if nk is None:
            continue
        out.setdefault(nk, [])
        for s in succ:
            if isinstance(s, str):
                sb = extract_theorem_name_and_branch(s)
                if sb and sb not in out[nk]:
                    out[nk].append(sb)
    return out


def merge_theorem_tpgs(tpgs: List[Dict[str, List[str]]], candidate_theorems: Set[str]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {"START": []}
    for tpg in tpgs:
        for k, succ in tpg.items():
            fs = [s for s in succ if s in candidate_theorems]
            if k == "START":
                for s in fs:
                    if s not in merged["START"]:
                        merged["START"].append(s)
            elif k in candidate_theorems:
                merged.setdefault(k, [])
                for s in fs:
                    if s not in merged[k]:
                        merged[k].append(s)
    return merged


def merge_theorem_tpgs_with_edge_weights(tpgs: List[Dict[str, List[str]]], candidate_theorems: Set[str]) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float]]:
    merged_tpg: Dict[str, List[str]] = {"START": []}
    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    num_tpgs = len(tpgs) if tpgs else 1

    for tpg in tpgs:
        for key, successors in tpg.items():
            filtered_successors = [s for s in successors if s in candidate_theorems]
            if key == "START":
                for s in filtered_successors:
                    if s not in merged_tpg["START"]:
                        merged_tpg["START"].append(s)
                    edge_counts[("START", s)] += 1
            elif key in candidate_theorems:
                if key not in merged_tpg:
                    merged_tpg[key] = []
                for s in filtered_successors:
                    if s not in merged_tpg[key]:
                        merged_tpg[key].append(s)
                    edge_counts[(key, s)] += 1

    edge_weights: Dict[Tuple[str, str], float] = {
        edge: count / num_tpgs for edge, count in edge_counts.items()
    }
    return merged_tpg, edge_weights


def compute_tpg_dynamic_priorities(tpgs: List[Dict[str, List[str]]], candidate_theorems: Set[str]) -> Tuple[Dict[str, float], Dict[str, int]]:
    if not tpgs:
        return {}, {}

    start_counts: Dict[str, int] = defaultdict(int)
    unlock_sets: Dict[str, Set[str]] = defaultdict(set)

    for tpg in tpgs:
        for theorem in tpg.get("START", []):
            if theorem in candidate_theorems:
                start_counts[theorem] += 1

        for prereq, successors in tpg.items():
            if prereq == "START":
                continue
            if prereq in candidate_theorems:
                for succ in successors:
                    if succ in candidate_theorems:
                        unlock_sets[prereq].add(succ)

    num_tpgs = len(tpgs)
    common_prefix_scores: Dict[str, float] = {}
    for theorem, count in start_counts.items():
        frequency = count / num_tpgs
        if frequency >= 0.2:
            common_prefix_scores[theorem] = frequency * 15.0

    unlock_scores: Dict[str, int] = {
        theorem: len(successors)
        for theorem, successors in unlock_sets.items()
    }

    return common_prefix_scores, unlock_scores


def merge_theorem_tpgs_with_edge_weights_branches(tpgs: List[Dict[Any, List[Tuple[str, int]]]], candidate_branches: Set[Tuple[str, int]]) -> Tuple[Dict[Any, List[Tuple[str, int]]], Dict[Tuple[Any, Tuple[str, int]], float]]:
    merged_tpg: Dict[Any, List[Tuple[str, int]]] = {"START": []}
    edge_counts: Dict[Tuple[Any, Tuple[str, int]], int] = defaultdict(int)
    num_tpgs = len(tpgs) if tpgs else 1

    for tpg in tpgs:
        for k, succ in tpg.items():
            fs = [s for s in succ if s in candidate_branches]
            if k == "START":
                for s in fs:
                    if s not in merged_tpg["START"]:
                        merged_tpg["START"].append(s)
                    edge_counts[("START", s)] += 1
            elif k in candidate_branches:
                merged_tpg.setdefault(k, [])
                for s in fs:
                    if s not in merged_tpg[k]:
                        merged_tpg[k].append(s)
                    edge_counts[(k, s)] += 1

    edge_weights: Dict[Tuple[Any, Tuple[str, int]], float] = {
        edge: count / num_tpgs for edge, count in edge_counts.items()
    }
    return merged_tpg, edge_weights


def compute_tpg_dynamic_priorities_branches(tpgs: List[Dict[Any, List[Tuple[str, int]]]], candidate_branches: Set[Tuple[str, int]]) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], int]]:
    start_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    unlock_sets: Dict[Tuple[str, int], Set[Tuple[str, int]]] = {br: set() for br in candidate_branches}

    for tpg in tpgs:
        if "START" in tpg:
            for branch in tpg["START"]:
                if branch in candidate_branches:
                    start_counts[branch] += 1

        for prereq, successors in tpg.items():
            if prereq == "START" or prereq not in candidate_branches:
                continue
            for succ in successors:
                if succ in candidate_branches:
                    unlock_sets[prereq].add(succ)

    num_tpgs = len(tpgs) if tpgs else 1
    common_prefix_scores: Dict[Tuple[str, int], float] = {}
    for branch, count in start_counts.items():
        frequency = count / num_tpgs
        if frequency >= 0.2:
            common_prefix_scores[branch] = frequency * 15.0

    unlock_scores: Dict[Tuple[str, int], int] = {
        branch: len(successors) for branch, successors in unlock_sets.items()
    }

    return common_prefix_scores, unlock_scores


def prepare_embedding_store(mode: str, split: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Path]:
    store_path = EMBEDDING_STORE_PATHS[mode]
    store = load_embedding_store(store_path)
    entries = {str(k): v for k, v in (store.get("entries") or {}).items() if isinstance(v, dict)}
    requested = {str(pid) for pid in DATA_SPLITS[split]}
    filtered = {k: v for k, v in entries.items() if k in requested}
    return (filtered or entries), store, store_path


def rank_results(query_vector: Sequence[float], entries: Dict[str, Dict[str, Any]], top_k: int, exclude_id: str) -> List[Tuple[str, float, Dict[str, Any]]]:
    scored: List[Tuple[str, float, Dict[str, Any]]] = []
    for pid, entry in entries.items():
        if pid == exclude_id:
            continue
        emb = entry.get("embedding")
        if not isinstance(emb, Sequence):
            continue
        try:
            vec = [float(x) for x in emb]
        except Exception:
            continue
        scored.append((pid, cosine_similarity(query_vector, vec), entry))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k] if top_k > 0 else scored


def compute_theorem_complexity(theorem_name: str, parsed_theorem_GDL: Dict[str, Any]) -> float:
    theorem_def = parsed_theorem_GDL.get(theorem_name)
    if not theorem_def:
        return 2.0

    all_prereq_counts = []
    has_complex_predicates = False

    body = theorem_def.get("body", {})
    for _, gpl in body.items():
        products = len(gpl.get("products", []))
        logic = len(gpl.get("logic_constraints", []))
        algebra = len(gpl.get("algebra_constraints", []))
        prereq_count = products + logic + algebra
        all_prereq_counts.append(prereq_count)

        for predicate, _ in gpl.get("products", []):
            if predicate in {
                "Polygon",
                "Cocircular",
                "Parallelogram",
                "Trapezoid",
                "CongruentBetweenTriangle",
                "SimilarBetweenTriangle",
                "MirrorCongruentBetweenTriangle",
                "MirrorSimilarBetweenTriangle",
            }:
                has_complex_predicates = True
                break

        if algebra > 0:
            has_complex_predicates = True

    if not all_prereq_counts:
        return 2.0

    avg_prereqs = sum(all_prereq_counts) / len(all_prereq_counts)
    if has_complex_predicates:
        avg_prereqs += 1.0

    return avg_prereqs


def filter_candidates_by_precheck(candidates: List[Union[str, Tuple[str, int]]], problem: Any, *, max_check_attempts: int = 50) -> Tuple[Dict[str, Set[int]], Dict[Tuple[str, int], int], Dict[Tuple[str, int], List[str]]]:
    valid_branches: Dict[str, Set[int]] = defaultdict(set)
    readiness_scores: Dict[Tuple[str, int], int] = {}
    branch_details: Dict[Tuple[str, int], List[str]] = defaultdict(list)

    if candidates and isinstance(candidates[0], tuple):
        theorem_branches_map: Dict[str, Set[int]] = defaultdict(set)
        for thm_name, branch_idx in candidates:
            theorem_branches_map[thm_name].add(branch_idx)
        restricted_branches = True
    else:
        theorem_branches_map = {thm: None for thm in candidates}
        restricted_branches = False

    check_count = 0
    for candidate in theorem_branches_map.keys():
        if check_count >= max_check_attempts:
            break

        theorem_def = problem.parsed_theorem_GDL.get(candidate)
        if theorem_def is None:
            readiness_scores[(candidate, 1)] = 1
            check_count += 1
            continue

        branches = theorem_def.get("body", {})
        if not branches:
            if not restricted_branches or 1 in theorem_branches_map[candidate]:
                readiness_scores[(candidate, 1)] = 2
                valid_branches[candidate].add(1)
            continue

        for branch_name, gpl in branches.items():
            try:
                branch_idx = int(branch_name)
            except ValueError:
                continue

            if restricted_branches and branch_idx not in theorem_branches_map[candidate]:
                continue

            products = gpl.get("products", [])
            missing_count = 0
            checked_count = 0
            satisfied_count = 0
            missing_predicates = []

            def is_gdl_variable(elem):
                return isinstance(elem, str) and len(elem) == 1 and elem.islower()

            for predicate, item in products:
                all_variables = all(is_gdl_variable(elem) for elem in item)
                if all_variables:
                    checked_count += 1
                    instances = problem.condition.items_group.get(predicate, [])
                    has_type = len(instances) > 0
                    if has_type:
                        satisfied_count += 1
                    else:
                        missing_count += 1
                        if predicate not in missing_predicates:
                            missing_predicates.append(predicate)
                    continue

                checked_count += 1
                instances = problem.condition.items_group.get(predicate, [])
                has_type = len(instances) > 0
                if not has_type:
                    missing_count += 1
                    if predicate not in missing_predicates:
                        missing_predicates.append(predicate)
                    continue

                has_any_variable = any(is_gdl_variable(elem) for elem in item)
                if not has_any_variable:
                    has_instance = problem.condition.has(predicate, tuple(item))
                    if has_instance:
                        satisfied_count += 1
                    else:
                        if predicate not in missing_predicates:
                            missing_predicates.append(predicate)
                else:
                    satisfied_count += 1

            if checked_count == 0:
                branch_score = 3.0
            elif satisfied_count == checked_count:
                branch_score = 3.0
            elif missing_count == 0:
                branch_score = 2.5
            elif satisfied_count >= checked_count * 0.7:
                branch_score = 2.0
            elif satisfied_count >= checked_count * 0.4:
                branch_score = 1.5
            elif satisfied_count > 0:
                branch_score = 1.0
            else:
                branch_score = 0.0

            readiness_scores[(candidate, branch_idx)] = branch_score
            if missing_predicates:
                branch_details[(candidate, branch_idx)] = missing_predicates
            if branch_score >= 1.5:
                valid_branches[candidate].add(branch_idx)

        check_count += 1

    return valid_branches, readiness_scores, branch_details


def adaptive_candidate_ranking(
    candidates: List[Tuple[str, int]],
    *,
    forward_hint_heads: Sequence[Tuple[str, int]],
    failed_calls: Dict[str, Dict[str, Any]],
    history_records: Sequence[Dict[str, Any]],
    pending_prereqs: Dict[str, Dict[str, Any]],
    problem: Any,
    readiness_scores: Optional[Dict[Tuple[str, int], int]] = None,
    attempt_mode: str = "normal",
    failed_theorem_name: Optional[str] = None,
    tpg_common_prefix_theorems: Optional[Dict[Tuple[str, int], float]] = None,
    tpg_unlock_scores: Optional[Dict[Tuple[str, int], int]] = None,
    time_budget_ratio: float = 1.0,
    current_step: int = 0,
    last_successful_theorem: Optional[str] = None,
    merged_tpg: Optional[Dict[Any, List[Tuple[str, int]]]] = None,
    tpg_edge_weights: Optional[Dict[Tuple[Any, Tuple[str, int]], float]] = None,
    expected_steps: int = 0,
) -> List[Tuple[str, int]]:
    scores: Dict[Tuple[str, int], float] = {}

    if tpg_common_prefix_theorems is None:
        tpg_common_prefix_theorems = {}
    if tpg_unlock_scores is None:
        tpg_unlock_scores = {}
    if merged_tpg is None:
        merged_tpg = {}
    if tpg_edge_weights is None:
        tpg_edge_weights = {}

    tpg_successors: Set[Tuple[str, int]] = set()
    if last_successful_theorem and merged_tpg:
        for key in merged_tpg.keys():
            if key != "START" and isinstance(key, tuple) and key[0] == last_successful_theorem:
                for successor_branch in merged_tpg[key]:
                    tpg_successors.add(successor_branch)

    unique_theorems = {name for name, _ in candidates}
    complexity_cache: Dict[str, float] = {}
    for theorem_name in unique_theorems:
        complexity_cache[theorem_name] = compute_theorem_complexity(
            theorem_name, problem.parsed_theorem_GDL
        )

    goal_type = get_goal_type(problem)
    time_urgency_factor = 1.0 + (1.0 - time_budget_ratio) * 0.5

    step_urgency_factor = 1.0
    if expected_steps > 0:
        step_progress_ratio = current_step / expected_steps
        if step_progress_ratio > 1.5:
            step_urgency_factor = 1.5
        elif step_progress_ratio > 1.0:
            step_urgency_factor = 1.0 + (step_progress_ratio - 1.0)
        elif step_progress_ratio > 0.8:
            step_urgency_factor = 1.0 + (step_progress_ratio - 0.8) * 0.5

    urgency_factor = max(time_urgency_factor, step_urgency_factor)

    successful_successor_from_history: Set[str] = set()
    if last_successful_theorem and history_records:
        prev_theorem = None
        for record in history_records:
            if not record.get("updated"):
                prev_theorem = None
                continue
            current_theorem = record.get("call", "").split("(", 1)[0]
            if prev_theorem == last_successful_theorem and current_theorem:
                successful_successor_from_history.add(current_theorem)
            prev_theorem = current_theorem

    for candidate_name, branch_idx in candidates:
        score = 0.0
        for prereq_info in pending_prereqs.values():
            if candidate_name in prereq_info.get("suggestions", []):
                score += 10.0

        branch_key = (candidate_name, branch_idx)
        if branch_key in forward_hint_heads:
            complexity = complexity_cache[candidate_name]
            score += 10.0 if complexity >= 4.0 else (5.0 if complexity >= 2.0 else 0.0)

        for record in history_records:
            call = record.get("call", "")
            if call and call.startswith(candidate_name) and record.get("updated"):
                score += 5.0

        failed_count = sum(1 for key in failed_calls.keys() if key.startswith(candidate_name))
        score -= failed_count * 5.0

        if attempt_mode == "recovery" and failed_theorem_name and candidate_name == failed_theorem_name:
            score -= 20.0

        if readiness_scores:
            branch_score = readiness_scores.get((candidate_name, branch_idx), 0)
            base_readiness_bonus = 0.0
            if branch_score >= 3.0:
                base_readiness_bonus = 15.0
            elif branch_score >= 2.5:
                base_readiness_bonus = 12.0
            elif branch_score >= 2.0:
                base_readiness_bonus = 10.0
            elif branch_score >= 1.5:
                base_readiness_bonus = 5.0
            score += base_readiness_bonus * urgency_factor

        c_name = candidate_name.lower()
        if goal_type == "Angle":
            if "angle" in c_name or "parallel" in c_name or "perpendicular" in c_name or "congruent" in c_name:
                score += 5.0
        elif goal_type == "Area":
            if "area" in c_name or "ratio" in c_name:
                score += 5.0
        elif goal_type == "Length":
            if "length" in c_name or "line" in c_name or "pythagorean" in c_name or "similar" in c_name:
                score += 5.0

        if branch_key in tpg_common_prefix_theorems:
            prefix_weight = tpg_common_prefix_theorems[branch_key]
            early_stage_multiplier = 1.5 if current_step < 5 else 1.0
            score += prefix_weight * early_stage_multiplier

        if branch_key in tpg_unlock_scores:
            unlock_count = tpg_unlock_scores[branch_key]
            score += min(unlock_count * 2.0, 10.0)

        if branch_key in tpg_successors:
            base_successor_bonus = 15.0 if time_budget_ratio < 0.5 else 10.0
            edge_bonus_applied = False
            if last_successful_theorem and merged_tpg:
                for source_key in merged_tpg.keys():
                    if source_key != "START" and isinstance(source_key, tuple) and source_key[0] == last_successful_theorem:
                        if branch_key in merged_tpg.get(source_key, []):
                            edge_key = (source_key, branch_key)
                            if edge_key in tpg_edge_weights:
                                edge_freq = tpg_edge_weights[edge_key]
                                frequency_multiplier = 0.5 + edge_freq
                                successor_bonus = base_successor_bonus * frequency_multiplier
                                score += successor_bonus
                                edge_bonus_applied = True
                                break
            if not edge_bonus_applied:
                score += base_successor_bonus
        else:
            if last_successful_theorem and merged_tpg:
                for direct_successor_branch in tpg_successors:
                    second_order_successors = set(merged_tpg.get(direct_successor_branch, []))
                    if branch_key in second_order_successors:
                        second_order_bonus = 4.0 if time_budget_ratio < 0.5 else 2.5
                        for source_key in merged_tpg.keys():
                            if source_key != "START" and isinstance(source_key, tuple) and source_key[0] == last_successful_theorem:
                                if direct_successor_branch in merged_tpg.get(source_key, []):
                                    edge_key_1 = (source_key, direct_successor_branch)
                                    edge_key_2 = (direct_successor_branch, branch_key)
                                    if edge_key_1 in tpg_edge_weights and edge_key_2 in tpg_edge_weights:
                                        path_freq = tpg_edge_weights[edge_key_1] * tpg_edge_weights[edge_key_2]
                                        frequency_multiplier = 0.5 + path_freq
                                        second_order_bonus *= frequency_multiplier
                                    break
                        score += second_order_bonus
                        break
        if candidate_name in successful_successor_from_history:
            history_path_bonus = 6.0 if time_budget_ratio < 0.5 else 4.0
            score += history_path_bonus

        scores[(candidate_name, branch_idx)] = score

    return sorted(candidates, key=lambda c: scores.get(c, 0.0), reverse=True)


def build_branch_level_tpg_info_for_llm(
    candidate_branches: List[Tuple[str, int]],
    merged_tpg: Dict[Any, List[Tuple[str, int]]],
    readiness_scores: Dict[Tuple[str, int], int],
    tpg_edge_weights: Dict[Tuple[Any, Tuple[str, int]], float],
    last_successful_theorem: Optional[str],
    applied_theorem_names: Set[str],
) -> Dict[str, Any]:
    highly_ready = []
    moderately_ready = []
    needs_work = []

    for branch_tuple in candidate_branches:
        score = readiness_scores.get(branch_tuple, 0)
        thm_name, branch_idx = branch_tuple
        if thm_name in applied_theorem_names:
            continue
        entry = {"theorem": thm_name, "branch": branch_idx, "readiness": score}
        if score >= 3.0:
            highly_ready.append(entry)
        elif score >= 2.0:
            moderately_ready.append(entry)
        else:
            needs_work.append(entry)

    tpg_successors = []
    if last_successful_theorem and merged_tpg:
        successor_freq_map = {}
        for key in merged_tpg.keys():
            if key != "START" and isinstance(key, tuple) and key[0] == last_successful_theorem:
                for succ_branch in merged_tpg.get(key, []):
                    edge_key = (key, succ_branch)
                    freq = tpg_edge_weights.get(edge_key, 0.0)
                    if succ_branch not in successor_freq_map or freq > successor_freq_map[succ_branch]:
                        successor_freq_map[succ_branch] = freq
        for succ_branch, freq in sorted(successor_freq_map.items(), key=lambda x: -x[1]):
            thm_name, branch_idx = succ_branch
            if thm_name in applied_theorem_names:
                continue
            tpg_successors.append(
                {
                    "theorem": thm_name,
                    "branch": branch_idx,
                    "frequency": round(freq, 2),
                    "confidence": "high" if freq >= 0.7 else "medium" if freq >= 0.4 else "low",
                }
            )
        tpg_successors = tpg_successors[:5]

    start_branches = []
    if "START" in merged_tpg:
        for branch_tuple in merged_tpg["START"]:
            thm_name, branch_idx = branch_tuple
            if thm_name in applied_theorem_names:
                continue
            if branch_tuple in candidate_branches:
                score = readiness_scores.get(branch_tuple, 0)
                start_branches.append({"theorem": thm_name, "branch": branch_idx, "readiness": score})
        start_branches = start_branches[:5]

    unlock_entries = []
    for branch_tuple in candidate_branches[:15]:
        thm_name, branch_idx = branch_tuple
        if thm_name in applied_theorem_names:
            continue
        successors = merged_tpg.get(branch_tuple, [])
        valid_successors = [
            f"{s_name}({s_br})" for s_name, s_br in successors if s_name not in applied_theorem_names
        ]
        if valid_successors:
            branch_label = f"{thm_name}({branch_idx})"
            trimmed = valid_successors[:3]
            unlock_entries.append(
                {
                    "branch": branch_label,
                    "readiness": readiness_scores.get(branch_tuple, 0),
                    "unlock_count": len(valid_successors),
                    "unlocks": trimmed,
                }
            )

    if unlock_entries:
        unlock_entries.sort(key=lambda x: (x["unlock_count"], x["readiness"]), reverse=True)
        unlock_entries = unlock_entries[:6]

    return {
        "readiness_summary": {
            "highly_ready": highly_ready[:8],
            "moderately_ready": moderately_ready[:5],
            "needs_prerequisites": needs_work[:3],
        },
        "tpg_guidance": {
            "last_applied": last_successful_theorem,
            "recommended_next": tpg_successors,
            "unlocking_branches": unlock_entries,
        },
        "common_start_points": start_branches,
    }


def build_branch_planning_suggestions(
    candidate_branches: List[Tuple[str, int]],
    merged_tpg: Dict[Any, List[Tuple[str, int]]],
    readiness_scores: Dict[Tuple[str, int], int],
    tpg_edge_weights: Dict[Tuple[Any, Tuple[str, int]], float],
    goal_type: str,
    applied_theorem_names: Set[str],
) -> List[Dict[str, Any]]:
    goal_relevant_keywords = {
        "Length": ["pythagorean", "length", "line_addition", "similar", "ratio"],
        "Angle": ["angle", "parallel", "perpendicular", "complementary", "vertical"],
        "Area": ["area", "triangle_area", "similar", "ratio"],
    }
    relevant_keywords = goal_relevant_keywords.get(goal_type, [])

    suggestions: List[Dict[str, Any]] = []
    for branch_tuple in candidate_branches[:20]:
        thm_name, branch_idx = branch_tuple
        if thm_name in applied_theorem_names:
            continue
        score = readiness_scores.get(branch_tuple, 0)
        if score < 1.5:
            continue

        goal_relevance = 0.0
        thm_lower = thm_name.lower()
        for keyword in relevant_keywords:
            if keyword in thm_lower:
                goal_relevance += 2.0
                break

        successors = merged_tpg.get(branch_tuple, [])
        best_chain = []
        max_chain_freq = 0.0

        for succ in successors[:3]:
            s_name, s_br = succ
            if s_name in applied_theorem_names:
                continue
            edge_freq = tpg_edge_weights.get((branch_tuple, succ), 0.0)
            second_order = merged_tpg.get(succ, [])
            if second_order:
                for so in second_order[:2]:
                    so_name, so_br = so
                    if so_name not in applied_theorem_names:
                        so_freq = tpg_edge_weights.get((succ, so), 0.0)
                        total_freq = edge_freq * so_freq
                        if total_freq > max_chain_freq:
                            max_chain_freq = total_freq
                            best_chain = [
                                f"{thm_name}({branch_idx})",
                                f"{s_name}({s_br})",
                                f"{so_name}({so_br})",
                            ]
            else:
                if edge_freq > max_chain_freq:
                    max_chain_freq = edge_freq
                    best_chain = [f"{thm_name}({branch_idx})", f"{s_name}({s_br})"]

        for succ in successors[:3]:
            s_name, _ = succ
            if any(kw in s_name.lower() for kw in relevant_keywords):
                goal_relevance += 1.0
                break

        unlock_value = len([s for s in successors if s[0] not in applied_theorem_names])
        total_score = goal_relevance * 3 + unlock_value * 2 + score + max_chain_freq * 5

        if total_score <= 3.0:
            continue

        reason_parts = []
        if goal_relevance >= 2.0:
            reason_parts.append(f"goal={goal_type}")
        if unlock_value > 0:
            reason_parts.append(f"unlocks={unlock_value}")
        if score >= 3.0:
            reason_parts.append("ready")
        elif score >= 2.5:
            reason_parts.append("nearly-ready")
        if max_chain_freq >= 0.5:
            reason_parts.append(f"freq={int(max_chain_freq*100)}%")

        suggestions.append(
            {
                "start": f"{thm_name}({branch_idx})",
                "readiness": score,
                "chain": " -> ".join(best_chain) if best_chain else f"{thm_name}({branch_idx})",
                "chain_frequency": round(max_chain_freq, 2) if max_chain_freq > 0 else None,
                "reason": "; ".join(reason_parts),
                "total_score": total_score,
            }
        )
    suggestions.sort(key=lambda x: x["total_score"], reverse=True)
    return suggestions[:3]


def analyze_branch_chains(
    merged_tpg: Dict[Any, List[Tuple[str, int]]],
    candidate_branches: List[Tuple[str, int]],
    tpg_edge_weights: Dict[Tuple[Any, Tuple[str, int]], float],
    applied_theorem_names: Set[str],
    max_depth: int = 3,
) -> List[Dict[str, Any]]:
    if not merged_tpg:
        return []

    candidate_set = set(candidate_branches)
    chains = []

    def find_chains(current: Tuple[str, int], path: List[Tuple[str, int]], visited: Set[Tuple[str, int]], path_freq: float, depth: int = 0):
        if depth >= max_depth:
            if len(path) >= 2:
                chains.append((list(path), path_freq))
            return

        successors = merged_tpg.get(current, [])
        valid_successors = [
            s for s in successors if s in candidate_set and s not in visited and s[0] not in applied_theorem_names
        ]

        if not valid_successors:
            if len(path) >= 2:
                chains.append((list(path), path_freq))
            return

        for succ in valid_successors[:3]:
            edge_key = (current, succ)
            edge_freq = tpg_edge_weights.get(edge_key, 0.1)
            new_freq = path_freq * edge_freq
            find_chains(succ, path + [succ], visited | {succ}, new_freq, depth + 1)

    for start_branch in merged_tpg.get("START", [])[:10]:
        if start_branch in candidate_set and start_branch[0] not in applied_theorem_names:
            find_chains(start_branch, [start_branch], {start_branch}, 1.0)

    chains.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    result = []
    seen_starts = set()
    for path, freq in chains[:5]:
        if path[0] in seen_starts:
            continue
        chain_str = " -> ".join([f"{name}({br})" for name, br in path])
        result.append({"chain": chain_str, "frequency": round(freq, 2), "length": len(path)})
        seen_starts.add(path[0])
    return result


def build_one_step_messages(
    problem: Dict[str, Any],
    theorem_branches: Sequence[Tuple[str, int]],
    gdl_map: Dict[str, Dict[str, Any]],
    image_data_url: str,
    state_payload: Dict[str, Any],
    applied_calls: Sequence[str],
    applied_history: Sequence[Dict[str, Any]],
    recent_feedback: Sequence[str],
    step_index: int,
    max_steps: int,
    forward_hints: Sequence[str],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    theorems_desc = []
    for theorem_name, branch_idx in theorem_branches:
        b_scores = {}
        b_details = {}
        readiness_scores = kwargs.get("readiness_scores")
        branch_details = kwargs.get("branch_details")
        if readiness_scores:
            branch_score = readiness_scores.get((theorem_name, branch_idx))
            if branch_score is not None:
                b_scores[branch_idx] = branch_score
        if branch_details:
            details = branch_details.get((theorem_name, branch_idx))
            if details is not None:
                b_details[branch_idx] = details
        entry = build_theorem_prompt_entry(
            theorem_name,
            gdl_map.get(theorem_name, {}),
            {branch_idx},
            b_scores,
            b_details,
        )
        entry_s = copy.deepcopy(entry)
        entry_s["argument_roles"] = []
        entry_s["note"] = _redact_text_remove_objects(entry_s.get("note", ""))
        entry_s["valid_call_example"] = _redact_text_remove_objects(entry_s.get("valid_call_example", ""))
        theorems_desc.append(entry_s)

    applied_history_redacted = []
    for entry in applied_history:
        if not isinstance(entry, dict):
            continue
        ce = copy.deepcopy(entry)
        if "call" in ce and ce.get("call"):
            ce["call"] = _reduce_theorem_call_for_llm(str(ce.get("call")))
        applied_history_redacted.append(ce)

    payload = {
        "instruction": ONE_STEP_PROMPT_TEMPLATE,
        "problem": _build_problem_dict(problem),
        "candidate_theorems": theorems_desc,
        "state": {
            "readable_summary": state_payload.get("readable_summary", ""),
            "raw_conditions": state_payload.get("raw_conditions", {}),
            "symbol_mappings": state_payload.get("symbol_mappings", []),
            "known_symbol_values": state_payload.get("known_symbol_values", []),
            "delta_from_last_step": state_payload.get("delta_from_last_step", {}),
        },
        "applied_calls": [_reduce_theorem_call_for_llm(str(c)) for c in applied_calls],
        "applied_history": applied_history_redacted,
        "recent_feedback": list(recent_feedback),
        "step_index": step_index,
        "max_steps": max_steps,
        "output_format": {"calls": ["theorem_name", "theorem_name(branch)"]},
    }

    branch_tpg_info = kwargs.get("branch_tpg_info")
    if branch_tpg_info:
        payload["branch_readiness"] = branch_tpg_info.get("readiness_summary", {})
        payload["tpg_guidance"] = branch_tpg_info.get("tpg_guidance", {})
        if branch_tpg_info.get("common_start_points"):
            payload["frequent_starts"] = branch_tpg_info["common_start_points"]

    recovery_payload = kwargs.get("recovery_payload")
    attempt_mode = kwargs.get("attempt_mode", "normal")
    payload["mode"] = {"type": attempt_mode}
    if recovery_payload:
        payload["recent_failure"] = recovery_payload
        if attempt_mode == "recovery":
            payload["mode"]["recovery_context"] = {
                "failed_call": recovery_payload.get("failed_call"),
                "reason": recovery_payload.get("reason", "unknown"),
            }
    user_content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    if image_data_url:
        user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    return [{"role": "user", "content": user_content}]


def call_llm(client: Any, model: str, messages: List[Dict[str, Any]], temperature: float, max_retries: int, logger: Optional[Any] = None, api_keys: Optional[List[str]] = None, api_base: Optional[str] = None) -> str:
    global _current_api_key_index
    api_keys = api_keys or LLM_API_KEYS
    api_base = api_base or LLM_API_BASE
    if len(api_keys) < 2:
        raise ValueError("At least 2 API keys required for fallback strategy")

    clients = {
        0: client,
        1: OpenAI(api_key=api_keys[1], base_url=api_base, timeout=300.0),
    }
    last_error = ""
    for attempt in range(1, max_retries + 1):
        for key_idx, key in enumerate(api_keys):
            try:
                c = clients.get(key_idx)
                if c is None:
                    c = OpenAI(api_key=key, base_url=api_base, timeout=300.0)
                    clients[key_idx] = c
                if logger:
                    logger.info("[LLM] attempt=%s/%s key_index=%s", attempt, max_retries, key_idx)
                start_time = time.time()
                resp = c.chat.completions.create(model=model, messages=messages, temperature=temperature)
                if logger:
                    logger.info("[LLM] request completed in %.2f seconds (key_index=%s)", time.time() - start_time, key_idx)
                _current_api_key_index = key_idx
                return (getattr(resp.choices[0].message, "content", "") or "").strip()
            except Exception as exc:
                last_error = str(exc)
                if logger:
                    logger.warning(
                        "[LLM] attempt=%s/%s key_index=%s failed -> %s",
                        attempt,
                        max_retries,
                        key_idx,
                        last_error,
                    )
        if attempt < max_retries:
            sleep_time = random.uniform(5.0, 15.0)
            if logger:
                logger.info("[LLM] All keys failed, waiting %.1f seconds before retry...", sleep_time)
            time.sleep(sleep_time)
    raise RuntimeError(
        f"LLM request failed after {max_retries} attempts across all {len(api_keys)} API keys: {last_error}"
    )


def extract_calls(raw_text: str) -> List[str]:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 1)[1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except Exception:
        return []
    calls = data.get("calls")
    return [str(x) for x in calls] if isinstance(calls, list) else []


def build_candidate_theorems(problem_id: int, *, problems_dir: Path, diagrams_dir: Path, mode: str, top_k: int, entries: Dict[str, Dict[str, Any]], embedder: MultiModalEmbedding) -> Tuple[List[Tuple[str, int]], List[Dict[str, Any]], Dict[str, Any], Dict[Any, List[Tuple[str, int]]], Dict[Tuple[str, int], float], Dict[Tuple[str, int], int], Dict[Tuple[Any, Tuple[str, int]], float], int]:
    pid = str(problem_id)
    text_items, shapes, cleaned_cdl, image_path, layout_desc = load_problem_embedding_input_adapter(pid, mode, problems_dir, diagrams_dir)
    if mode in IMAGE_REQUIRED_MODES and not image_path:
        raise ValueError(f"Problem {problem_id} lacks image required by mode {mode}")
    if mode == "text-en":
        query_text_items = text_items
        query_image_arg = None
    elif mode in {"text-image", "text-image-en", "layout", "raw-cdl-image"}:
        query_text_items = text_items
        query_image_arg = str(image_path) if image_path else None
    elif mode == "image-only":
        query_text_items = []
        query_image_arg = str(image_path) if image_path else None
    else:
        query_text_items = text_items if text_items else []
        query_image_arg = str(image_path) if image_path else None

    if not query_text_items and not query_image_arg:
        raise ValueError(f"Problem {problem_id} has no usable payload for mode {mode}")

    query_vec = embedder.embed(query_text_items, query_image_arg)
    top = rank_results(query_vec, entries, top_k, pid)
    aggregated: List[Tuple[str, int]] = []
    seen: Set[Tuple[str, int]] = set()
    recs: List[Dict[str, Any]] = []
    tpgs: List[Dict[Any, List[Tuple[str, int]]]] = []
    steps: List[int] = []
    for i, (cid, score, _entry) in enumerate(top, start=1):
        seqs = load_theorem_sequences(cid, problems_dir)
        if seqs:
            steps.append(len(seqs))
        tpg = load_theorem_tpg_with_branches(cid, problems_dir)
        if tpg:
            tpgs.append(tpg)
        entry_shapes = _entry.get("shapes")
        if isinstance(entry_shapes, list):
            shapes_preview = [str(item) for item in entry_shapes if isinstance(item, (str, int, float))][:5]
        elif isinstance(entry_shapes, (str, int, float)):
            shapes_preview = [str(entry_shapes)]
        else:
            shapes_preview = []
        text_items_raw = _entry.get("text_items")
        if isinstance(text_items_raw, list) and text_items_raw:
            summary = str(text_items_raw[0])
        elif isinstance(text_items_raw, str):
            summary = text_items_raw
        else:
            summary = ""
        recs.append(
            {
                "rank": i,
                "problem_id": int(cid),
                "score": score,
                "shapes": shapes_preview,
                "text_item_preview": summary,
                "theorems": seqs,
            }
        )
        for t in seqs:
            if t not in seen:
                seen.add(t)
                aggregated.append(t)

    theorem_metadata: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for rec in recs:
        score = rec["score"]
        rank = rec["rank"]
        theorems = rec["theorems"]
        for position, thm_tuple in enumerate(theorems, start=1):
            if thm_tuple not in theorem_metadata:
                theorem_metadata[thm_tuple] = {"occurrences": [], "positions": [], "scores": []}
            theorem_metadata[thm_tuple]["occurrences"].append(
                {"rank": rank, "score": score, "position": position}
            )
            theorem_metadata[thm_tuple]["positions"].append(position)
            theorem_metadata[thm_tuple]["scores"].append(score)

    goal_type = "Other"
    try:
        problem_data = load_problem(pid, problems_dir)
        goal_type = get_goal_type(problem_data)
    except Exception:
        pass

    theorem_weights: Dict[Tuple[str, int], float] = {}
    for thm_tuple in aggregated:
        meta = theorem_metadata[thm_tuple]
        original_weight = 0.0
        for occ in meta["occurrences"]:
            score = occ["score"]
            rank = occ["rank"]
            original_weight += score / (rank + 1)
        count = len(meta["occurrences"])
        avg_score = sum(meta["scores"]) / len(meta["scores"])
        freq_weight = (count + 1) * avg_score
        avg_position = sum(meta["positions"]) / len(meta["positions"])
        pos_weight = 1.0 / (avg_position + 1)

        type_bonus = 0.0
        thm_name_lower = thm_tuple[0].lower()
        if goal_type == "Angle":
            if any(kw in thm_name_lower for kw in ["angle", "parallel", "perpendicular", "congruent", "bisector", "ratio"]):
                type_bonus = 1.0
        elif goal_type == "Area":
            if any(kw in thm_name_lower for kw in ["area", "ratio", "triangle", "quadrilateral"]):
                type_bonus = 1.0

        theorem_weights[thm_tuple] = original_weight + 0.4 * freq_weight + 0.25 * pos_weight + 0.20 * type_bonus

    aggregated = sorted(aggregated, key=lambda t: theorem_weights.get(t, 0.0), reverse=True)

    merged_tpg, edge_w = merge_theorem_tpgs_with_edge_weights_branches(tpgs, seen)
    prefix, unlock = compute_tpg_dynamic_priorities_branches(tpgs, seen)
    expected = 0 if not steps else sorted(steps)[len(steps) // 2]
    query_payload = {"text_items": text_items, "shapes": shapes, "cdl": cleaned_cdl, "image_path": str(image_path) if image_path else "", "layout": layout_desc}
    return aggregated, recs, query_payload, merged_tpg, prefix, unlock, edge_w, expected
