# Standard library
import argparse
import base64
import copy
import json
import logging
import math
import os
import random
import re
import sys
import time
import warnings
import multiprocessing
import threading
from collections import defaultdict
from pathlib import Path

# Add project root to Python path for local imports.
# engine.py lives under solve/, so project root is one level up.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from multiprocessing import Pool, Manager, Process, Queue as MPQueue
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set, Union

# Third-party libraries
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import sympy as sp

try:
	from openai import OpenAI
except ImportError:
	OpenAI = None

# FormalGeo
from formalgeo.data import DatasetLoader
from formalgeo.parse import (
	inverse_parse_one,
	inverse_parse_one_theorem,
	parse_theorem_seqs,
	get_equation_from_tree,
)
from formalgeo.solver import Interactor
from formalgeo.solver.forward_search import get_p2t_map_fw
from formalgeo.core import GeometryPredicateLogicExecutor as GPLExecutor
from formalgeo.core import EquationKiller as EqKiller
from formalgeo.tools import show_solution

# Local RAG modules
from RAG.create_RAG_base import (
	DATA_SPLITS as _DEFAULT_DATA_SPLITS,  # Default FormalGeo7k splits
	DEFAULT_SPLIT_FOR_RETRIEVAL,
	EMBEDDING_STORE_PATHS,
	IMAGE_REQUIRED_MODES,
	MultiModalEmbedding,
	build_problem_payload,
	load_embedding_store,
	cosine_similarity,
	PROBLEMS_DIR as RAG_PROBLEMS_DIR,
	DIAGRAMS_DIR as RAG_DIAGRAMS_DIR,
)
from RAG import retrieval as rag_retrieval

from .ops import (
	extract_theorem_name as _extract_theorem_name,
	extract_theorem_name_and_branch as _extract_theorem_name_and_branch,
	normalize_theorem_call as _normalize_theorem_call,
	ensure_theorem_call_has_branch as _ensure_theorem_call_has_branch,
	is_wrapped_entity as _is_wrapped_entity,
	_build_problem_dict as _llm_build_problem_dict,
	_build_redacted_problem_dict as _llm_build_redacted_problem_dict,
	_reduce_theorem_call_for_llm as _llm_reduce_theorem_call_for_llm,
	_redact_text_remove_objects as _llm_redact_text_remove_objects,
	build_one_step_messages as _build_one_step_messages,
	call_llm as _call_llm,
	extract_calls as _extract_calls,
	build_candidate_theorems as _build_candidate_theorems,
	load_problem as _load_problem,
	load_problem_embedding_input_adapter as _load_problem_embedding_input_adapter,
	load_theorem_sequences as _load_theorem_sequences,
	load_theorem_tpg as _load_theorem_tpg,
	load_theorem_tpg_with_branches as _load_theorem_tpg_with_branches,
	prepare_embedding_store as _prepare_embedding_store,
	rank_results as _rank_results,
	merge_theorem_tpgs as _merge_theorem_tpgs,
	merge_theorem_tpgs_with_edge_weights as _merge_theorem_tpgs_with_edge_weights,
	compute_tpg_dynamic_priorities as _compute_tpg_dynamic_priorities,
	merge_theorem_tpgs_with_edge_weights_branches as _merge_theorem_tpgs_with_edge_weights_branches,
	compute_tpg_dynamic_priorities_branches as _compute_tpg_dynamic_priorities_branches,
	adaptive_candidate_ranking as _adaptive_candidate_ranking,
	build_branch_level_tpg_info_for_llm as _build_branch_level_tpg_info_for_llm,
	build_branch_planning_suggestions as _build_branch_planning_suggestions,
	analyze_branch_chains as _analyze_branch_chains,
	compute_theorem_complexity as _compute_theorem_complexity,
	filter_candidates_by_precheck as _filter_candidates_by_precheck,
	get_goal_type as _get_goal_type,
)


# =============================================================================
# SECTION 1: CONSTANTS AND CONFIG
# =============================================================================

# LLM prompt template (one-step mode only)


# Dataset config
DEFAULT_DATASETS_PATH = _project_root / "datasets"
DEFAULT_DATASET_NAME = "formalgeo7k_v2"

# Embedding config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v4")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "https://aihubmix.com/v1")
EMBEDDING_INSTRUCTION = None

# LLM config (multi-key fallback)
LLM_API_KEYS = [
	os.environ.get("OPENAI_API_KEY")
]
LLM_API_KEY = LLM_API_KEYS[0]
LLM_API_BASE = os.environ.get("OPENAI_BASE_URL", "https://aihubmix.com/v1")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
LLM_TEMPERATURE = 0.1
LLM_MAX_RETRIES = 3

# Output config
OUTPUT_DIR = Path("output_solver_runs")
QUICK_START_DEFAULT_FILE = OUTPUT_DIR / "GPT_5_2.json"

# Internal caches
THEOREM_CACHE: Dict[str, List[str]] = {}
LLM_CLIENT_CACHE: Dict[Tuple[str, str], Any] = {}
_current_api_key_index = 0
QUICK_START_REPLAY_CACHE: Dict[Path, Dict[int, List[str]]] = {}

# Dataset split config
DATA_SPLITS: Dict[str, Tuple[int, ...]] = _DEFAULT_DATA_SPLITS.copy()
TEST_SPLIT_NAME = DEFAULT_SPLIT_FOR_RETRIEVAL
TEST_PROBLEM_IDS: List[int] = []  # Initialized in main.

# -----------------------------------------------------------------------------
# Solver constants
# -----------------------------------------------------------------------------

MAX_RECOVERY_ATTEMPTS = 3  # Max recovery retries per step.
FV_WARNING_MARKERS: Tuple[str, ...] = ("FV check not passed",)  # Forward-check failure marker.

# -----------------------------------------------------------------------------
# Theorem-level timeout control
# -----------------------------------------------------------------------------
# Prevent long-running theorem execution from stalling the solver.

MAX_THEOREM_EXECUTION_TIME = 300  # Max time per theorem execution (seconds).
THEOREM_TIMEOUT_BLACKLIST: Set[str] = set()  # Runtime timeout blacklist.

# -----------------------------------------------------------------------------
# Reusable theorems whitelist
# -----------------------------------------------------------------------------

REUSABLE_THEOREMS_WHITELIST: Set[str] = {
	"radius_of_circle_property_length_equal",
	"line_addition",
	"parallel_property_collinear_extend",
	"triangle_property_angle_sum",
	"similar_triangle_property_line_ratio",
	"parallelogram_property_opposite_line_equal",
	"angle_addition",
	"tangent_of_circle_property_perpendicular",
	"arc_property_circumference_angle_external",
	"isosceles_triangle_judgment_line_equal",
	"parallel_property_alternate_interior_angle",
	"adjacent_complementary_angle",
	"mirror_similar_triangle_property_line_ratio",
	"isosceles_triangle_property_angle_equal",
	"parallel_judgment_ipsilateral_internal_angle",
	"right_triangle_judgment_angle",
	"right_triangle_property_pythagorean",
	"sine_theorem",
	"parallel_property_corresponding_angle",
	"arc_property_center_angle",
	"circle_property_chord_perpendicular_bisect_chord",
	"cosine_theorem",
	"tangent_of_circle_property_length_equal",
	"isosceles_triangle_judgment_angle_equal",
	"mirror_congruent_triangle_property_line_equal",
	"similar_triangle_judgment_aa",
	"equilateral_triangle_property_angle",
}

# Log whitelist exemptions
def log_whitelist_exemption(problem_id: str, step_idx: int, theorem_name: str, logger: Optional[logging.Logger]):
	"""Log that a theorem is exempted from the blacklist."""
	if logger:
		logger.info(
			"[Whitelist] problem=%s step=%s theorem=%s exempted from blacklist (reusable theorem)",
			problem_id, step_idx, theorem_name
		)


# =============================================================================
# SECTION 2: CLI ARGUMENT PARSING & CONFIGURATION
# =============================================================================

def parse_cli_args() -> argparse.Namespace:
	"""Parse CLI arguments."""
	def parse_bool_flag(value: str) -> bool:
		text = str(value).strip().lower()
		if text in {"true", "1", "yes", "y", "on"}:
			return True
		if text in {"false", "0", "no", "n", "off"}:
			return False
		raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

	parser = argparse.ArgumentParser(
		description="FormalGeo theorem-only prediction solver"
	)
	
	# Problem selection
	parser.add_argument("--problem-id", type=int, default=None, help="Solve one problem ID")
	parser.add_argument("--problem-ids", type=str, default=None, help="Solve a comma-separated list, e.g., 1854,1862,1916")
	parser.add_argument("--max-problems", type=int, default=0, help="Max problems to solve (0 = all)")
	parser.add_argument("--start-problem-id", type=int, default=0, help="Start from a specific problem ID (0 = from start)")
	
	# RAG retrieval
	parser.add_argument("--top-k", type=int, default=200, help="Retrieve top-K similar problems")
	parser.add_argument("--retrieval-mode", choices=tuple(EMBEDDING_STORE_PATHS.keys()), default="text-image-en", help="Embedding retrieval mode")
	parser.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL, help="Embedding model (defaults to EMBEDDING_MODEL env)")
	parser.add_argument("--one-step-max-steps", type=int, default=20, help="Max steps for one-step mode")
	# Output options
	parser.add_argument("--show-solution", type=parse_bool_flag, default=False, help="Show detailed solving process")
	parser.add_argument("--quick-start", type=parse_bool_flag, default=True, help="Replay from output_solver_runs/GPT_5_4.json without LLM calls")
	parser.add_argument("--quick-start-file", type=str, default=str(QUICK_START_DEFAULT_FILE), help="Quick-start replay file path")
	
	# Performance
	parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
	parser.add_argument("--time-limit", type=int, default=600, help="Per-problem time limit in seconds")
	
	return parser.parse_args()


def resolve_problem_ids(
	problem_id: Optional[int],
	problem_ids_str: Optional[str],
	max_problems: int,
	start_problem_id: Optional[int],
) -> List[int]:
	split_ids = TEST_PROBLEM_IDS  # Already sorted numerically
	
	# If specific problem requested, return only that one
	if problem_id is not None:
		if problem_id not in split_ids:
			raise ValueError(f"Problem {problem_id} not part of {TEST_SPLIT_NAME} split")
		return [problem_id]
	
	# If multiple specific problems requested via comma-separated string
	if problem_ids_str is not None:
		try:
			requested_ids = [int(pid.strip()) for pid in problem_ids_str.split(",") if pid.strip()]
		except ValueError as e:
			raise ValueError(f"Invalid problem IDs format: {problem_ids_str}. Use comma-separated integers like '1854,1862,1916'") from e
		
		# Validate all IDs are in the test split
		invalid_ids = [pid for pid in requested_ids if pid not in split_ids]
		if invalid_ids:
			pass
		return sorted(requested_ids)
	
	# Default: process the full test split (subject to start/max slicing).
	# Determine starting index
	start_index = 0
	if start_problem_id is not None and start_problem_id > 0:
		if start_problem_id not in split_ids:
			raise ValueError(f"Problem {start_problem_id} not in {TEST_SPLIT_NAME} split")
		start_index = split_ids.index(start_problem_id)
	
	# Determine ending index
	if max_problems > 0:
		# Process max_problems starting from start_index
		result_ids = split_ids[start_index : start_index + max_problems]
	else:
		# Process all remaining problems from start_index
		result_ids = split_ids[start_index:]
	
	return result_ids


def ensure_openai_client(api_key: str, api_base: str, model: str) -> Any:
	if OpenAI is None:
		raise RuntimeError("openai package not available; install openai to run the LLM stage")
	if not api_key:
		raise ValueError("LLM API key is required")
	if not api_base:
		raise ValueError("LLM API base URL is required")
	if not model:
		raise ValueError("LLM model name is required")
	cache_key = (api_key, api_base)
	client = LLM_CLIENT_CACHE.get(cache_key)
	if client is None:
		# Set timeout to prevent hanging on slow API responses
		# Increased to 300s to allow more time for complex problems
		client = OpenAI(api_key=api_key, base_url=api_base, timeout=300.0)
		LLM_CLIENT_CACHE[cache_key] = client
	return client


def load_quick_start_replay_map(replay_file: Path, logger: Optional[logging.Logger] = None) -> Dict[int, List[str]]:
	"""Load quick-start replay data from a historical summary file.

	Returns:
		Dict[problem_id, execution_calls]
	"""
	replay_file = replay_file.resolve()
	if replay_file in QUICK_START_REPLAY_CACHE:
		return QUICK_START_REPLAY_CACHE[replay_file]

	if not replay_file.exists():
		if logger:
			logger.warning("[QuickStart] replay file not found: %s", replay_file)
		QUICK_START_REPLAY_CACHE[replay_file] = {}
		return {}

	try:
		with open(replay_file, "r", encoding="utf-8") as f:
			payload = json.load(f)
	except Exception as exc:
		if logger:
			logger.warning("[QuickStart] failed to read replay file %s -> %s", replay_file, exc)
		QUICK_START_REPLAY_CACHE[replay_file] = {}
		return {}

	replay_map: Dict[int, List[str]] = {}
	for entry in payload.get("problems", []):
		if not isinstance(entry, dict):
			continue
		pid = entry.get("problem_id")
		if not isinstance(pid, int):
			continue
		calls: List[str] = []
		for step_text in entry.get("execution_steps", []) or []:
			if not isinstance(step_text, str):
				continue
			match = re.search(r"Executed:\s*([^|]+)\s*\|", step_text)
			if match:
				call = str(match.group(1) or "").strip()
				if call:
					calls.append(call)
		replay_map[pid] = calls

	QUICK_START_REPLAY_CACHE[replay_file] = replay_map
	if logger:
		logger.debug("[QuickStart] loaded replay map: %s problems from %s", len(replay_map), replay_file)
	return replay_map


def build_quick_start_llm_response(
	problem_id: int,
	step_idx: int,
	replay_map: Dict[int, List[str]],
	logger: Optional[logging.Logger] = None,
) -> str:
	"""Build an LLM-like JSON response from replayed execution steps."""
	calls = replay_map.get(problem_id, [])
	call = calls[step_idx - 1] if 0 <= step_idx - 1 < len(calls) else None
	if call:
		if logger:
			logger.info("[QuickStart] problem=%s step=%s replay_call=%s", problem_id, step_idx, call)
		return json.dumps({"calls": [call]}, ensure_ascii=False)
	if logger:
		logger.info("[QuickStart] problem=%s step=%s no replay call available", problem_id, step_idx)
	return json.dumps({"calls": []}, ensure_ascii=False)

# =============================================================================
# SECTION 3: DATA LOADING & PREPROCESSING
# =============================================================================


def load_problem_embedding_input_adapter(
	problem_id: str, mode: str, problems_dir: Path, diagrams_dir: Path
) -> Tuple[List[str], List[str], List[str], Optional[Path], str]:
	return _load_problem_embedding_input_adapter(problem_id, mode, problems_dir, diagrams_dir)


def load_problem(problem_id: str, problems_dir: Path) -> Dict[str, Any]:
	return _load_problem(problem_id, problems_dir)


def load_theorem_sequences(problem_id: str, problems_dir: Path) -> List[Tuple[str, int]]:
	return _load_theorem_sequences(problem_id, problems_dir)


# -----------------------------------------------------------------------------
# TPG (Theorem Precedence Graph) cache
# -----------------------------------------------------------------------------
THEOREM_TPG_CACHE: Dict[str, Dict[str, List[str]]] = {}


def load_theorem_tpg(problem_id: str, problems_dir: Path) -> Dict[str, List[str]]:
	return _load_theorem_tpg(problem_id, problems_dir)


def load_theorem_tpg_with_branches(problem_id: str, problems_dir: Path) -> Dict[Tuple[str, int], List[Tuple[str, int]]]:
	return _load_theorem_tpg_with_branches(problem_id, problems_dir)


def extract_theorem_name(call: str) -> str:
	return _extract_theorem_name(call)


def extract_theorem_name_and_branch(call: str) -> Optional[Tuple[str, int]]:
	return _extract_theorem_name_and_branch(call)


def normalize_theorem_call_with_branch(call: str, gdl_map: Dict[str, Dict[str, Any]], logger=None) -> Tuple[str, bool]:
	"""Check whether the call includes a branch index."""
	if not call:
		return call, False
	
	# Missing parentheses implies no branch provided.
	if "(" not in call:
		theorem_name = call.strip()
		if logger:
			logger.warning(
				"[Normalize] LLM output missing branch: '%s' -> will use solver auto-search (try all branches)",
				call,
			)
		return call, True
	
	# Parse the call.
	parsed = parse_theorem_seqs([call])
	if not parsed:
		# Parse failed; return original call.
		return call, False
	
	t_name, t_branch, t_para = parsed[0]
	
	if t_branch is None:
		# Call has args but no branch; allow solver to auto-search.
		if logger:
			logger.warning(
				"[Normalize] LLM output missing branch: '%s' -> will use solver auto-search (try all branches)",
				call,
			)
		return call, True
	
	# Branch present; no auto-search.
	return call, False




def load_gdl_signatures(path: Path) -> Dict[str, Dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as f:
		gdl = json.load(f)
	mapping: Dict[str, Dict[str, Any]] = {}
	for full_key, forms in gdl.items():
		if "(" not in full_key or not full_key.endswith(")"):
			continue
		name, args = full_key.split("(", 1)
		args = args[:-1]
		form_list: List[Dict[str, Any]] = []
		if isinstance(forms, dict):
			for k, payload in forms.items():
				try:
					idx = int(k)
				except Exception:
					continue
				premise = payload.get("premise", "") if isinstance(payload, dict) else ""
				conclusion = payload.get("conclusion", []) if isinstance(payload, dict) else []
				form_list.append({"index": idx, "premise": premise, "conclusion": conclusion})
			form_list.sort(key=lambda x: x["index"])
		mapping[name] = {"signature": f"{name}({args})", "forms": form_list}
	return mapping

# =============================================================================
# SECTION 4: STATE INSPECTION & FORMATTING UTILITIES
# =============================================================================

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
	
	# ALWAYS include branch index in valid_call_example, even for single-branch theorems
	# This ensures LLM always outputs theorem_name(branch_index) format
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
	branch_details: Optional[Dict[int, List[str]]] = None
) -> Dict[str, Any]:
	details = compute_signature_details(name, info)
	argument_roles = details["argument_roles"]
	valid_example = details["valid_call_example"]
	has_forms = details["has_forms"]
	
	# Filter forms if valid_indices provided
	if valid_indices is not None and has_forms:
		details["forms"] = [f for f in details["forms"] if f["index"] in valid_indices]
	
	# Annotate forms with readiness
	if branch_scores and has_forms:
		readiness_labels = {
			3: "✓ Ready",
			2: "~ Partial",
			1: "? Unknown",
			0: "✗ Low",
		}
		for form in details["forms"]:
			idx = form["index"]
			if idx in branch_scores:
				score = branch_scores[idx]
				label = readiness_labels.get(score, "")
				
				# Add missing details for Partial matches
				if score == 2 and branch_details and idx in branch_details:
					missing = branch_details[idx]
					if missing:
						label += f" (needs: {', '.join(missing)})"
				
				form["readiness"] = label
				form["readiness_score"] = score

	note_parts: List[str] = [f"Valid call example: {valid_example}."]
	if has_forms:
		note_parts.append("REQUIRED: Replace 'branch_index' with a specific integer form id from 'forms' list.")
	else:
		note_parts.append("REQUIRED: This theorem has 1 branch, always use branch_index=1.")
	if argument_roles:
		note_parts.append("Use raw tokens matching the listed argument_roles in order.")
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


def format_condition_component(item: Any) -> str:
	if isinstance(item, tuple):
		inner = ",".join(format_condition_component(elem) for elem in item)
		return f"({inner})"
	return str(item)


def collect_raw_conditions(problem: Any, *, limit_per_predicate: int = 40) -> Dict[str, List[str]]:
	raw: Dict[str, List[str]] = {}
	for predicate, items in problem.condition.items_group.items():
		if not items:
			continue
		formatted = [str(item) if predicate == "Equation" else f"{predicate}{format_condition_component(item)}" for item in items[:limit_per_predicate]]
		if formatted:
			raw[predicate] = formatted
	return raw


def collect_symbol_details(problem: Any, *, limit_symbols: int = 60) -> Tuple[List[str], List[str]]:
	mapping_entries = [f"{sym} -> {attr}{format_condition_component(extend_items[0] if extend_items else ())}" 
	                   for sym, (attr, extend_items) in problem.condition.attr_of_sym.items()]
	value_entries = [f"{sym} = {value}" for sym, value in problem.condition.value_of_sym.items() if value is not None]
	return mapping_entries[:limit_symbols], value_entries[:limit_symbols]


def _format_args(item: Any) -> str:
	if isinstance(item, (list, tuple)):
		if all(isinstance(elem, str) for elem in item):
			return "".join(item)
		return ",".join(_format_args(elem) for elem in item)
	return str(item)


def build_symbol_map(problem: Any) -> Dict[str, str]:
	mapping: Dict[str, str] = {}
	for sym, (attr, extend_items) in problem.condition.attr_of_sym.items():
		base_item = extend_items[0] if extend_items else ()
		args_repr = _format_args(base_item)
		mapping[str(sym)] = f"{attr}({args_repr})"
	return mapping


def expr_to_readable(expr: sp.Expr, symbol_map: Dict[str, str]) -> str:
	text = sp.sstr(expr)
	for name, replacement in sorted(symbol_map.items(), key=lambda x: -len(x[0])):
		text = text.replace(name, replacement)
	text = text.replace("**", "^")
	return text


def format_equation(problem: Any, expr: sp.Expr, symbol_map: Dict[str, str]) -> str:
	if expr is None:
		return "(invalid equation)"
	try:
		simplified = sp.simplify(expr)
		expanded = sp.expand(simplified)
	except Exception:
		return str(expr)
	
	terms = sp.Add.make_args(expanded)
	left_terms, right_terms = [], []
	for term in terms:
		if term is None:
			continue
		(right_terms.append(-term) if term.could_extract_minus_sign() else left_terms.append(term))
	
	left_expr = sp.Add(*left_terms) if left_terms else sp.Integer(0)
	right_expr = sp.Add(*right_terms) if right_terms else sp.Integer(0)
	left_text = expr_to_readable(sp.simplify(left_expr), symbol_map)
	right_text = expr_to_readable(sp.simplify(right_expr), symbol_map)
	return f"{left_text} = {right_text}"


def describe_theorem(problem: Any, theorem: Optional[Tuple[Any, ...]]) -> str:
	if not theorem:
		return "(unknown)"
	name = theorem[0]
	if name in {None, "prerequisite", "extended", "init_problem", "check_goal"}:
		return str(name)
	if name == "solve_eq":
		return "solve_eq"
	try:
		return inverse_parse_one_theorem(theorem, problem.parsed_theorem_GDL)
	except Exception:
		branch = theorem[1] if len(theorem) > 1 else None
		params = theorem[2] if len(theorem) > 2 else None
		args = ""
		if params:
			args = ", ".join(map(str, params))
		if branch:
			return f"{name}[{branch}]({args})"
		return f"{name}({args})"


def collect_recent_derivations(
	problem: Any,
	*,
	symbol_map: Optional[Dict[str, str]] = None,
	max_steps: int = 3,
	max_items: int = 12,
) -> List[str]:
	if symbol_map is None:
		symbol_map = build_symbol_map(problem)
	derivations: List[str] = []
	start_step = max(problem.condition.step_count - max_steps, 0)
	for step in range(problem.condition.step_count - 1, start_step - 1, -1):
		ids = problem.condition.ids_of_step.get(step, [])
		if not ids:
			continue
		for item_id in ids:
			predicate, payload, premise, theorem, _ = problem.condition.items[item_id]
			if predicate in problem.parsed_predicate_GDL["Preset"].get("Construction", {}):
				continue
			if predicate in problem.parsed_predicate_GDL["Preset"].get("BasicEntity", {}):
				continue
			if predicate == "Equation":
				try:
					statement = format_equation(problem, payload, symbol_map)
				except Exception:
					statement = str(payload)
			else:
				try:
					statement = inverse_parse_one(predicate, payload, problem)
				except Exception:
					statement = f"{predicate}{payload}"
			theorem_label = describe_theorem(problem, theorem)
			premise_text = ",".join(map(str, premise)) if premise else "-"
			derivations.append(f"{statement} | via {theorem_label} | premises={premise_text}")
			if len(derivations) >= max_items:
				return derivations
	return derivations


def collect_unsolved_equations(
	problem: Any,
	*,
	symbol_map: Optional[Dict[str, str]] = None,
	limit: int = 20,
) -> List[str]:
	if symbol_map is None:
		symbol_map = build_symbol_map(problem)
	entries: List[str] = []
	for eq_expr in problem.condition.simplified_equation.keys():
		if len(entries) >= limit:
			break
		if eq_expr is None:
			continue
		try:
			entries.append(format_equation(problem, eq_expr, symbol_map))
		except Exception:
			entries.append(str(eq_expr))
	return entries


def summarize_goal_progress(
	problem: Any,
	*,
	symbol_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
	goal = getattr(problem, "goal", None)
	if goal is None:
		return {}
	if symbol_map is None:
		symbol_map = build_symbol_map(problem)
	progress: Dict[str, Any] = {
		"type": goal.type,
		"solved": bool(getattr(goal, "solved", False)),
	}
	if goal.type == "algebra":
		try:
			progress["target"] = expr_to_readable(goal.item, symbol_map)
		except Exception:
			progress["target"] = str(goal.item)
		progress["expected"] = str(goal.answer)
		if goal.solved_answer is not None:
			progress["current"] = str(goal.solved_answer)
	elif goal.type == "logic":
		progress["target_predicate"] = goal.item
		progress["expected"] = tuple(goal.answer) if isinstance(goal.answer, tuple) else goal.answer
		if goal.solved_answer is not None:
			progress["current"] = goal.solved_answer
	if getattr(goal, "premise", None):
		progress["premise"] = list(goal.premise)
	if getattr(goal, "theorem", None):
		progress["theorem"] = describe_theorem(problem, goal.theorem)
	return progress


def compute_theorem_complexity(
	theorem_name: str,
	parsed_theorem_GDL: Dict[str, Any]
) -> float:
	return _compute_theorem_complexity(theorem_name, parsed_theorem_GDL)


def precheck_theorem_call(
    problem: Any,
    t_name: str,
    t_branch: Optional[str],
    t_para: Tuple[Any, ...],
    *,
    symbol_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    warnings_list: List[str] = []
    theorem_def = problem.parsed_theorem_GDL.get(t_name)
    if theorem_def is None:
        warnings_list.append(f"Theorem {t_name} is not defined in the current GDL.")
        return warnings_list
    branches = theorem_def.get("body", {})
    if t_branch not in branches:
        warnings_list.append(
            f"Theorem {t_name} has no branch {t_branch}. Available branches: {list(branches.keys())}"
        )
        return warnings_list
    vars_seq = theorem_def.get("vars", [])
    letters = {}
    for idx, var in enumerate(vars_seq):
        if idx < len(t_para):
            letters[var] = t_para[idx]
    resolved_symbol_map = symbol_map or build_symbol_map(problem)
    gpl = branches[t_branch]
    missing_conditions: List[str] = []
    logic_requirements = gpl.get("products", []) + gpl.get("logic_constraints", [])
    for predicate, item in logic_requirements:
        oppose = False
        if predicate.startswith("~"):
            oppose = True
            predicate = predicate.replace("~", "", 1)
        resolved_item = tuple(letters[val] for val in item)
        has_item = problem.condition.has(predicate, resolved_item)
        if (not oppose and not has_item) or (oppose and has_item):
            formatted = f"{predicate}{format_condition_component(resolved_item)}"
            if oppose:
                formatted = f"not {formatted}"
            missing_conditions.append(formatted)
    if missing_conditions:
        warnings_list.append("Prerequisites not satisfied: " + "; ".join(missing_conditions))
    algebra_issues: List[str] = []
    for _, item in gpl.get("algebra_constraints", []):
        try:
            eq_expr = get_equation_from_tree(problem, item, True, letters)
        except Exception:
            continue
        if not problem.condition.has("Equation", eq_expr):
            try:
                formatted_eq = format_equation(problem, eq_expr, resolved_symbol_map)
            except Exception:
                formatted_eq = str(eq_expr)
            algebra_issues.append(formatted_eq)
    if algebra_issues:
        warnings_list.append("Algebraic constraints pending: " + "; ".join(algebra_issues))
    return warnings_list


# =============================================================================
# SECTION 4.5: TPG analysis and planning
# =============================================================================

def merge_theorem_tpgs(
	tpgs: List[Dict[str, List[str]]],
	candidate_theorems: Set[str],
) -> Dict[str, List[str]]:
	return _merge_theorem_tpgs(tpgs, candidate_theorems)


def merge_theorem_tpgs_with_edge_weights(
	tpgs: List[Dict[str, List[str]]],
	candidate_theorems: Set[str],
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float]]:
	return _merge_theorem_tpgs_with_edge_weights(tpgs, candidate_theorems)


def merge_theorem_tpgs_with_edge_weights_branches(
	tpgs: List[Dict[Any, List[Tuple[str, int]]]],
	candidate_branches: Set[Tuple[str, int]],
) -> Tuple[Dict[Any, List[Tuple[str, int]]], Dict[Tuple[Any, Tuple[str, int]], float]]:
	return _merge_theorem_tpgs_with_edge_weights_branches(tpgs, candidate_branches)


def compute_tpg_dynamic_priorities(
	tpgs: List[Dict[str, List[str]]],
	candidate_theorems: Set[str],
) -> Tuple[Dict[str, float], Dict[str, int]]:
	return _compute_tpg_dynamic_priorities(tpgs, candidate_theorems)


def compute_tpg_dynamic_priorities_branches(
	tpgs: List[Dict[Any, List[Tuple[str, int]]]],
	candidate_branches: Set[Tuple[str, int]],
) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], int]]:
	return _compute_tpg_dynamic_priorities_branches(tpgs, candidate_branches)


def build_branch_level_tpg_info_for_llm(
	candidate_branches: List[Tuple[str, int]],
	merged_tpg: Dict[Any, List[Tuple[str, int]]],
	readiness_scores: Dict[Tuple[str, int], int],
	tpg_edge_weights: Dict[Tuple[Any, Tuple[str, int]], float],
	last_successful_theorem: Optional[str],
	applied_theorem_names: Set[str],
) -> Dict[str, Any]:
	return _build_branch_level_tpg_info_for_llm(
		candidate_branches,
		merged_tpg,
		readiness_scores,
		tpg_edge_weights,
		last_successful_theorem,
		applied_theorem_names,
	)


def build_branch_planning_suggestions(
	candidate_branches: List[Tuple[str, int]],
	merged_tpg: Dict[Any, List[Tuple[str, int]]],
	readiness_scores: Dict[Tuple[str, int], int],
	tpg_edge_weights: Dict[Tuple[Any, Tuple[str, int]], float],
	goal_type: str,
	applied_theorem_names: Set[str],
) -> List[Dict[str, Any]]:
	return _build_branch_planning_suggestions(
		candidate_branches,
		merged_tpg,
		readiness_scores,
		tpg_edge_weights,
		goal_type,
		applied_theorem_names,
	)


def analyze_branch_chains(
	merged_tpg: Dict[Any, List[Tuple[str, int]]],
	candidate_branches: List[Tuple[str, int]],
	tpg_edge_weights: Dict[Tuple[Any, Tuple[str, int]], float],
	applied_theorem_names: Set[str],
	max_depth: int = 3,
) -> List[Dict[str, Any]]:
	return _analyze_branch_chains(
		merged_tpg,
		candidate_branches,
		tpg_edge_weights,
		applied_theorem_names,
		max_depth=max_depth,
	)

# =============================================================================
# SECTION 5: FORWARD HINTS & CANDIDATE FILTERING
# =============================================================================

def format_conclusion(problem: Any, predicate: str, item: Any) -> str:
	try:
		if predicate == "Equation":
			return str(item)
		return inverse_parse_one(predicate, tuple(item), problem)
	except Exception:
		return f"{predicate}{item}"


def format_theorem_call(t_name: str, t_branch: Optional[str], t_para: Sequence[Any]) -> str:
	args = [str(x) for x in t_para]
	if t_branch not in {None, "", "None"}:
		args = [str(t_branch)] + args
	return f"{t_name}({','.join(args)})" if args else f"{t_name}()"


def collect_forward_hints(
	problem: Any,
	p2t_map: Dict[str, List[Tuple[str, str, Sequence[str]]]],
	candidate_set: Set[str],
	*,
	max_hints: int = 15,
	symbol_map: Optional[Dict[str, str]] = None,
	failed_calls: Optional[Dict[str, Dict[str, Any]]] = None,
	attempted_calls: Optional[Set[str]] = None,
) -> List[str]:
	if not p2t_map:
		return []
	if symbol_map is None:
		symbol_map = build_symbol_map(problem)
	
	# Initialize filtering sets
	if failed_calls is None:
		failed_calls = {}
	if attempted_calls is None:
		attempted_calls = set()
	
	related_pres: List[Tuple[str, str, Dict[str, str]]] = []
	related_syms: List[Any] = []
	# Check ALL steps, not just recent 5, to ensure we don't miss initial conditions
	start_step = 0
	for step in range(start_step, problem.condition.step_count):
		for _id in problem.condition.ids_of_step.get(step, []):
			predicate, item, _, _, _ = problem.condition.items[_id]
			if predicate == "Equation":
				try:
					for sym in item.free_symbols:
						if sym not in related_syms:
							related_syms.append(sym)
				except AttributeError:
					continue
			elif predicate in p2t_map:
				candidates = p2t_map[predicate]
				for t_name, t_branch, p_vars in candidates:
					if t_name not in candidate_set:
						continue
					if len(p_vars) != len(item):
						continue
					letters = {p_vars[idx]: item[idx] for idx in range(len(p_vars))}
					entry = (t_name, t_branch, letters)
					if entry not in related_pres:
						related_pres.append(entry)
	hints: List[str] = []
	seen_calls: Set[str] = set()
	# Logic-based hints
	for t_name, t_branch, t_letters in related_pres:
		try:
			gpl = problem.parsed_theorem_GDL[t_name]["body"][t_branch]
		except KeyError:
			continue
		try:
			results = GPLExecutor.run(gpl, problem, t_letters)
		except Exception:
			continue
		for letters_out, premise, conclusion in results:
			t_para = tuple(letters_out[var] for var in problem.parsed_theorem_GDL[t_name]["vars"])
			call = format_theorem_call(t_name, t_branch, t_para)
			if call in seen_calls:
				continue
			
			# Filter out failed or already attempted calls
			if call in failed_calls or call in attempted_calls:
				continue
			
			formatted_conclusions: List[str] = []
			for predicate, payload in conclusion:
				try:
					item_tuple = tuple(payload)
				except TypeError:
					item_tuple = payload
				if not problem.check(predicate, item_tuple, premise, t_name):
					continue
				formatted_conclusions.append(format_conclusion(problem, predicate, payload))
			if not formatted_conclusions:
				continue
			seen_calls.add(call)
			hints.append(f"{call} -> { '; '.join(formatted_conclusions) }")
			if len(hints) >= max_hints:
				return hints
	# Algebra-based hints
	if related_syms:
		paras_of_attrs: Dict[str, List[Tuple[Any, ...]]] = {}
		for sym in related_syms:
			if sym not in problem.condition.attr_of_sym:
				continue
			attr, paras = problem.condition.attr_of_sym[sym]
			if attr not in p2t_map:
				continue
			paras_of_attrs.setdefault(attr, [])
			for para in paras:
				if para not in paras_of_attrs[attr]:
					paras_of_attrs[attr].append(para)
		for attr, paras in paras_of_attrs.items():
			for t_name, t_branch, p_vars in p2t_map.get(attr, []):
				if t_name not in candidate_set:
					continue
				try:
					gpl = problem.parsed_theorem_GDL[t_name]["body"][t_branch]
				except KeyError:
					continue
				for para in paras:
					letters = {p_vars[idx]: para[idx] for idx in range(len(p_vars))}
					try:
						results = GPLExecutor.run(gpl, problem, letters)
					except Exception:
						continue
					for letters_out, premise, conclusion in results:
						t_para = tuple(letters_out[var] for var in problem.parsed_theorem_GDL[t_name]["vars"])
						call = format_theorem_call(t_name, t_branch, t_para)
						if call in seen_calls:
							continue
						
						# Filter out failed or already attempted calls
						if call in failed_calls or call in attempted_calls:
							continue
						
						formatted_conclusions: List[str] = []
						for predicate, payload in conclusion:
							try:
								item_tuple = tuple(payload)
							except TypeError:
								item_tuple = payload
							if not problem.check(predicate, item_tuple, premise, t_name):
								continue
							formatted_conclusions.append(format_conclusion(problem, predicate, payload))
						if not formatted_conclusions:
							continue
						seen_calls.add(call)
						hints.append(f"{call} -> { '; '.join(formatted_conclusions) }")
						if len(hints) >= max_hints:
							return hints
	return hints


def extract_call_from_hint(hint: str) -> Optional[str]:
	"""Return the theorem call portion from a forward hint."""
	if not hint:
		return None
	call_part, _, _ = hint.partition("->")
	call = call_part.strip()
	return call or None


def filter_candidates_by_precheck(
	candidates: List[Union[str, Tuple[str, int]]],
	problem: Any,
	*,
	max_check_attempts: int = 50,
) -> Tuple[Dict[str, Set[int]], Dict[Tuple[str, int], int], Dict[Tuple[str, int], List[str]]]:
	return _filter_candidates_by_precheck(candidates, problem, max_check_attempts=max_check_attempts)


def get_goal_type(problem: Any) -> str:
	return _get_goal_type(problem)


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
	return _adaptive_candidate_ranking(
		candidates,
		forward_hint_heads=forward_hint_heads,
		failed_calls=failed_calls,
		history_records=history_records,
		pending_prereqs=pending_prereqs,
		problem=problem,
		readiness_scores=readiness_scores,
		attempt_mode=attempt_mode,
		failed_theorem_name=failed_theorem_name,
		tpg_common_prefix_theorems=tpg_common_prefix_theorems,
		tpg_unlock_scores=tpg_unlock_scores,
		time_budget_ratio=time_budget_ratio,
		current_step=current_step,
		last_successful_theorem=last_successful_theorem,
		merged_tpg=merged_tpg,
		tpg_edge_weights=tpg_edge_weights,
		expected_steps=expected_steps,
	)


def prioritize_with_forward_hints(candidate_list: List[str], hint_heads: Sequence[str]) -> List[str]:
	if not hint_heads:
		return candidate_list
	seen: Set[str] = set()
	ordered: List[str] = []
	for head in hint_heads:
		if head in candidate_list and head not in seen:
			ordered.append(head)
			seen.add(head)
	for name in candidate_list:
		if name not in seen:
			ordered.append(name)
	return ordered

def collect_step_statements(problem: Any, step: int, symbol_map: Dict[str, str]) -> List[str]:
	statements: List[str] = []
	step_ids = problem.condition.ids_of_step.get(step, [])
	idx = 0
	while idx < len(step_ids):
		item_id = step_ids[idx]
		predicate, payload = problem.condition.items[item_id][0], problem.condition.items[item_id][1]

		if (
			predicate in problem.parsed_predicate_GDL["Preset"].get("Construction", {})
			or predicate in problem.parsed_predicate_GDL["Preset"].get("BasicEntity", {})
		):
			idx += 1
			continue

		if predicate == "Equation":
			try:
				statements.append(format_equation(problem, payload, symbol_map))
			except Exception:
				statements.append(str(payload))
			idx += 1
			continue

		statements.append(inverse_parse_one(predicate, payload, problem))
		if predicate in problem.parsed_predicate_GDL.get("Entity", {}):
			idx += len(problem.parsed_predicate_GDL["Entity"][predicate]["multi"]) + 1
		elif predicate in problem.parsed_predicate_GDL.get("Relation", {}):
			idx += len(problem.parsed_predicate_GDL["Relation"][predicate]["multi"]) + 1
		else:
			idx += 1

	return statements



def summarize_problem_state(problem: Any, symbol_map: Optional[Dict[str, str]] = None) -> str:
	if symbol_map is None:
		symbol_map = build_symbol_map(problem)
	statements: List[str] = []
	for step in range(problem.condition.step_count):
		statements.extend(collect_step_statements(problem, step, symbol_map))
	if statements:
		return "\n".join(statements)
	return "(no symbolic relations recorded)"


def gather_problem_state(problem: Any) -> Dict[str, Any]:
	symbol_map = build_symbol_map(problem)
	readable = summarize_problem_state(problem, symbol_map)
	raw_conditions = collect_raw_conditions(problem)
	symbol_mappings, known_values = collect_symbol_details(problem)
	recent_derivations = collect_recent_derivations(problem, symbol_map=symbol_map)
	unsolved_equations = collect_unsolved_equations(problem, symbol_map=symbol_map)
	goal_progress = summarize_goal_progress(problem, symbol_map=symbol_map)
	return {
		"readable_summary": readable,
		"raw_conditions": raw_conditions,
		"symbol_mappings": symbol_mappings,
		"known_symbol_values": known_values,
		"recent_derivations": recent_derivations,
		"unsolved_equations": unsolved_equations,
		"goal_progress": goal_progress,
	}


def append_feedback(feedback_list: List[str], message: str, limit: int = 10) -> None:
	feedback_list.append(message)
	excess = len(feedback_list) - limit
	if excess > 0:
		del feedback_list[:excess]


def local_image_to_data_url(image_path: Optional[Path]) -> str:
	if image_path is None or not image_path.exists():
		return ""
	mime_map = {
		".png": "image/png",
		".jpg": "image/jpeg",
		".jpeg": "image/jpeg",
		".gif": "image/gif",
		".webp": "image/webp",
	}
	mime = mime_map.get(image_path.suffix.lower(), "application/octet-stream")
	with open(image_path, "rb") as f:
		encoded = base64.b64encode(f.read()).decode("utf-8")
	return f"data:{mime};base64,{encoded}"


def prepare_embedding_store(mode: str, split: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Path]:
	return _prepare_embedding_store(mode, split)


def rank_results(
	query_vector: Sequence[float],
	entries: Dict[str, Dict[str, Any]],
	top_k: int,
	exclude_id: str,
) -> List[Tuple[str, float, Dict[str, Any]]]:
	return _rank_results(query_vector, entries, top_k, exclude_id)


def build_candidate_theorems(
	problem_id: int,
	*,
	problems_dir: Path,
	diagrams_dir: Path,
	mode: str,
	top_k: int,
	entries: Dict[str, Dict[str, Any]],
	embedder: MultiModalEmbedding,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any], Dict[str, List[str]], Dict[str, float], Dict[str, int], Dict[Tuple[str, str], float], int]:
	return _build_candidate_theorems(
		problem_id,
		problems_dir=problems_dir,
		diagrams_dir=diagrams_dir,
		mode=mode,
		top_k=top_k,
		entries=entries,
		embedder=embedder,
	)

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
	*,
	forward_hint_calls: Optional[Sequence[str]] = None,
	pending_prereqs: Optional[Sequence[Dict[str, Any]]] = None,
	recovery_payload: Optional[Dict[str, Any]] = None,
	attempt_mode: str = "normal",
	readiness_scores: Optional[Dict[Tuple[str, int], int]] = None,
	valid_branches: Optional[Dict[str, Set[int]]] = None,
	branch_details: Optional[Dict[Tuple[str, int], List[str]]] = None,
	branch_tpg_info: Optional[Dict[str, Any]] = None,
	branch_planning: Optional[List[Dict[str, Any]]] = None,
	branch_chains: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
	return _build_one_step_messages(
		problem,
		theorem_branches,
		gdl_map,
		image_data_url,
		state_payload,
		applied_calls,
		applied_history,
		recent_feedback,
		step_index,
		max_steps,
		forward_hints,
		forward_hint_calls=forward_hint_calls,
		pending_prereqs=pending_prereqs,
		recovery_payload=recovery_payload,
		attempt_mode=attempt_mode,
		readiness_scores=readiness_scores,
		valid_branches=valid_branches,
		branch_details=branch_details,
		branch_tpg_info=branch_tpg_info,
		branch_planning=branch_planning,
		branch_chains=branch_chains,
	)


def setup_logger(output_dir: Path, run_id: Optional[str] = None) -> Tuple[logging.Logger, str]:
	if run_id is None:
		run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
	logger = logging.getLogger("formalgeo.solve")
	logger.setLevel(logging.INFO)
	logger.propagate = False
	for handler in list(logger.handlers):
		logger.removeHandler(handler)
	formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	logger.info("Logger initialized (console only)")
	return logger, run_id


# =============================================================================
# SECTION 7: LLM INTERACTION & RESPONSE PARSING
# =============================================================================

def _build_problem_dict(problem: Dict[str, Any]) -> Dict[str, Any]:
	return _llm_build_problem_dict(problem)


def _reduce_theorem_call_for_llm(call: str) -> str:
	return _llm_reduce_theorem_call_for_llm(call)


def _redact_text_remove_objects(text: str) -> str:
	return _llm_redact_text_remove_objects(text)


def _build_redacted_problem_dict(problem: Dict[str, Any]) -> Dict[str, Any]:
	return _llm_build_redacted_problem_dict(problem)



def call_llm(
	client: Any,
	model: str,
	messages: List[Dict[str, Any]],
	temperature: float,
	max_retries: int,
	logger: Optional[logging.Logger] = None,
	api_keys: Optional[List[str]] = None,
	api_base: Optional[str] = None,
) -> str:
	return _call_llm(
		client,
		model,
		messages,
		temperature,
		max_retries,
		logger=logger,
		api_keys=api_keys,
		api_base=api_base,
	)

def extract_calls(raw_text: str) -> List[str]:
	return _extract_calls(raw_text)


def precheck_theorem_branch(problem: Any, t_name: str, t_branch: Optional[str], *, max_samples: int = 3) -> List[str]:
	"""Branch-level precheck used in only-predict-theorem mode.
	Do a lightweight existence check by running the branch GPl executor; if no
	parameter binding is possible, return a warning message list.
	"""
	warnings_list: List[str] = []
	theorem_def = problem.parsed_theorem_GDL.get(t_name)
	if theorem_def is None:
		warnings_list.append(f"Theorem {t_name} is not defined in the current GDL.")
		return warnings_list
	branches = theorem_def.get("body", {})
	if t_branch is not None and t_branch not in branches:
		warnings_list.append(f"Theorem {t_name} has no branch {t_branch}.")
		return warnings_list

	# Choose branch to probe: if t_branch is None, probe all branches until one yields results
	to_check = [t_branch] if t_branch is not None else list(branches.keys())
	found_any = False
	for b in to_check:
		try:
			gpl = branches[b]
		except Exception:
			continue
		try:
			results = GPLExecutor.run(gpl, problem)
		except Exception:
			# If GPL execution fails, skip this branch for precheck (do not treat as fatal)
			results = []
		if results:
			found_any = True
			break

	if not found_any:
		warnings_list.append(f"No valid parameter bindings found for theorem {t_name} branch {t_branch} in current state.")
	return warnings_list


def collect_applied_instances_since(problem: Any, start_step: int) -> List[Dict[str, Any]]:
	"""Collect applied theorem instances (with parameters) from problem.condition
	for steps in [start_step, ..., current_step-1]. Returns list of dicts with
	keys: 'theorem' (human label) and 'conclusions' (list of readable conclusions).
	"""
	instances_map: Dict[str, Dict[str, Any]] = {}
	curr = problem.condition.step_count
	for step in range(start_step, curr):
		ids = problem.condition.ids_of_step.get(step, [])
		for _id in ids:
			try:
				predicate, payload, premise, theorem, _ = problem.condition.items[_id]
			except Exception:
				continue
			if not theorem or not isinstance(theorem, (tuple, list)):
				continue
			t_name = theorem[0]
			if t_name in {None, 'prerequisite', 'extended', 'init_problem', 'check_goal', 'solve_equations'}:
				continue
			try:
				human = inverse_parse_one_theorem(theorem, problem.parsed_theorem_GDL)
			except Exception:
				human = f"{theorem[0]}[{theorem[1]}]({theorem[2]})"
			entry = instances_map.setdefault(human, {"theorem": human, "conclusions": []})
			# format conclusion
			try:
				conclusion_text = format_conclusion(problem, predicate, payload)
			except Exception:
				conclusion_text = f"{predicate}{payload}"
			if conclusion_text not in entry["conclusions"]:
				entry["conclusions"].append(conclusion_text)

	return list(instances_map.values())


def build_feedback_from_instances(instances: List[Dict[str, Any]], *, top_k: int = 5) -> List[str]:
	msgs: List[str] = []
	for inst in instances[:top_k]:
		th = inst.get("theorem")
		conclusions = inst.get("conclusions", [])
		if conclusions:
			msgs.append(f"Applied {th} -> { '; '.join(conclusions) }.")
		else:
			msgs.append(f"Applied {th}.")
	return msgs


# =============================================================================
# DYNAMIC ENTITY WRAPPER DETECTION & PREREQUISITE SUGGESTION
# =============================================================================

def is_wrapped_entity(token: str) -> bool:
	return _is_wrapped_entity(token)


# Global cache for predicate->theorem mappings loaded from GDL
_PREDICATE_TO_THEOREMS_CACHE: Optional[Dict[str, List[str]]] = None
_THEOREM_GDL_CACHE: Optional[Dict[str, Any]] = None


def load_predicate_theorem_mapping(theorem_gdl_path: Path) -> Dict[str, List[str]]:
	"""
	Build dynamic predicate->theorem mapping from theorem_GDL.json.
	
	This analyzes all theorem conclusions to determine which theorems can produce
	each predicate type. Replaces the hardcoded PREREQ_SUGGESTION_LOOKUP.
	
	Returns:
		Dict mapping predicate names to lists of theorem names that can produce them.
	"""
	global _PREDICATE_TO_THEOREMS_CACHE, _THEOREM_GDL_CACHE
	
	if _PREDICATE_TO_THEOREMS_CACHE is not None:
		return _PREDICATE_TO_THEOREMS_CACHE
	
	if not theorem_gdl_path.exists():
		return {}
	
	with open(theorem_gdl_path, "r", encoding="utf-8") as f:
		theorem_gdl = json.load(f)
	
	_THEOREM_GDL_CACHE = theorem_gdl
	
	pred_to_theorems: Dict[str, Set[str]] = {}
	
	for theorem_full_name, branches in theorem_gdl.items():
		theorem_name = theorem_full_name.split("(")[0]
		
		for branch_data in branches.values():
			if not isinstance(branch_data, dict):
				continue
			
			conclusions = branch_data.get("conclusion", [])
			for conclusion_item in conclusions:
				if not isinstance(conclusion_item, str):
					continue
				
				# Extract predicate name from conclusion
				predicate = conclusion_item.split("(")[0]
				
				if predicate not in pred_to_theorems:
					pred_to_theorems[predicate] = set()
				pred_to_theorems[predicate].add(theorem_name)
	
	# Convert to sorted lists and cache
	_PREDICATE_TO_THEOREMS_CACHE = {
		pred: sorted(list(thms)) for pred, thms in pred_to_theorems.items()
	}
	
	return _PREDICATE_TO_THEOREMS_CACHE


def suggest_theorems_for_predicate(
	predicate: str,
	candidates: Sequence[str],
	*,
	theorem_gdl_path: Optional[Path] = None,
	max_suggestions: int = 8,
) -> List[str]:
	"""
	Dynamically suggest theorems that can produce a missing predicate.
	
	Strategy:
	1. Load predicate->theorem mapping from GDL (cached)
	2. Find theorems that directly produce this predicate
	3. Use semantic name matching as fallback
	4. Filter by candidate availability
	
	This replaces the hardcoded PREREQ_SUGGESTION_LOOKUP with dynamic analysis.
	
	Args:
		predicate: Missing predicate name (e.g., "Collinear", "ParallelBetweenLine")
		candidates: Available theorem names to choose from
		theorem_gdl_path: Path to theorem_GDL.json (uses default if None)
		max_suggestions: Maximum number of suggestions to return
	
	Returns:
		List of theorem names that can establish the missing predicate
	"""
	if theorem_gdl_path is None:
		theorem_gdl_path = DEFAULT_DATASETS_PATH / DEFAULT_DATASET_NAME / "gdl" / "theorem_GDL.json"
	
	# Load mapping (cached after first call)
	mapping = load_predicate_theorem_mapping(theorem_gdl_path)
	
	suggestions: List[str] = []
	candidate_set = set(candidates)
	
	# Strategy 1: Direct mapping from GDL analysis
	direct_theorems = mapping.get(predicate, [])
	for theorem_name in direct_theorems:
		if theorem_name in candidate_set and theorem_name not in suggestions:
			suggestions.append(theorem_name)
	
	# Strategy 2: Semantic name matching (fallback)
	if len(suggestions) < max_suggestions:
		predicate_lower = predicate.lower()
		predicate_keywords = set(predicate_lower.split("_")) | {predicate_lower}
		
		for theorem_name in candidates:
			if theorem_name in suggestions:
				continue
			
			theorem_lower = theorem_name.lower()
			
			# Check if any significant keyword from predicate appears in theorem name
			if any(keyword in theorem_lower for keyword in predicate_keywords if len(keyword) > 3):
				suggestions.append(theorem_name)
				if len(suggestions) >= max_suggestions:
					break
	
	return suggestions[:max_suggestions]

def _split_call_arguments(text: str) -> List[str]:
	parts: List[str] = []
	current: List[str] = []
	depth = 0
	for ch in text:
		if ch == "," and depth == 0:
			parts.append("".join(current))
			current = []
			continue
		if ch == "(":
			depth += 1
		elif ch == ")":
			depth = max(depth - 1, 0)
		current.append(ch)
	if current:
		parts.append("".join(current))
	return parts


def _normalize_argument(token: str) -> str:
	"""
	Normalize an argument by collapsing wrapped entity prefixes.
	
	Now uses dynamic detection via is_wrapped_entity() instead of hardcoded prefixes.
	Examples:
		- Triangle(A,B,C) -> ABC
		- Line(A,B) -> AB
		- Segment(M,N) -> MN
	"""
	token = token.strip()
	if not token:
		return token
	
	# Use dynamic detection instead of hardcoded list
	if is_wrapped_entity(token):
		# Extract content between parentheses
		first_paren = token.find("(")
		if first_paren != -1 and token.endswith(")"):
			inner = token[first_paren + 1 : -1]
			inner_parts = _split_call_arguments(inner)
			normalized_inner = [_normalize_argument(part) for part in inner_parts]
			collapsed = "".join(normalized_inner)
			if collapsed:
				return collapsed
			return "".join(ch for ch in inner if ch.isalnum())
	
	return token


def _normalize_argument_list(text: str) -> str:
	parts = _split_call_arguments(text)
	if not parts:
		return text
	return ",".join(_normalize_argument(part) for part in parts)


def _normalize_wrapped_entities(call: str) -> str:
	first_paren = call.find("(")
	if first_paren == -1:
		return call
	head = call[:first_paren]
	body = call[first_paren + 1 :]
	if not body:
		return call
	depth = 1
	inner_chars: List[str] = []
	suffix = ""
	for idx, ch in enumerate(body):
		if ch == "(":
			depth += 1
			inner_chars.append(ch)
		elif ch == ")":
			depth -= 1
			if depth == 0:
				suffix = body[idx + 1 :]
				break
			inner_chars.append(ch)
		else:
			inner_chars.append(ch)
	if depth != 0:
		return call
	inner = "".join(inner_chars)
	normalized_inner = _normalize_argument_list(inner)
	return f"{head}({normalized_inner}){suffix}"


def normalize_theorem_call(call: str) -> str:
	return _normalize_theorem_call(call)


def ensure_theorem_call_has_branch(call: str, default_branch: int = 1) -> str:
	return _ensure_theorem_call_has_branch(call, default_branch=default_branch)


def parse_missing_prereq_items(message: str) -> List[str]:
	if not message:
		return []
	body = message.split(":", 1)[1] if ":" in message else message
	results: List[str] = []
	for chunk in body.replace(" and ", ";").split(";"):
		item = chunk.strip().strip(".")
		if not item:
			continue
		if item.lower().startswith("and "):
			item = item[4:].strip()
		results.append(item)
	return results


def _parse_requirement(requirement: str) -> Optional[Tuple[str, Tuple[str, ...], bool]]:
	text = requirement.strip()
	is_negative = False
	if text.lower().startswith("not "):
		is_negative = True
		text = text[4:].strip()
	if "(" not in text or not text.endswith(")"):
		return None
	predicate, args_text = text.split("(", 1)
	args_text = args_text[:-1]
	args = tuple(token.strip() for token in args_text.split(",") if token.strip())
	return predicate.strip(), args, is_negative


# =============================================================================
# SECTION 6: DEPENDENCY MANAGEMENT & PREREQUISITE HANDLING
# =============================================================================

def suggest_theorems_for_requirement(
	requirement: str,
	candidates: Sequence[str],
	*,
	theorem_gdl_path: Optional[Path] = None,
) -> List[str]:
	"""
	Suggest theorems to establish a missing prerequisite requirement.
	
	Now uses dynamic GDL analysis instead of hardcoded PREREQ_SUGGESTION_LOOKUP.
	
	Args:
		requirement: Missing requirement string (e.g., "Collinear(ABC)")
		candidates: Available theorem names
		theorem_gdl_path: Path to theorem_GDL.json (optional)
	
	Returns:
		List of suggested theorem names
	"""
	parsed = _parse_requirement(requirement)
	if parsed is None:
		return []
	
	predicate, _, _ = parsed
	
	# Use new dynamic suggestion function
	return suggest_theorems_for_predicate(
		predicate,
		candidates,
		theorem_gdl_path=theorem_gdl_path,
		max_suggestions=5,
	)


def detect_circular_dependency(
	pending_prereqs: Dict[str, Dict[str, Any]],
	max_depth: int = 10
) -> List[List[str]]:
	"""
	Detect circular dependencies in prerequisite requirements.
	Returns a list of circular dependency chains.
	"""
	cycles: List[List[str]] = []
	
	def find_cycle(current: str, visited: List[str], depth: int) -> Optional[List[str]]:
		if depth > max_depth:
			return None
		if current in visited:
			# Found a cycle
			cycle_start = visited.index(current)
			return visited[cycle_start:] + [current]
		
		entry = pending_prereqs.get(current)
		if entry is None:
			return None
		
		suggestions = entry.get("suggestions", [])
		for suggestion in suggestions:
			# Check if this suggestion leads to another pending prereq
			for other_req in pending_prereqs.keys():
				if suggestion.lower() in other_req.lower():
					cycle = find_cycle(other_req, visited + [current], depth + 1)
					if cycle is not None:
						return cycle
		return None
	
	for req in pending_prereqs.keys():
		cycle = find_cycle(req, [], 0)
		if cycle is not None and cycle not in cycles:
			cycles.append(cycle)
	
	return cycles


def break_circular_dependencies(
	pending_prereqs: Dict[str, Dict[str, Any]],
	cycles: List[List[str]]
) -> None:
	"""
	Break circular dependencies by removing problematic suggestions.
	Modifies pending_prereqs in place.
	"""
	for cycle in cycles:
		if len(cycle) < 2:
			continue
		# Remove the last suggestion that causes the cycle
		first_req = cycle[0]
		if first_req in pending_prereqs:
			# Clear suggestions that might cause cycles
			entry = pending_prereqs[first_req]
			entry["suggestions"] = []


def is_requirement_satisfied(problem: Any, requirement: str) -> bool:
	parsed = _parse_requirement(requirement)
	if parsed is None:
		return False
	predicate, args, is_negative = parsed
	try:
		present = problem.condition.has(predicate, tuple(args))
	except Exception:
		return False
	return not present if is_negative else bool(present)


def prioritize_candidate_list(candidates: List[str], pending: Sequence[Dict[str, Any]]) -> List[str]:
	if not pending:
		return candidates
	ordered: List[str] = []
	seen: Set[str] = set()
	for entry in pending:
		for name in entry.get("suggestions", []):
			if name in candidates and name not in seen:
				ordered.append(name)
				seen.add(name)
	for name in candidates:
		if name not in seen:
			ordered.append(name)
	return ordered


def build_prereq_hints(pending: Sequence[Dict[str, Any]]) -> List[str]:
	hints: List[str] = []
	for entry in pending:
		req = entry.get("requirement")
		if not req:
			continue
		suggestions = entry.get("suggestions") or []
		if suggestions:
			hints.append(f"Establish {req} via {', '.join(suggestions[:3])}.")
		else:
			hints.append(f"Establish prerequisite: {req}.")
	return hints


def canonicalize_theorem_call(problem: Any, call: str) -> str:
	"""Return a solver-friendly call rendered in the same format as ground truth."""
	if not call:
		return ""
	try:
		parsed = parse_theorem_seqs([call])
	except Exception:
		return ensure_theorem_call_has_branch(call)
	if not parsed:
		return ensure_theorem_call_has_branch(call)
	try:
		return ensure_theorem_call_has_branch(inverse_parse_one_theorem(parsed[0], problem.parsed_theorem_GDL))
	except Exception:
		return ensure_theorem_call_has_branch(call)


# =============================================================================
# SECTION 8: SOLVER EXECUTION (ONE-STEP & BATCH)
# =============================================================================

def run_symbolic_solver(
	dataset_loader: DatasetLoader,
	problem: Dict[str, Any],
	predicted_calls: Sequence[str],
	logger: Optional[logging.Logger] = None,
) -> Tuple[Interactor, List[Dict[str, Any]]]:
	solver = Interactor(dataset_loader.predicate_GDL, dataset_loader.theorem_GDL)
	solver.load_problem(problem)
	step_logs: List[Dict[str, Any]] = []
	for call in predicted_calls:
		parsed = parse_theorem_seqs([call])
		for t_name, t_branch, t_para in parsed:
			start = time.time()
			solver.apply_theorem(t_name, t_branch, None)
			try:
				func_timeout(120.0, solver.problem.check_goal)
			except FunctionTimedOut:
				if logger:
					logger.warning("[CheckGoal][Timeout] check_goal timeout after 120s in execute_with_ground_truth")
			
			log_entry = {
				"call": call,
				"name": t_name,
				"branch": t_branch,
				"para": list(t_para) if t_para else None,
				"duration": round(time.time() - start, 6),
				"goal_solved": bool(getattr(solver.problem.goal, "solved", False)),
			}
			step_logs.append(log_entry)
			if logger:
				logger.info(
					"[Solver] call=%s branch=%s goal_solved=%s duration=%.4f",
					log_entry["call"],
					log_entry["branch"],
					log_entry["goal_solved"],
					log_entry["duration"],
				)
	return solver, step_logs


def run_quick_start_replay_solver(
	dataset_loader: DatasetLoader,
	problem: Dict[str, Any],
	replay_calls: Sequence[str],
	logger: Optional[logging.Logger] = None,
) -> Tuple[Interactor, List[str], List[Dict[str, Any]], str, List[Dict[str, Any]]]:
	"""Run solver from replayed theorem calls without any LLM/embedding dependency."""
	solver = Interactor(dataset_loader.predicate_GDL, dataset_loader.theorem_GDL)
	solver.load_problem(problem)

	predicted_calls: List[str] = []
	step_logs: List[Dict[str, Any]] = []
	history_records: List[Dict[str, Any]] = []
	termination_reason = "quick_start_replay_exhausted"

	for step_idx, raw_call in enumerate(replay_calls, 1):
		call = ensure_theorem_call_has_branch(str(raw_call or ""))
		if not call:
			continue
		# Simulate one virtual LLM call latency per replay step.
		time.sleep(1.0)
		predicted_calls.append(call)

		updated_any = False
		goal_solved = bool(getattr(solver.problem.goal, "solved", False))
		warnings_list: List[str] = []
		error_msg: Optional[str] = None

		parsed = parse_theorem_seqs([call])
		if not parsed:
			warnings_list.append("parse_failed")
		else:
			for t_name, t_branch, _ in parsed:
				start = time.time()
				try:
					# In quick-start replay, suppress lower-level warnings.
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						updated = solver.apply_theorem(t_name, t_branch, None)
					updated_any = updated_any or bool(updated)
					try:
						with warnings.catch_warnings():
							warnings.simplefilter("ignore")
							func_timeout(120.0, solver.problem.check_goal)
					except FunctionTimedOut:
						warnings_list.append("check_goal_timeout")
				except Exception as exc:
					error_msg = str(exc)
				duration = round(time.time() - start, 6)
				goal_solved = bool(getattr(solver.problem.goal, "solved", False))
				step_logs.append(
					{
						"call": call,
						"name": t_name,
						"branch": t_branch,
						"duration": duration,
						"goal_solved": goal_solved,
						"update": bool(updated_any),
						"warnings": list(warnings_list),
					}
				)

		history_entry: Dict[str, Any] = {
			"step": step_idx,
			"call": call,
			"updated": bool(updated_any),
			"goal_solved": bool(goal_solved),
			"warnings": list(warnings_list),
		}
		if error_msg:
			history_entry["error"] = error_msg
		history_records.append(history_entry)

		if logger:
			logger.info("[QuickStart][Step %s] Executed: %s | Updated: %s | Goal solved: %s", step_idx, call, bool(updated_any), bool(goal_solved))

		if goal_solved:
			termination_reason = "goal_solved"
			break

	if not replay_calls:
		termination_reason = "quick_start_no_replay"

	return solver, predicted_calls, step_logs, termination_reason, history_records


def run_one_step_solver(
	dataset_loader: DatasetLoader,
	problem: Dict[str, Any],
	candidate_theorems: Sequence[Tuple[str, int]],
	gdl_map: Dict[str, Dict[str, Any]],
	image_data_url: str,
	*,
	merged_tpg: Dict[Any, List[Tuple[str, int]]],
	tpg_common_prefix: Dict[Tuple[str, int], float],
	tpg_unlock_scores: Dict[Tuple[str, int], int],
	tpg_edge_weights: Optional[Dict[Tuple[Any, Tuple[str, int]], float]] = None,
	expected_steps: int = 0,
	llm_client: Any,
	args: argparse.Namespace,
	logger: Optional[logging.Logger],
	problem_id: int,
) -> Tuple[Interactor, List[str], List[Dict[str, Any]], List[List[Dict[str, Any]]], List[str], str, List[Dict[str, Any]], Dict[str, float]]:
	# Track wall-clock time for timeout handling.
	solver_start_time = time.time()
	time_limit = getattr(args, 'time_limit', 300)

	# Initialize solver.
	solver = Interactor(dataset_loader.predicate_GDL, dataset_loader.theorem_GDL)
	solver.load_problem(problem)

	# Initialize progress tracker for timeout recovery.
	global _worker_progress_tracker
	_worker_progress_tracker = {
		"problem_id": problem_id,
		"steps_completed": 0,
		"predicted_calls": [],
		"history_records": [],
		"timing_stats": {"start_time": solver_start_time},
		"last_step_info": None,
		"goal_solved": False,
	}
	
	predicted_calls: List[str] = []
	step_logs: List[Dict[str, Any]] = []
	llm_requests: List[List[Dict[str, Any]]] = []
	llm_responses: List[str] = []
	history_records: List[Dict[str, Any]] = []
	recent_feedback: List[str] = []

	applied_theorem_names: Set[str] = set()

	last_successful_theorem: Optional[str] = None

	call_attempts: Dict[str, int] = defaultdict(int)
	failed_call_reasons: Dict[str, Dict[str, Any]] = {}
	pending_prereqs: Dict[str, Dict[str, Any]] = {}
	successful_calls: Set[str] = set()

	initial_candidate_theorems = list(candidate_theorems)
	candidate_list = list(initial_candidate_theorems)
	candidate_set = set(candidate_list)

	detailed_timings = {
		"forward_hints": 0.0,
		"precheck_ranking": 0.0,
		"tpg_analysis": 0.0,
		"llm_inference": 0.0,
		"symbolic_solver": 0.0,
	}
	previous_state_snapshot: Optional[Dict[str, Any]] = None
	last_step_delta: Optional[Dict[str, Any]] = None

	forward_hint_map: Optional[Dict[str, List[Tuple[str, str, Sequence[str]]]]] = None
	t_info = {name: (0, 1) for name in solver.parsed_theorem_GDL}
	forward_hint_map = get_p2t_map_fw(t_info, solver.parsed_theorem_GDL)
	quick_start_enabled = bool(getattr(args, "quick_start", False))
	quick_start_replay_map = getattr(args, "_quick_start_replay_map", {}) or {}
	
	termination_reason = "max_steps_exhausted"
	abort_loop = False
	BASE_CONSECUTIVE_NO_UPDATE_STEPS = 3
	consecutive_no_update_steps = 0
	
	def get_dynamic_max_no_update(current_step: int, elapsed: float, limit: float) -> int:
		"""Compute the max consecutive no-update steps based on time/step."""
		remaining_ratio = (limit - elapsed) / limit if limit > 0 else 0

		if current_step < 5:
			max_steps = 5
		elif current_step < 15:
			max_steps = 3
		else:
			max_steps = 2

		if remaining_ratio < 0.2:
			max_steps = min(max_steps, 1)
		elif remaining_ratio < 0.4:
			max_steps = min(max_steps, 2)
		
		return max_steps

	# Main loop: predict and execute theorems.
	
	for step_idx in range(1, args.one_step_max_steps + 1):
		if abort_loop:
			break

		elapsed_time = time.time() - solver_start_time
		if elapsed_time > time_limit:
			if logger:
				logger.warning(
					"[TIMEOUT] problem=%s step=%s elapsed=%.2fs limit=%ds - aborting solver",
					problem_id, step_idx, elapsed_time, time_limit
				)
			termination_reason = "timeout"
			abort_loop = True
			break
		
		# Reset candidates at each step.
		candidate_list = list(initial_candidate_theorems)
		
		attempt_mode = "normal"
		recovery_payload: Optional[Dict[str, Any]] = None
		recovery_attempts = 0
		inner_iter_count = 0

		valid_branches: Dict[str, Set[int]] = {}
		readiness_scores: Dict[Tuple[str, int], float] = {}
		branch_details: Dict[Tuple[str, int], Dict[str, Any]] = {}
		candidate_list_with_branches: List[Tuple[str, int]] = []
		candidate_set: Set[Tuple[str, int]] = set()
		candidate_theorem_names: Set[str] = set()

		while True:
			inner_iter_count += 1

			if inner_iter_count > 20:
				if logger:
					logger.error(
						"[LOOP-GUARD] problem=%s step=%s inner_iter=%s - breaking infinite loop!",
						problem_id, step_idx, inner_iter_count
					)
				termination_reason = "loop_guard"
				abort_loop = True
				break
			
			# Timeout check inside inner loop.
			elapsed_time = time.time() - solver_start_time
			if elapsed_time > time_limit:
				if logger:
					logger.warning(
						"[TIMEOUT] problem=%s step=%s elapsed=%.2fs limit=%ds - aborting inner loop",
						problem_id, step_idx, elapsed_time, time_limit
					)
				termination_reason = "timeout"
				abort_loop = True
				break
			
			# Ensure candidate_list remains a list of strings for precheck.
			if isinstance(candidate_list, list) and candidate_list and isinstance(candidate_list[0], tuple):
				candidate_list = list(initial_candidate_theorems)
			
			# --- PHASE 1: Resolve prerequisites and detect circular dependencies ---
			if pending_prereqs:
				resolved = [req for req in list(pending_prereqs.keys()) if is_requirement_satisfied(solver.problem, req)]
				for req in resolved:
					pending_prereqs.pop(req, None)
				
				# Detect and break circular dependencies
				cycles = detect_circular_dependency(pending_prereqs)
				if cycles and logger:
					logger.warning(
						"[OneStep] problem=%s step=%s detected_cycles=%s",
						problem_id,
						step_idx,
						len(cycles),
					)
				break_circular_dependencies(pending_prereqs, cycles)
			
			# Initialize early variables
			symbol_map = build_symbol_map(solver.problem)
			forward_hints: List[str] = []
			forward_hint_calls: List[str] = []
			forward_hint_heads: List[Tuple[str, int]] = []
			forward_hint_new_heads: List[Tuple[str, int]] = []
			
			# --- PHASE 2: Collect forward hints FIRST ---
			candidate_set = set(candidate_list)
			# collect_forward_hints indexes by theorem name.
			candidate_names = {thm_name for thm_name, _ in candidate_list}
			
			# Populate forward hints if enabled (must do this before ranking)
			if forward_hint_map:
				fh_start = time.time()
				# Suppress FV warnings during forward hint collection
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					forward_hints = collect_forward_hints(
						solver.problem,
						forward_hint_map,
						candidate_names,
						failed_calls=failed_call_reasons,
						attempted_calls=set(predicted_calls),  # Filter out already tried calls
					)
				detailed_timings["forward_hints"] += time.time() - fh_start
				
				for hint in forward_hints:
					call_text = extract_call_from_hint(hint)
					if not call_text:
						continue
					forward_hint_calls.append(call_text)
					head_tuple = extract_theorem_name_and_branch(call_text)
					if head_tuple:
						forward_hint_heads.append(head_tuple)
						if head_tuple not in candidate_set:
							candidate_list.append(head_tuple)
							candidate_set.add(head_tuple)
							forward_hint_new_heads.append(head_tuple)
			
			# --- PHASE 3: Intelligent candidate ranking and filtering ---
			# Only run precheck/TPG on first attempt; recovery mode reuses existing results
			if attempt_mode != "recovery":
				precheck_start = time.time()
				# First, calculate readiness scores for ALL candidates
				valid_branches, readiness_scores, branch_details = filter_candidates_by_precheck(
					candidate_list,
					solver.problem,
				)

				candidate_list_with_branches = candidate_list.copy()
			
				# Maintain two sets: one for (name, branch) pairs, one for theorem names only
				candidate_set: Set[Tuple[str, int]] = set(candidate_list_with_branches)
				candidate_theorem_names: Set[str] = {name for name, _ in candidate_list_with_branches}
				
				# --- PHASE 3.5: TPG analysis (before ranking) ---
				tpg_start = time.time()
			
				branch_tpg_info: Dict[str, Any] = {}
				branch_planning: List[Dict[str, Any]] = []
				branch_chains: List[Dict[str, Any]] = []
				
				if merged_tpg:
					goal_type = get_goal_type(solver.problem)
					
					branch_tpg_info = build_branch_level_tpg_info_for_llm(
						candidate_list_with_branches,
						merged_tpg,
						readiness_scores,
						tpg_edge_weights,
						last_successful_theorem,
						applied_theorem_names,
					)
					branch_planning = build_branch_planning_suggestions(
						candidate_list_with_branches,
						merged_tpg,
						readiness_scores,
						tpg_edge_weights,
						goal_type,
						applied_theorem_names,
					)
					branch_chains = analyze_branch_chains(
						merged_tpg,
						candidate_list_with_branches,
						tpg_edge_weights,
						applied_theorem_names,
					)
					
				# Log TPG analysis summary.
				if logger:
					readiness_summary = branch_tpg_info.get("readiness_summary", {})
					highly_ready = len(readiness_summary.get("highly_ready", []))
					
					tpg_guidance = branch_tpg_info.get("tpg_guidance")
					recommended_next = len(tpg_guidance.get("recommended_next", [])) if tpg_guidance else 0
					
					logger.info(
						"[Step %s][TPG-Branch] highly_ready=%s recommended=%s planning=%s chains=%s",
						step_idx,
						highly_ready,
						recommended_next,
						len(branch_planning),
						len(branch_chains),
					)
				
				detailed_timings["tpg_analysis"] += time.time() - tpg_start
			else:
				pass
			
			# Extract failed theorem name for recovery mode (needed for ranking)
			failed_theorem_name = None
			if attempt_mode == "recovery" and recovery_payload:
				failed_call = recovery_payload.get("failed_call")
				if failed_call:
					failed_theorem_name = extract_theorem_name(failed_call)
			
			elapsed_time = time.time() - solver_start_time
			time_budget_ratio = max(0.0, (time_limit - elapsed_time) / time_limit)
			
			# Now rank at branch level using readiness scores
			original_count = len(candidate_list_with_branches)
			candidate_list_with_branches = adaptive_candidate_ranking(
				candidate_list_with_branches,
				forward_hint_heads=forward_hint_heads,
				failed_calls=failed_call_reasons,
				history_records=history_records,
				pending_prereqs=pending_prereqs,
				problem=solver.problem,
				readiness_scores=readiness_scores,
				attempt_mode=attempt_mode,
				failed_theorem_name=failed_theorem_name,
				tpg_common_prefix_theorems=tpg_common_prefix,
				tpg_unlock_scores=tpg_unlock_scores,
				time_budget_ratio=time_budget_ratio,
				current_step=step_idx,
				last_successful_theorem=last_successful_theorem,
				merged_tpg=merged_tpg,
				tpg_edge_weights=tpg_edge_weights,
				expected_steps=expected_steps,
			)
			detailed_timings["precheck_ranking"] += time.time() - precheck_start
			
			# From this point on, candidate_list contains (theorem_name, branch_idx) tuples
			candidate_list = candidate_list_with_branches
			
			# --- RECOVERY FILTERING: Avoid repeating recently failed theorems ---
			# In recovery mode, remove the theorem that just failed to prevent infinite loops
			# NOTE: The ranking function already applied -20 penalty to this theorem,
			# but we also physically remove it here as a double safeguard.
			if attempt_mode == "recovery" and failed_theorem_name:
				original_count_before_recovery = len(candidate_list)
				# Remove ALL branches of the failed theorem
				candidate_list = [
					(name, branch) for name, branch in candidate_list
					if name != failed_theorem_name
				]
				removed_by_recovery = original_count_before_recovery - len(candidate_list)
				if logger and removed_by_recovery > 0:
					logger.info(
						"[OneStep][Recovery] problem=%s step=%s removed_failed_theorem=%s branches_removed=%s",
						problem_id,
						step_idx,
						failed_theorem_name,
						removed_by_recovery,
					)
			
			# Log filtering results
			if logger:
				filtered_count = len(candidate_list)
				removed_count = original_count - filtered_count
				# Count readiness across all branches (new 4-level scoring)
				ready_count = sum(1 for s in readiness_scores.values() if s >= 3.0)  # 3.0
				high_partial_count = sum(1 for s in readiness_scores.values() if 2.5 <= s < 3.0)  # 2.5
				partial_count = sum(1 for s in readiness_scores.values() if 2.0 <= s < 2.5)  # 2.0
				low_partial_count = sum(1 for s in readiness_scores.values() if 1.5 <= s < 2.0)  # 1.5
				uncertain_count = sum(1 for s in readiness_scores.values() if 1.0 <= s < 1.5)  # 1.0
				low_count = sum(1 for s in readiness_scores.values() if s < 1.0)  # 0.0
				logger.info(
					"[OneStep][Precheck] problem=%s step=%s original=%s filtered=%s removed=%s (ready=%s high_partial=%s partial=%s low_partial=%s uncertain=%s low=%s)",
					problem_id,
					step_idx,
					original_count,
					filtered_count,
					removed_count,
					ready_count,
					high_partial_count,
					partial_count,
					low_partial_count,
					uncertain_count,
					low_count,
				)
			
			# From this point on, candidate_list contains (theorem_name, branch_idx) tuples
			candidate_list = candidate_list_with_branches
			
			# Limit candidates passed to the LLM.
			MAX_CANDIDATES_FOR_LLM = 30
			if len(candidate_list) > MAX_CANDIDATES_FOR_LLM:
				candidate_list = candidate_list[:MAX_CANDIDATES_FOR_LLM]
				if logger:
					logger.info(
						"[OneStep][Truncate] problem=%s step=%s truncated_to=%s (from %s candidates)",
						problem_id,
						step_idx,
						MAX_CANDIDATES_FOR_LLM,
						len(candidate_list_with_branches),
					)
			
			# --- PHASE 4: Gather problem state and build messages ---
			state_payload = gather_problem_state(solver.problem)
			
			# Add delta information from last step if available
			if last_step_delta is not None:
				state_payload["delta_from_last_step"] = last_step_delta
			else:
				state_payload["delta_from_last_step"] = {
					"new_conditions": [],
					"removed_conditions": [],
					"new_derivations": [],
				}
			prereq_payload: List[Dict[str, Any]] = []
			if pending_prereqs:
				for info in pending_prereqs.values():
					entry = {
						"requirement": info.get("requirement"),
						"suggestions": info.get("suggestions", []),
						"source_calls": sorted(info.get("source_calls", [])),
					}
					prereq_payload.append(entry)
			prereq_hints = build_prereq_hints(prereq_payload)
			combined_forward_hints = list(forward_hints)
			for hint in prereq_hints:
				if hint not in combined_forward_hints:
					combined_forward_hints.append(hint)
			if combined_forward_hints:
				state_payload["forward_hints"] = combined_forward_hints
			if forward_hint_calls:
				state_payload["forward_hint_calls"] = list(forward_hint_calls)
			
			# REMOVED: system_guidance construction
			# Recovery context is now embedded in payload["mode"]["recovery_context"]
			# No need for separate system message
			
			state_snapshot = build_state_log_snapshot(state_payload, combined_forward_hints)
			state_before_snapshot = copy.deepcopy(state_snapshot)
			
			# Simplified logging: show key info only
			if logger:
				logger.info(
					"[Step %s] Forward hints: %s | Candidates: %s | Mode: %s",
					step_idx,
					len(combined_forward_hints),
					len(candidate_list),
					attempt_mode,
				)
			history_excerpt = history_records[-5:]
			feedback_excerpt = recent_feedback[-3:]
			
			# Simplify history for LLM to reduce token usage
			# Only keep essential fields: step, call, success status, and state changes
			simplified_history = [
				{
					"step": entry.get("step"),
					"call": entry.get("call"),
					"updated": entry.get("updated"),
					"goal_solved": entry.get("goal_solved"),
					"state_delta": entry.get("state_delta", {}),
					"warnings": entry.get("warnings", [])[:2],  # Only first 2 warnings
				}
				for entry in history_excerpt
			]
			
			prompt_details = {
				"attempt_mode": attempt_mode,
				"forward_hint_enabled": True,
				"forward_hints": list(combined_forward_hints),
				"forward_hint_calls": list(forward_hint_calls),
				"forward_hint_new_heads": list(forward_hint_new_heads),
				"pending_prereqs": copy.deepcopy(prereq_payload),
				"recovery_context": copy.deepcopy(recovery_payload) if recovery_payload else None,
				# Convert (theorem_name, branch_idx) tuples to readable strings for logging
				"candidate_snapshot": [f"{name}[branch={idx}]" for name, idx in candidate_list[:30]],
				"recent_feedback": list(feedback_excerpt),
				"history_excerpt": copy.deepcopy(history_excerpt),
				"state_snapshot": state_before_snapshot,
			}
			messages = build_one_step_messages(
				problem,
				candidate_list,
				gdl_map,
				image_data_url,
				state_payload,
				predicted_calls,
				simplified_history,  # Use simplified version for LLM
				feedback_excerpt,
				step_idx,
				args.one_step_max_steps,
				combined_forward_hints,
				forward_hint_calls=forward_hint_calls,
				pending_prereqs=prereq_payload,
				recovery_payload=recovery_payload,
				attempt_mode=attempt_mode,
				readiness_scores=readiness_scores,
				valid_branches=valid_branches,
				branch_details=branch_details,
				branch_tpg_info=branch_tpg_info,
				branch_planning=branch_planning,
				branch_chains=branch_chains,
			)
			llm_requests.append(messages)
			
			llm_start = time.time()
			if quick_start_enabled:
				raw_llm = build_quick_start_llm_response(
					problem_id,
					step_idx,
					quick_start_replay_map,
					logger,
				)
			else:
				raw_llm = call_llm(
					llm_client,
					LLM_MODEL,
					messages,
					temperature=LLM_TEMPERATURE,
					max_retries=LLM_MAX_RETRIES,
					logger=logger,
				)
			detailed_timings["llm_inference"] += time.time() - llm_start
			llm_responses.append(raw_llm)
			
			if logger:
				logger.info("[Step %s] LLM: %s", step_idx, raw_llm[:80])
			calls = extract_calls(raw_llm)
			normalized_calls = [normalize_theorem_call(call) for call in calls]
			calls = [call for call in normalized_calls if call]
			
			if logger and calls:
				logger.info("[Step %s] Predicted: %s", step_idx, calls[0])
			
			if not calls:
				feedback = f"Step {step_idx}: The model did not output an executable theorem call."
				append_feedback(recent_feedback, feedback)
				
				if recovery_attempts < MAX_RECOVERY_ATTEMPTS:
					# Record history for this empty prediction
					history_entry = {
						"step": step_idx,
						"call": None,
						"updated": False,
						"goal_solved": bool(getattr(solver.problem.goal, "solved", False)),
						"warnings": ["no_prediction_recovery"],
						"feedback": feedback,
					}
					history_entry["prompt_details"] = copy.deepcopy(prompt_details)
					history_entry["llm_raw_response"] = raw_llm
					history_entry["parsed_calls"] = list(calls)
					history_entry["forward_hint_new_heads"] = list(forward_hint_new_heads)
					history_entry["state_before"] = copy.deepcopy(state_before_snapshot)
					history_entry["state_after"] = copy.deepcopy(state_before_snapshot)
					history_entry["state_delta"] = {"readable_added": [], "readable_removed": [], "recent_derivations_added": [], "recent_derivations_removed": []}
					history_entry["solver_new_logs"] = []
					history_records.append(history_entry)
					if logger:
						logger.warning("[OneStep] problem=%s step=%s returned no calls, attempting recovery", problem_id, step_idx)
					recovery_payload = {
						"failed_call": None,
						"reason": "no prediction",
						"feedback": feedback,
					}
					attempt_mode = "recovery"
					recovery_attempts += 1
					continue
				
				# Recovery exhausted; terminate this problem.
				termination_reason = "no_prediction"
				history_entry = {
					"step": step_idx,
					"call": None,
					"updated": False,
					"goal_solved": bool(getattr(solver.problem.goal, "solved", False)),
					"warnings": ["no_prediction_exhausted"],
					"feedback": feedback,
				}
				history_entry["prompt_details"] = copy.deepcopy(prompt_details)
				history_entry["llm_raw_response"] = raw_llm
				history_entry["parsed_calls"] = list(calls)
				history_entry["forward_hint_new_heads"] = list(forward_hint_new_heads)
				history_entry["state_before"] = copy.deepcopy(state_before_snapshot)
				history_entry["state_after"] = copy.deepcopy(state_before_snapshot)
				history_entry["state_delta"] = {"readable_added": [], "readable_removed": [], "recent_derivations_added": [], "recent_derivations_removed": []}
				history_entry["solver_new_logs"] = []
				history_records.append(history_entry)
				if logger:
					logger.warning("[OneStep] problem=%s step=%s no prediction after recovery exhausted", problem_id, step_idx)
				abort_loop = True
				break
			
			if not calls:
				# Safety check - should not reach here but just in case
				abort_loop = True
				break
			
			# Execute the top-ranked theorem.
			top_k_limit = 1
			top_k_all_failed = True
			successful_rank = 0
			top_k_attempted_calls: List[str] = []
			
			for rank_idx in range(top_k_limit):
				raw_call = calls[rank_idx]
				display_call = canonicalize_theorem_call(solver.problem, raw_call)
				call_key = display_call or raw_call
				top_k_attempted_calls.append(display_call)
				
				if logger and rank_idx > 0:
					logger.info(
						"[OneStep][TopK] problem=%s step=%s trying_rank=%s/%s call=%s (prev_failed)",
						problem_id,
						step_idx,
						rank_idx + 1,
						top_k_limit,
						display_call,
					)

				# --- QUICK RECOVERY DUPLICATE SUPPRESSION ---
				# If we're in recovery mode and the LLM suggests the same call that just failed
				# (or a call already recorded as a previous failure), skip this candidate
				if attempt_mode == "recovery" and recovery_payload:
					prev_failed = recovery_payload.get("failed_call") or recovery_payload.get("raw_call")
					# Normalize comparison using call_key
					if prev_failed and (call_key == prev_failed or call_key in failed_call_reasons):
						if logger:
							logger.warning(
								"[OneStep][TopK][Skip] problem=%s step=%s rank=%s skipped_redundant=%s",
								problem_id,
								step_idx,
								rank_idx + 1,
								display_call,
							)
						if call_key not in failed_call_reasons:
							failed_call_reasons[call_key] = {
								"reason": "redundant in recovery",
								"suggestions": [],
								"requirements": [],
							}
						continue  # Skip to next candidate in top-k
			
				# Check if this call was in failed_call_reasons (previous failure with unmet prereqs)
				failure_record = failed_call_reasons.get(call_key)
				if failure_record:
					requirements = failure_record.get("requirements") or []
					remaining_requirements = [req for req in requirements if not is_requirement_satisfied(solver.problem, req)]
					if requirements and not remaining_requirements:
						# Prerequisites now met, remove from blacklist and try
						failed_call_reasons.pop(call_key, None)
					else:
						# Still has unmet prerequisites, skip this candidate
						reason_text = failure_record.get("reason", "previous failure")
						if logger:
							logger.warning(
								"[OneStep][TopK][Skip] problem=%s step=%s rank=%s skipped_previous_failure=%s reason=%s",
								problem_id,
								step_idx,
								rank_idx + 1,
								display_call,
								reason_text,
							)
						continue  # Skip to next candidate in top-k
				
				# Check if this call was already successfully applied (prevent redundant re-application)
				if call_key in successful_calls:
					if logger:
						logger.warning(
							"[OneStep][TopK][Skip] problem=%s step=%s rank=%s skipped_already_successful=%s",
							problem_id,
							step_idx,
							rank_idx + 1,
							display_call,
						)
					failed_call_reasons[call_key] = {
						"reason": "redundant (already successfully applied)",
						"suggestions": [],
						"requirements": [],
					}
					continue  # Skip to next candidate in top-k
				
				# Check repeat limit (same call proposed too many times)
				attempt_index = call_attempts[call_key] + 1
				call_attempts[call_key] = attempt_index
				if attempt_index > 3:
					if logger:
						logger.warning(
							"[OneStep][TopK][Skip] problem=%s step=%s rank=%s skipped_repeat_limit=%s attempts=%s",
							problem_id,
							step_idx,
							rank_idx + 1,
							display_call,
							attempt_index,
						)
					failed_call_reasons[call_key] = {
						"reason": "repeat limit exceeded (>3 attempts)",
						"suggestions": [],
						"requirements": [],
					}
					continue  # Skip to next candidate in top-k
				
				# Try to parse this candidate
				repeat_note: Optional[str] = None
				if attempt_index > 1:
					repeat_note = (
						f"Step {step_idx}: Theorem {display_call} repeated attempt {attempt_index} (maximum 3 allowed)."
					)
					if logger:
						logger.info(
							"[OneStep] problem=%s step=%s call=%s repeated_attempt=%s",
							problem_id,
							step_idx,
							display_call,
							attempt_index,
						)
				
				parsed = parse_theorem_seqs([raw_call])
				if not parsed:
					# Parse failure - record failure and skip to next candidate
					if logger:
						logger.warning(
							"[OneStep][TopK][Skip] problem=%s step=%s rank=%s parse_failed=%s",
							problem_id,
							step_idx,
							rank_idx + 1,
							display_call,
						)
					failed_call_reasons[call_key] = {
						"reason": "parse failure (invalid theorem format)",
						"suggestions": [],
						"requirements": [],
					}
					continue  # Skip to next candidate in top-k
				
				t_name, t_branch, t_para = parsed[0]
				needs_auto_search = (t_branch is None)
				
				if needs_auto_search and logger:
					logger.warning(
						"[OneStep][AutoSearch] problem=%s step=%s rank=%s call=%s - LLM did not specify branch, will use solver auto-search (try all branches)",
						problem_id,
						step_idx,
						rank_idx + 1,
						display_call,
					)
				
				# Execute the candidate theorem.
				predicted_calls.append(display_call)
			call_warnings: List[str] = []
			missing_prereq_messages: List[str] = []
			call_updated = False
			human_label: Optional[str] = display_call
			prev_log_len = len(step_logs)
			# Record current solver step index so we can collect instances applied by the solver
			prev_step = solver.problem.condition.step_count
			solver_error_message = ""
			for t_name, t_branch, t_para in parsed:
				theorem_def = solver.parsed_theorem_GDL.get(t_name)
				if theorem_def is None:
					msg = f"Theorem {t_name} is not defined in the current session."
					call_warnings.append(msg)
					solver_error_message = msg
					log_entry = {
						"call": display_call,
						"name": t_name,
						"branch": t_branch,
						"duration": 0.0,
						"goal_solved": bool(getattr(solver.problem.goal, "solved", False)),
						"update": False,
						"warnings": [],
						"error": msg,
					}
					step_logs.append(log_entry)
					failed_call_reasons[call_key] = {
						"reason": msg,
						"suggestions": [],
						"requirements": [],
					}
					continue
				vars_seq = theorem_def.get("vars", ())
				# In only-predict-theorem mode we ignore any LLM-provided parameters and
				# let the solver (GPL) enumerate and apply all valid bindings. If the LLM
				# nevertheless included parameters, note it in warnings and discard them.
				if t_para:
					call_warnings.append("LLM provided object parameters which are ignored in only-predict-theorem mode.")
				t_para = None
				# Branch-level lightweight precheck
				precheck_warnings = precheck_theorem_branch(solver.problem, t_name, t_branch)
				if precheck_warnings:
					call_warnings.extend(precheck_warnings)
					missing_prereq_messages.extend(precheck_warnings)
					if logger:
						for warn_msg in precheck_warnings:
							logger.warning(
								"[OneStep][Precheck] problem=%s step=%s warning=%s",
								problem_id,
								step_idx,
								warn_msg,
							)
				# Theorem-level timeout control.
				theorem_identifier = f"{t_name}({t_branch})" if t_branch else t_name
				
				if theorem_identifier in THEOREM_TIMEOUT_BLACKLIST:
					msg = f"Theorem {theorem_identifier} is in timeout blacklist, skipping to avoid repeated timeout."
					call_warnings.append(msg)
					if logger:
						logger.warning(
							"[TheoremTimeout][Blacklist] problem=%s step=%s theorem=%s",
							problem_id,
							step_idx,
							theorem_identifier,
						)
					log_entry = {
						"call": display_call,
						"name": t_name,
						"branch": t_branch,
						"duration": 0.0,
						"goal_solved": bool(getattr(solver.problem.goal, "solved", False)),
						"update": False,
						"warnings": [msg],
						"skipped_reason": "timeout_blacklist",
					}
					step_logs.append(log_entry)
					continue
				
				with warnings.catch_warnings(record=True) as caught:
					warnings.simplefilter("always")
					start = time.time()
					update = False
					theorem_execution_timeout = False
					
					elapsed_total = time.time() - solver_start_time
					remaining_time = time_limit - elapsed_total if time_limit > 0 else float('inf')
					
					effective_timeout = min(MAX_THEOREM_EXECUTION_TIME, remaining_time * 0.5) if remaining_time < float('inf') else MAX_THEOREM_EXECUTION_TIME
					
					if effective_timeout < 10:
						if logger:
							logger.warning(
								"[TheoremTimeout][Skip] problem=%s step=%s theorem=%s remaining_time=%.1fs too_low_skip",
								problem_id,
								step_idx,
								theorem_identifier,
								remaining_time,
							)
						log_entry = {
							"call": display_call,
							"name": t_name,
							"branch": t_branch,
							"duration": 0.0,
							"goal_solved": bool(getattr(solver.problem.goal, "solved", False)),
							"update": False,
							"warnings": [f"Skipped due to insufficient remaining time: {remaining_time:.1f}s"],
							"skipped_reason": "low_remaining_time",
						}
						step_logs.append(log_entry)
						continue
					
					try:
						# Use GPL auto-search to find and apply ALL valid instances
						update = solver.apply_theorem(t_name, t_branch, None)
						
						check_goal_timeout = min(120.0, effective_timeout * 0.5)
						try:
							func_timeout(check_goal_timeout, solver.problem.check_goal)
						except FunctionTimedOut:
							if logger:
								logger.warning(
									"[CheckGoal][Timeout] problem=%s step=%s check_goal timeout after %.1fs",
									problem_id,
									step_idx,
									check_goal_timeout
								)
						# check_goal timeout does not affect theorem execution result.
						execution_time = time.time() - start
						if execution_time > MAX_THEOREM_EXECUTION_TIME:
							theorem_execution_timeout = True
							msg = (
								f"Theorem {theorem_identifier} execution took {execution_time:.1f}s "
								f"(limit: {MAX_THEOREM_EXECUTION_TIME}s). "
								f"Added to blacklist to prevent future slow executions."
							)
							call_warnings.append(msg)

							
							if logger:
								logger.warning(
									"[TheoremTimeout][Slow] problem=%s step=%s theorem=%s elapsed=%.1fs blacklist_size=%d",
									problem_id,
									step_idx,
									theorem_identifier,
									execution_time,
									len(THEOREM_TIMEOUT_BLACKLIST),
								)
					
					except Exception as exc:
						solver_error_message = str(exc)
					
					timing = round(time.time() - start, 6)
					detailed_timings["symbolic_solver"] += timing
				warning_messages = [str(w.message) for w in caught]
				fv_warning_reason: Optional[str] = None
				for warn_msg in warning_messages:
					if any(marker in warn_msg for marker in FV_WARNING_MARKERS):
						fv_warning_reason = warn_msg if fv_warning_reason is None else fv_warning_reason
				call_warnings.extend(warning_messages)
				log_entry = {
					"call": display_call,
					"name": t_name,
					"branch": t_branch,
					"duration": timing,
					"goal_solved": bool(getattr(solver.problem.goal, "solved", False)),
					"update": bool(update),
					"warnings": warning_messages,
				}
				if theorem_execution_timeout:
					log_entry["theorem_timeout"] = True
					log_entry["timeout_limit"] = MAX_THEOREM_EXECUTION_TIME
				if fv_warning_reason:
					log_entry["fv_warning_reason"] = fv_warning_reason
				log_entry["precheck_warnings"] = precheck_warnings
				if solver_error_message:
					log_entry["error"] = solver_error_message
				step_logs.append(log_entry)
				call_updated = call_updated or bool(update)
				# FV warnings should only invalidate the call if the theorem didn't actually update
				# In --only-predict-theorem mode, GPL auto-search produces many FV warnings for
				# invalid parameter combinations, but if update=True, the theorem DID succeed
				if fv_warning_reason and not update:
					call_warnings.append("Forward verification failed; treat this theorem as invalid.")
					missing_prereq_messages.append(fv_warning_reason)
					call_updated = False
				if human_label is None:
					human_label = inverse_parse_one_theorem((t_name, t_branch, t_para), solver.problem.parsed_theorem_GDL)
				if logger:
					logger.info(
						"[Step %s] Executed: %s | Updated: %s | Goal solved: %s",
						step_idx,
						human_label or display_call,
						bool(update),
						log_entry["goal_solved"],
					)
				if solver_error_message and not theorem_execution_timeout:
					break
				symbol_map = build_symbol_map(solver.problem)
			new_logs = step_logs[prev_log_len:]
			# Collect actual applied instances (parameterized theorems) by solver
			applied_instances = collect_applied_instances_since(solver.problem, prev_step)
			if applied_instances:
				# Add a concise feedback summary for the LLM / human
				fb_lines = build_feedback_from_instances(applied_instances, top_k=5)
				for line in fb_lines:
					append_feedback(recent_feedback, line)
			goal_status = any(entry["goal_solved"] for entry in new_logs) or bool(getattr(solver.problem.goal, "solved", False))
			if human_label is None:
				human_label = display_call
			post_state_payload = gather_problem_state(solver.problem)
			state_after_snapshot = build_state_log_snapshot(post_state_payload, combined_forward_hints)
			
			readable_before = set((state_before_snapshot.get("readable_summary") or "").splitlines())
			readable_after = set((state_after_snapshot.get("readable_summary") or "").splitlines())
			readable_added = sorted(readable_after - readable_before)
			readable_removed = sorted(readable_before - readable_after)
			recent_before = set(state_before_snapshot.get("recent_derivations") or [])
			recent_after = set(state_after_snapshot.get("recent_derivations") or [])
			
			state_delta = {
				"readable_added": readable_added,
				"readable_removed": readable_removed,
				"recent_derivations_added": sorted(recent_after - recent_before),
				"recent_derivations_removed": sorted(recent_before - recent_after),
				"goal_progress_after": post_state_payload.get("goal_progress"),
			}
			
			# Prepare delta for next step
			last_step_delta = {
				"new_conditions": readable_added,
				"removed_conditions": readable_removed,
				"new_derivations": sorted(recent_after - recent_before),
			}
			previous_state_snapshot = state_after_snapshot
			feedback_parts: List[str] = []
			
			# CRITICAL FIX: Only generate feedback for failures or important state changes
			# Success without issues should not add noise to the feedback
			if call_updated:
				# Success case: Only add feedback if goal is solved (important milestone)
				if goal_status:
					feedback_parts.append(f"Step {step_idx}: The goal is solved after applying {display_call}.")
				# Otherwise, no feedback needed - success speaks for itself
			else:
				# Failure case: Provide detailed feedback to guide recovery
				feedback_parts.append(
					f"Step {step_idx}: Theorem {display_call} did not update the state."
					" Verify the argument format and ensure all prerequisites hold."
				)
			
			if repeat_note:
				feedback_parts.append(repeat_note)
			if solver_error_message:
				call_warnings.append(f"Solver error: {solver_error_message}")
				theorem_head = extract_theorem_name(raw_call)
				if theorem_head and theorem_head in gdl_map:
					guidance = compute_signature_details(theorem_head, gdl_map[theorem_head])
					valid_example = guidance.get("valid_call_example")
					if valid_example:
						call_warnings.append(f"Valid call example: {valid_example}.")
					argument_roles = guidance.get("argument_roles") or []
					if argument_roles:
						call_warnings.append("Argument roles (ordered): " + ", ".join(argument_roles) + ".")
					if guidance.get("has_forms"):
						call_warnings.append("Select exactly one form index and place it as the first argument.")
				elif theorem_head and theorem_head not in candidate_theorem_names:
					# Show a preview of allowed theorem names
					preview_names = sorted(candidate_theorem_names)[:5]
					preview = ", ".join(preview_names)
					if preview:
						call_warnings.append(f"Allowed theorem names include: {preview}.")
				call_warnings.append("Never invent new theorem names or change argument order.")
			warnings_for_feedback = call_warnings
			if missing_prereq_messages:
				missing_details: List[str] = []
				for msg in missing_prereq_messages:
					missing_details.extend(parse_missing_prereq_items(msg))
				if missing_details:
					seen_reqs: List[str] = []
					for req in list(missing_details):
						if req not in seen_reqs:
							seen_reqs.append(req)
					missing_details = seen_reqs
				if missing_details:
					feedback_parts.append("Prerequisites still missing: " + "; ".join(missing_details))
					aggregated_suggestions: List[str] = []
					for requirement in missing_details:
						# Use candidate_theorem_names (Set[str]) instead of candidate_list (List[Tuple[str, int]])
						suggestions = suggest_theorems_for_requirement(requirement, sorted(candidate_theorem_names))
						entry = pending_prereqs.setdefault(
							requirement,
							{
								"requirement": requirement,
								"suggestions": list(suggestions),
								"source_calls": set(),
							},
						)
						if suggestions:
							existing = entry.get("suggestions", [])
							for suggestion in suggestions:
								if suggestion not in existing:
									existing.append(suggestion)
							entry["suggestions"] = existing
						entry.setdefault("source_calls", set()).add(display_call)
						for suggestion in suggestions:
							if suggestion not in aggregated_suggestions:
								aggregated_suggestions.append(suggestion)
					if len(aggregated_suggestions) > 5:
						aggregated_suggestions = aggregated_suggestions[:5]
					failed_call_reasons[call_key] = {
						"reason": "unmet prerequisites",
						"suggestions": aggregated_suggestions,
						"requirements": missing_details,
					}
				target_set = set(missing_prereq_messages)
				warnings_for_feedback = [w for w in call_warnings if w not in target_set]
			feedback_parts.extend(warnings_for_feedback)
			
			# Note: goal_status feedback is already added above in the success case
			# No need to duplicate it here
			
			feedback = " ".join(feedback_parts).strip()
			if feedback:
				append_feedback(recent_feedback, feedback)
			history_entry = {
				"step": step_idx,
				"call": human_label,
				"updated": call_updated,
				"goal_solved": goal_status,
				"warnings": call_warnings.copy(),
				"feedback": feedback,
			}
			if not top_k_all_failed and successful_rank > 0:
				history_entry["top_k_successful_rank"] = successful_rank
				history_entry["top_k_total_candidates"] = top_k_limit
			if top_k_attempted_calls:
				history_entry["top_k_attempted_calls"] = list(top_k_attempted_calls)
			if applied_instances:
				history_entry["applied_instances"] = copy.deepcopy(applied_instances)
			if raw_call != human_label:
				history_entry["raw_call"] = raw_call
			history_entry["prompt_details"] = copy.deepcopy(prompt_details)
			history_entry["llm_raw_response"] = raw_llm
			history_entry["parsed_calls"] = list(calls)
			history_entry["forward_hint_new_heads"] = list(forward_hint_new_heads)
			history_entry["state_before"] = copy.deepcopy(state_before_snapshot)
			history_entry["state_after"] = copy.deepcopy(state_after_snapshot)
			history_entry["state_delta"] = state_delta
			history_entry["solver_new_logs"] = copy.deepcopy(new_logs)
			if new_logs:
				aggregate_duration = round(sum(log.get("duration", 0.0) or 0.0 for log in new_logs), 6)
				updates = sum(1 for log in new_logs if log.get("update"))
				solver_notes: Dict[str, Any] = {
					"duration": aggregate_duration,
					"updates": updates,
				}
				errors = [log.get("error") for log in new_logs if log.get("error")]
				if errors:
					solver_notes["errors"] = errors
				warnings_flat: List[str] = []
				for log in new_logs:
					warnings_flat.extend(log.get("warnings") or [])
					warnings_flat.extend(log.get("precheck_warnings") or [])
				if warnings_flat:
					solver_notes["warnings"] = warnings_flat
				history_entry["solver_summary"] = solver_notes
			history_records.append(history_entry)
			
			_worker_progress_tracker["steps_completed"] = step_idx
			_worker_progress_tracker["predicted_calls"] = list(predicted_calls)
			_worker_progress_tracker["history_records"] = copy.deepcopy(history_records)
			_worker_progress_tracker["timing_stats"]["solver_execution"] = time.time() - solver_start_time
			_worker_progress_tracker["goal_solved"] = bool(getattr(solver.problem.goal, "solved", False))
			_worker_progress_tracker["last_step_info"] = {
				"step": step_idx,
				"call": human_label,
				"updated": call_updated,
			}
			
			# CRITICAL FIX: Check for success FIRST before checking for issues
			# If the call updated the state or solved the goal, this step is successful
			# regardless of warnings or precheck messages
			if bool(getattr(solver.problem.goal, "solved", False)):
				termination_reason = "goal_solved"
				abort_loop = True
				break
			
			if call_updated or goal_status:
				theorem_name_applied = extract_theorem_name(call_key)
				
				is_reusable = theorem_name_applied in REUSABLE_THEOREMS_WHITELIST if theorem_name_applied else False
				
				if not is_reusable:
					successful_calls.add(call_key)
					
					if logger:
						logger.info(
							"[OneStep][Blacklist] problem=%s step=%s added_to_blacklist=%s (size=%s)",
							problem_id,
							step_idx,
							call_key,
							len(successful_calls),
						)
				else:
					log_whitelist_exemption(problem_id, step_idx, theorem_name_applied, logger)
				
				if theorem_name_applied:
					applied_theorem_names.add(theorem_name_applied)
					last_successful_theorem = theorem_name_applied
				
				attempt_mode = "normal"
				recovery_payload = None
				consecutive_no_update_steps = 0
				top_k_all_failed = False
				successful_rank = rank_idx + 1
				if logger:
					logger.info(
						"[OneStep][TopK][Success] problem=%s step=%s successful_rank=%s/%s call=%s",
						problem_id,
						step_idx,
						successful_rank,
						top_k_limit,
						display_call,
					)
				break  # ← SUCCESS: Exit top-k loop and proceed to next step
			else:
				issue_reason: Optional[str] = None
				if solver_error_message:
					issue_reason = "solver_error"
				elif not call_updated and not goal_status:
					issue_reason = "no_update"
				elif call_warnings:
					issue_reason = "warnings"
				elif missing_prereq_messages:
					issue_reason = "unmet_prerequisites"
				
				if issue_reason:
					recorded_requirements: List[str] = []
					if issue_reason == "unmet_prerequisites":
						recorded_requirements = list(missing_prereq_messages)
					
					if call_key not in failed_call_reasons:
						failed_call_reasons[call_key] = {
							"reason": issue_reason.replace("_", " "),
							"suggestions": [],
							"requirements": recorded_requirements,
						}
					
					if logger:
						logger.warning(
							"[OneStep][TopK][Failed] problem=%s step=%s rank=%s/%s call=%s reason=%s",
							problem_id,
							step_idx,
							rank_idx + 1,
							top_k_limit,
							display_call,
							issue_reason,
						)
					
					if rank_idx + 1 < top_k_limit:
						continue  # Try next candidate in top-k
			
			# End of top-k for loop
			
			if top_k_all_failed:
				if logger:
					logger.warning(
						"[OneStep][TopK][AllFailed] problem=%s step=%s all %s candidates failed: %s",
						problem_id,
						step_idx,
						top_k_limit,
						top_k_attempted_calls,
					)
				
				#recovery mode
				if recovery_attempts < MAX_RECOVERY_ATTEMPTS:
					failed_details = []
					for call in top_k_attempted_calls:
						reason = failed_call_reasons.get(call, {}).get("reason", "unknown")
						failed_details.append(f"{call} ({reason})")
					
					recovery_payload = {
						"failed_call": top_k_attempted_calls[0] if top_k_attempted_calls else None,
						"all_failed_calls": top_k_attempted_calls,
						"reason": f"all {top_k_limit} candidates failed",
						"feedback": f"All {top_k_limit} candidates failed: {'; '.join(failed_details)}. Try different theorems.",
					}
					attempt_mode = "recovery"
					recovery_attempts += 1
					continue  # ← RECOVERY: Ask LLM for new suggestions
				
				# ❌ Recovery attempts exhausted
				consecutive_no_update_steps += 1
				current_elapsed = time.time() - solver_start_time
				dynamic_max = get_dynamic_max_no_update(step_idx, current_elapsed, time_limit)
				
				if logger:
					logger.warning(
						"[CONSECUTIVE NO_UPDATE] problem=%s step=%s count=%s/%s (dynamic_max=%s)",
						problem_id, step_idx, consecutive_no_update_steps, dynamic_max, dynamic_max
					)
				
				if consecutive_no_update_steps >= dynamic_max:
					if logger:
						logger.warning(
							"[TERMINATION] problem=%s: %s consecutive no_update steps (dynamic_max=%s) - aborting",
							problem_id, consecutive_no_update_steps, dynamic_max
						)
					termination_reason = "consecutive_no_update"
					abort_loop = True
					break
				else:
					if logger:
						logger.info(
							"[CONTINUE] problem=%s: no_update but only %s/%s consecutive (dynamic) - trying next step",
							problem_id, consecutive_no_update_steps, dynamic_max
						)
					break
			
		# End of while True (recovery loop)
		if abort_loop:
			break
	# End of for step_idx in range(max_steps)
	return solver, predicted_calls, step_logs, llm_requests, llm_responses, termination_reason, history_records, detailed_timings


def save_result(path: Path, data: Dict[str, Any]) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)


def summarize_problem_outcome(result: Dict[str, Any]) -> Dict[str, Any]:
	"""Create a lightweight per-problem summary for the run summary file."""
	call_count = len(result.get("predicted_calls") or [])
	status = result.get("status")
	if not status:
		status = "solved" if result.get("goal_solved") else "failed"
	entry: Dict[str, Any] = {
		"problem_id": result.get("problem_id"),
		"status": status,
		"goal_solved": bool(result.get("goal_solved")),
		"call_count": call_count,
	}
	
	# Add timing information
	timing_stats = result.get("timing_stats", {})
	if timing_stats and not result.get("quick_start", False):
		entry["total_time"] = round(timing_stats.get("total_time", 0), 2)
	
	if "one_step_termination" in result:
		entry["termination"] = result["one_step_termination"]
	if status == "error" and result.get("error"):
		entry["error"] = result.get("error")
	
	# Add error message for timeout
	if status == "timeout":
		entry["error"] = result.get("error", f"Timeout after {entry.get('total_time', 'unknown')}s")
	
	# Add step-by-step execution details for one-step mode
	if result.get("one_step_history"):
		steps = []
		for hist in result["one_step_history"]:
			step_num = hist.get("step")
			call = ensure_theorem_call_has_branch(str(hist.get("call") or ""))
			updated = hist.get("updated", False)
			goal_solved = hist.get("goal_solved", False)
			if call:
				step_info = f"[Step {step_num}] Executed: {call} | Updated: {updated} | Goal solved: {goal_solved}"
				steps.append(step_info)
		entry["execution_steps"] = steps

	# DO NOT preserve full one_step_history - it causes massive file sizes (GB)
	# execution_steps already contains the essential information
	
	return entry


def sanitize_problem_result(result: Dict[str, Any]) -> Dict[str, Any]:
	"""Produce a readable record for the detailed output file."""
	predicted_calls = list(result.get("predicted_calls") or [])
	ground_truth_calls = list(result.get("theorem_seqs") or [])
	sanitized: Dict[str, Any] = {
		"problem_id": result.get("problem_id"),
		"status": result.get("status"),
		"run_id": result.get("run_id"),
		"goal_solved": result.get("goal_solved"),
		"predicted_call_count": len(predicted_calls),
		"predicted_calls": predicted_calls,
	}
	if ground_truth_calls:
		sanitized["ground_truth_calls"] = ground_truth_calls
	candidate_theorems = list(result.get("candidate_theorems") or [])
	if candidate_theorems:
		sanitized["candidate_theorems"] = candidate_theorems
	sanitized["problem_answer"] = result.get("problem_answer")
	sanitized["goal_cdl"] = result.get("goal_cdl")
	if "one_step_termination" in result:
		sanitized["one_step_termination"] = result["one_step_termination"]
	# DO NOT include one_step_history - it causes massive file sizes
	if result.get("error"):
		sanitized["error"] = result["error"]
	cleaned: Dict[str, Any] = {}
	for key, value in sanitized.items():
		if value is None:
			continue
		if isinstance(value, (list, dict)) and not value:
			continue
		cleaned[key] = value
	return cleaned


def build_state_log_snapshot(state_payload: Dict[str, Any], forward_hints: Sequence[str]) -> Dict[str, Any]:
	"""Condense solver state for readable logging output."""
	return {
		"readable_summary": state_payload.get("readable_summary"),
		"recent_derivations": state_payload.get("recent_derivations"),
		"unsolved_equations": state_payload.get("unsolved_equations"),
		"goal_progress": state_payload.get("goal_progress"),
		"forward_hints": list(forward_hints) if forward_hints else [],
	}


# =============================================================================
# SECTION 9: MAIN PIPELINE & ORCHESTRATION
# =============================================================================

def process_problem(
	problem_id: int,
	args: argparse.Namespace,
	dataset_loader: DatasetLoader,
	embedder: Optional[MultiModalEmbedding],
	entries: Optional[Dict[str, Dict[str, Any]]],
	problems_dir: Path,
	diagrams_dir: Path,
	gdl_map: Dict[str, Dict[str, Any]],
	llm_client: Any,
	logger: logging.Logger,
) -> Dict[str, Any]:
	"""Process a single problem with detailed timing statistics."""
	timing_stats = {}
	problem_start = time.time()
	
	# Step 1: Load problem
	step_start = time.time()
	problem = dataset_loader.get_problem(problem_id)
	timing_stats["load_problem"] = time.time() - step_start

	# Quick-start path: no embedding/RAG/LLM, replay theorem calls directly.
	if getattr(args, "quick_start", False):
		replay_map = getattr(args, "_quick_start_replay_map", {}) or {}
		replay_calls = replay_map.get(problem_id, [])
		if logger:
			logger.info("[QuickStart] problem=%s replay_calls=%s", problem_id, len(replay_calls))

		step_start = time.time()
		solver, predicted_calls, step_logs, termination_reason, history_records = run_quick_start_replay_solver(
			dataset_loader,
			problem,
			replay_calls,
			logger,
		)
		goal_solved = bool(getattr(solver.problem.goal, "solved", False))
		timing_stats["solver_execution"] = time.time() - step_start

		step_start = time.time()
		if args.show_solution:
			show_solution(solver.problem)
		result = {
			"problem_id": problem_id,
			"quick_start": True,
			"goal_cdl": problem.get("goal_cdl"),
			"problem_answer": problem.get("problem_answer"),
			"candidate_theorems": [],
			"retrieval_records": [],
			"query_payload": {
				"mode": "quick_start_replay",
				"replay_file": getattr(args, "quick_start_file", str(QUICK_START_DEFAULT_FILE)),
				"replay_call_count": len(replay_calls),
			},
			"predicted_calls": predicted_calls,
			"ground_truth_calls": problem.get("theorem_seqs", []) or [],
			"goal_solved": goal_solved,
			"llm_raw_response": [],
			"llm_request": [],
			"one_step_termination": termination_reason,
			"one_step_history": [
				{
					"step": hist.get("step"),
					"call": hist.get("call"),
					"updated": hist.get("updated", False),
					"goal_solved": hist.get("goal_solved", False),
				}
				for hist in history_records
			],
		}
		timing_stats["build_result"] = time.time() - step_start
		timing_stats["total_time"] = time.time() - problem_start
		result["timing_stats"] = timing_stats
		return result
	
	# Step 2: Build candidate theorems (includes RAG retrieval)
	step_start = time.time()
	candidate_theorems, retrieval_records, query_payload, merged_tpg, tpg_common_prefix, tpg_unlock_scores, tpg_edge_weights, expected_steps = build_candidate_theorems(
		problem_id,
		problems_dir=problems_dir,
		diagrams_dir=diagrams_dir,
		mode=args.retrieval_mode,
		top_k=args.top_k,
		entries=entries,
		embedder=embedder,
	)
	timing_stats["rag_retrieval"] = time.time() - step_start
	
	
	# Log TPG info and dynamic priorities
	if merged_tpg:
		logger.info(
			"[TPG] problem=%s start_theorems=%s dependency_edges=%s common_prefix_count=%s edge_weights=%s expected_steps=%s",
			problem_id,
			len(merged_tpg.get("START", [])),
			sum(len(v) for k, v in merged_tpg.items() if k != "START"),
			len(tpg_common_prefix),
			len(tpg_edge_weights),
			expected_steps,
		)
	
	ground_truth_calls = problem.get("theorem_seqs", []) or []

	# Step 3: Prepare image data
	step_start = time.time()
	image_path = Path(query_payload["image_path"]) if query_payload.get("image_path") else None
	image_data_url = local_image_to_data_url(image_path)
	timing_stats["image_preparation"] = time.time() - step_start
	
	# Step 4: Run one-step solver
	step_start = time.time()
	
	(
		solver,
		predicted_calls,
		step_logs,
		llm_requests,
		llm_responses,
		termination_reason,
		history_records,
		detailed_timings,
	) = run_one_step_solver(
		dataset_loader,
		problem,
		candidate_theorems,
		gdl_map,
		image_data_url,
		merged_tpg=merged_tpg,
		tpg_common_prefix=tpg_common_prefix,
		tpg_unlock_scores=tpg_unlock_scores,
		tpg_edge_weights=tpg_edge_weights, 
		expected_steps=expected_steps, 
		llm_client=llm_client,
		args=args,
		logger=logger,
		problem_id=problem_id,
	)
	goal_solved = bool(getattr(solver.problem.goal, "solved", False))
	timing_stats["solver_execution"] = time.time() - step_start
	timing_stats.update(detailed_timings)
	
	logger.info(
		"[OneStep] problem=%s goal_solved=%s termination=%s attempts=%s",
		problem_id,
		goal_solved,
		termination_reason,
		len(predicted_calls),
	)
	raw_llm_record = llm_responses
	llm_messages_record = llm_requests

	# Step 5: Build result dictionary
	step_start = time.time()
	if args.show_solution:
		show_solution(solver.problem)
	result = {
		"problem_id": problem_id,
		"goal_cdl": problem.get("goal_cdl"),
		"problem_answer": problem.get("problem_answer"),
		"candidate_theorems": candidate_theorems,
		"retrieval_records": retrieval_records,
		"query_payload": query_payload,
		"predicted_calls": predicted_calls,
		"ground_truth_calls": ground_truth_calls,
		"goal_solved": goal_solved,
		"llm_raw_response": raw_llm_record,
		"llm_request": llm_messages_record,
	}
	result["one_step_termination"] = termination_reason
	lightweight_history = []
	for hist in history_records:
		lightweight_history.append({
			"step": hist.get("step"),
			"call": hist.get("call"),
			"updated": hist.get("updated", False),
			"goal_solved": hist.get("goal_solved", False),
		})
	result["one_step_history"] = lightweight_history
	timing_stats["build_result"] = time.time() - step_start
	
	# Total timing
	timing_stats["total_time"] = time.time() - problem_start
	result["timing_stats"] = timing_stats
	
	# Log timing breakdown
	logger.info(
		"[Timing] problem=%s total=%.2fs (load=%.2fs, rag=%.2fs, image=%.2fs, solver=%.2fs, result=%.2fs)",
		problem_id,
		timing_stats["total_time"],
		timing_stats.get("load_problem", 0),
		timing_stats.get("rag_retrieval", 0),
		timing_stats.get("image_preparation", 0),
		timing_stats.get("solver_execution", 0),
		timing_stats.get("build_result", 0),
	)
	
	return result


# Global variables for worker processes (initialized once per worker)
_worker_dataset_loader = None
_worker_embedder = None
_worker_entries = None
_worker_gdl_map = None
_worker_llm_client = None
_worker_diagrams_dir = None
_worker_id = None  # Worker ID for dedicated API key assignment

# Progress tracking for timeout recovery - stores intermediate state during problem processing
# This allows us to recover partial results when a problem is forcibly terminated due to timeout
_worker_progress_tracker: Dict[str, Any] = {
	"problem_id": None,
	"steps_completed": 0,
	"predicted_calls": [],
	"history_records": [],
	"timing_stats": {},
	"last_step_info": None,
	"start_time": None,
}


def init_worker(dataset_loader_kwargs, retrieval_mode, embedding_model, problems_dir, worker_id_queue, show_solution, quick_start, quick_start_replay_file):
	"""
	Initialize worker process with shared data and dedicated API key.
	This runs once per worker process, not once per problem.
	Worker ID is assigned from a shared queue to ensure uniqueness across processes.
	
	Args:
		dataset_loader_kwargs: Dataset configuration
		retrieval_mode: Embedding retrieval mode
		problems_dir: Path to problems directory
		worker_id_queue: Shared queue containing worker IDs (0, 1, 2, 3, ...)
	"""
	global _worker_dataset_loader, _worker_embedder, _worker_entries
	global _worker_gdl_map, _worker_llm_client, _worker_diagrams_dir, _worker_id
	
	# Get unique worker ID from shared queue
	_worker_id = worker_id_queue.get()
	
	# Initialize logger for this worker
	worker_logger = logging.getLogger(f"worker_{os.getpid()}")
	worker_logger.setLevel(logging.INFO if show_solution else logging.CRITICAL)
	worker_logger.propagate = False
	for existing_handler in list(worker_logger.handlers):
		worker_logger.removeHandler(existing_handler)
	handler = logging.StreamHandler()
	handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S'))
	worker_logger.addHandler(handler)
	if not quick_start:
		worker_logger.info("[Worker] Initializing worker process PID=%s worker_id=%s", os.getpid(), _worker_id)
	
	start_time = time.time()
	
	# Load dataset
	_worker_dataset_loader = DatasetLoader(**dataset_loader_kwargs)
	
	# Load GDL
	gdl_path = Path(dataset_loader_kwargs["datasets_path"]) / dataset_loader_kwargs["dataset_name"] / "gdl" / "theorem_GDL.json"
	_worker_gdl_map = load_gdl_signatures(gdl_path)
	
	if quick_start:
		# Quick-start: 完全跳过 embedding/RAG 初始化，避免任何 API 依赖。
		_worker_entries = {}
		_worker_embedder = None
	else:
		# Load embedding store (train split for retrieval - same as test_rag_coverage.py)
		_worker_entries, store_metadata, store_path = prepare_embedding_store(retrieval_mode, "train")
		worker_logger.info("[Worker] Loaded embedding store: %s entries", len(_worker_entries))
		
		# Initialize embedder
		_worker_embedder = MultiModalEmbedding(
			model=embedding_model,
			api_key=EMBEDDING_API_KEY,
			api_base=EMBEDDING_API_URL,
			instruction=EMBEDDING_INSTRUCTION,
		)
	
	# Initialize LLM client (skip API key assignment and API client in quick_start mode)
	if quick_start:
		_worker_llm_client = None
		_worker_dataset_loader_args_quick_map = load_quick_start_replay_map(Path(quick_start_replay_file), worker_logger)
		# Store replay map on global args carrier via module variable (read in process_problem)
		globals()["_worker_quick_start_replay_map"] = _worker_dataset_loader_args_quick_map
	else:
		# Assign dedicated API key based on worker_id (with concurrent sharing support)
		key_index = _worker_id % len(LLM_API_KEYS)
		worker_api_key = LLM_API_KEYS[key_index]
		if _worker_id < len(LLM_API_KEYS):
			worker_logger.info("[Worker] Assigned API key index=%s/%s: %s... (primary worker for this key)", 
			                  key_index, len(LLM_API_KEYS), worker_api_key[:10])
		else:
			worker_logger.info("[Worker] Assigned API key index=%s/%s: %s... (concurrent worker, sharing key)", 
			                  key_index, len(LLM_API_KEYS), worker_api_key[:10])
		_worker_llm_client = ensure_openai_client(worker_api_key, LLM_API_BASE, LLM_MODEL)
		globals()["_worker_quick_start_replay_map"] = {}
	
	# Set diagrams directory
	_worker_diagrams_dir = problems_dir.parent / "diagrams"
	
	elapsed = time.time() - start_time
	if not quick_start:
		worker_logger.info("[Worker] Initialization complete in %.2f seconds", elapsed)


def _process_problem_inner(pid, args, problems_dir, logger):
	"""
	Inner function that actually processes the problem.
	This is wrapped by func_timeout for strict timeout enforcement.
	"""
	global _worker_dataset_loader, _worker_embedder, _worker_entries
	global _worker_gdl_map, _worker_llm_client, _worker_diagrams_dir
	
	# Attach quick-start replay map to args so run_one_step_solver can read it.
	setattr(args, "_quick_start_replay_map", globals().get("_worker_quick_start_replay_map", {}))

	return process_problem(
		pid,
		args,
		_worker_dataset_loader,
		_worker_embedder,
		_worker_entries,
		problems_dir,
		_worker_diagrams_dir,
		_worker_gdl_map,
		_worker_llm_client,
		logger,
	)


def process_problem_worker(args_tuple):
	"""
	Worker function for multiprocessing. Processes a single problem with STRICT timeout.
	Uses func_timeout to forcibly terminate if the problem takes too long.
	
	Args:
		args_tuple: Tuple containing (pid, args_dict, run_id, problems_dir)
	
	Returns:
		Dict with processing result
	"""
	task_start = time.time()
	pid, args_dict, run_id, problems_dir = args_tuple
	
	# Use pre-loaded global data
	global _worker_dataset_loader, _worker_embedder, _worker_entries
	global _worker_gdl_map, _worker_llm_client, _worker_diagrams_dir
	
	# Reconstruct args
	args = argparse.Namespace(**args_dict)
	
	# Get logger for this worker
	logger = logging.getLogger(f"worker_{os.getpid()}")
	
	# Get time limit from args (default 300 seconds)
	time_limit = getattr(args, 'time_limit', 300)
	
	try:
		# Use func_timeout for STRICT timeout enforcement
		# This will forcibly terminate the function if it exceeds time_limit
		result = func_timeout(
			time_limit,
			_process_problem_inner,
			args=(pid, args, problems_dir, logger)
		)
		
		result["run_id"] = run_id
		
		# Check if internal timeout was triggered (graceful)
		termination_reason = result.get("one_step_termination", "")
		if termination_reason == "timeout":
			task_total = time.time() - task_start
			result["status"] = "timeout"
			result["timeout_info"] = {
				"time_limit": time_limit,
				"elapsed_time": round(task_total, 2),
				"timeout_type": "internal",  # Graceful timeout with complete state
				"steps_completed": len(result.get("one_step_history", [])),
				"calls_attempted": len(result.get("predicted_calls", [])),
			}
			if not getattr(args, "quick_start", False):
				logger.warning(
					"[TIMEOUT] Problem %s completed with internal timeout: "
					"elapsed=%.2fs, limit=%ds, steps=%d, calls=%d",
					pid, task_total, time_limit,
					result["timeout_info"]["steps_completed"],
					result["timeout_info"]["calls_attempted"],
				)
		elif not result.get("status"):
			result_status = "solved" if result.get("goal_solved") else "failed"
			result["status"] = result_status
		
		# Calculate worker overhead
		task_total = time.time() - task_start
		process_time = result.get("timing_stats", {}).get("total_time", 0)
		worker_overhead = task_total - process_time
		
		if not getattr(args, "quick_start", False):
			logger.info(
				"[Pipeline] Problem %s status=%s calls=%s task_time=%.2fs (process=%.2fs, overhead=%.2fs)", 
				pid, 
				result.get("status"),
				len(result.get("predicted_calls", [])),
				task_total,
				process_time,
				worker_overhead
			)
		
		return result
		
	except FunctionTimedOut:
		# STRICT timeout triggered - func_timeout forcibly terminated the function
		# Recover partial progress from the global progress tracker
		global _worker_progress_tracker
		
		task_total = time.time() - task_start
		logger.error(
			"[STRICT_TIMEOUT] Problem %s forcibly terminated after %.2fs (limit=%ds) - "
			"likely stuck in FormalGeo equation solver",
			pid, task_total, time_limit
		)
		
		# Extract partial results from progress tracker
		steps_completed = _worker_progress_tracker.get("steps_completed", 0)
		partial_calls = _worker_progress_tracker.get("predicted_calls", [])
		partial_history = _worker_progress_tracker.get("history_records", [])
		last_step_info = _worker_progress_tracker.get("last_step_info")
		
		logger.info(
			"[STRICT_TIMEOUT] Recovered partial progress: steps=%s calls=%s",
			steps_completed, len(partial_calls)
		)
		
		# Build execution_steps from history records (same format as successful problems)
		execution_steps = []
		for hist in partial_history:
			step_call = hist.get("call", "unknown")
			step_updated = hist.get("updated", False)
			step_goal = hist.get("goal_solved", False)
			execution_steps.append(
				f"[Step {hist.get('step', '?')}] Executed: {step_call} | Updated: {step_updated} | Goal solved: {step_goal}"
			)
		
		timeout_result = {
			"problem_id": pid,
			"status": "timeout",
			"goal_solved": False,
			"error": f"Strict timeout: forcibly terminated after {time_limit} seconds",
			"run_id": run_id,
			"predicted_calls": partial_calls,
			"one_step_termination": "strict_timeout",
			"one_step_history": partial_history,  # Include partial history
			"execution_steps": execution_steps,   # Same format as successful problems
			"timeout_info": {
				"time_limit": time_limit,
				"elapsed_time": round(task_total, 2),
				"timeout_type": "strict",
				"steps_completed": steps_completed,
				"calls_attempted": len(partial_calls),
				"last_step": last_step_info,
				"note": "Process was forcibly terminated. Partial progress recovered from tracker.",
			},
			"timing_stats": {"total_time": task_total},
		}
		
		return timeout_result
		
	except Exception as exc:
		import traceback
		task_total = time.time() - task_start
		logger.error("[Error] Problem %s failed after %.2fs -> %s", pid, task_total, exc)
		logger.error("[Traceback] %s", traceback.format_exc())
		return {
			"problem_id": pid,
			"error": str(exc),
			"traceback": traceback.format_exc(),
			"run_id": run_id,
			"status": "error",
			"goal_solved": False,
			"predicted_calls": [],
			"timing_stats": {"total_time": task_total},
		}


# =============================================================================
# SECTION 9: MAIN PIPELINE & ORCHESTRATION
# =============================================================================

def _update_and_save_summary(
	summary_path: Path,
	summary_payload: Dict[str, Any],
	summary_counts: Dict[str, int],
	timing_total: float,
	timing_problem_count: int,
	timing_solved_total: float,
	timing_solved_count: int,
	processed_count: int,
	total_count: int,
	include_timing: bool = True,
) -> None:
	"""
	Update and save summary file in real-time.
	Called after each problem is processed to ensure data is persisted.
	"""
	# Calculate accuracy: solved / total
	total = summary_counts["total"] if summary_counts.get("total") is not None else total_count
	accuracy = summary_counts["solved"] / total if total > 0 else 0.0
	
	# Calculate timing averages
	avg_time = timing_total / timing_problem_count if timing_problem_count > 0 else 0.0
	avg_solved_time = timing_solved_total / timing_solved_count if timing_solved_count > 0 else 0.0
	
	# Update summary payload
	summary_payload["accuracy"] = round(accuracy, 4)
	summary_payload["summary"] = dict(summary_counts)
	summary_payload["progress"] = {
		"processed": processed_count,
		"total": total_count,
		"percent": round(processed_count / total_count * 100, 1) if total_count > 0 else 0,
	}
	if include_timing:
		summary_payload["timing"] = {
			"avg_time_all": round(avg_time, 2),
			"avg_time_solved": round(avg_solved_time, 2),
			"total_time": round(timing_total, 2),
			"problem_count": timing_problem_count,
			"solved_count": timing_solved_count,
		}
	else:
		summary_payload.pop("timing", None)
	
	# Sort problems by problem_id before saving
	summary_payload["problems"].sort(key=lambda x: x.get("problem_id", 0))
	
	# Save to file
	save_result(summary_path, summary_payload)


def main() -> None:
	args = parse_cli_args()
	
	global DATA_SPLITS, TEST_PROBLEM_IDS
	
	DATA_SPLITS = _DEFAULT_DATA_SPLITS.copy()

	TEST_PROBLEM_IDS = sorted(DATA_SPLITS[TEST_SPLIT_NAME])
	

	if not args.quick_start and not EMBEDDING_API_KEY:
		raise ValueError("Embedding API key is required. Set EMBEDDING_API_KEY in code or export EMBEDDING_API_KEY.")
	
	dataset_root = DEFAULT_DATASETS_PATH / DEFAULT_DATASET_NAME
	if not dataset_root.exists():
		raise FileNotFoundError(f"Dataset not found at {dataset_root}")
	
	problems_dir = dataset_root / "problems"
	diagrams_dir = dataset_root / "diagrams"
	gdl_path = dataset_root / "gdl" / "theorem_GDL.json"
	
	if not problems_dir.exists():
		raise FileNotFoundError(f"Problems directory missing: {problems_dir}")
	if not gdl_path.exists():
		raise FileNotFoundError(f"GDL file missing: {gdl_path}")
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	logger, run_id = setup_logger(OUTPUT_DIR)
	if not args.show_solution:
		logger.setLevel(logging.CRITICAL)
	else:
		logger.info("[Pipeline] show_solution=True: force single-thread execution with step-by-step logs")
		if args.num_workers != 1:
			logger.info("[Pipeline] Overriding num_workers %s -> 1 for sequential display", args.num_workers)
		args.num_workers = 1

	if not args.quick_start:
		logger.info(
			"[Init] dataset_root=%s problems_dir=%s diagrams_dir=%s num_workers=%s",
			dataset_root,
			problems_dir,
			diagrams_dir,
			args.num_workers,
		)
		logger.info("[Config] %s", json.dumps(vars(args), default=str, ensure_ascii=False))
	problem_ids = resolve_problem_ids(
		args.problem_id,
		args.problem_ids,
		args.max_problems, 
		args.start_problem_id,
	)
	
	summary_counts = {
		"total": len(problem_ids),
		"solved": 0,
		"failed": 0,
		"errors": 0,
		"timeout": 0,  # Timeout count (also counted in failed)
	}
	
	# DISABLED: details file is too large and causes performance issues
	# details_path = OUTPUT_DIR / f"run_{run_id}_details.json"
	# details_payload: Dict[str, Any] = {
	# 	"run_id": run_id,
	# 	"generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
	# 	"problems": [],
	# }
	# save_result(details_path, details_payload)
	
	summary_path = OUTPUT_DIR / f"run_{run_id}_summary.json"
	summary_payload: Dict[str, Any] = {
		"model": LLM_MODEL,
		"top_k": args.top_k,
		"accuracy": 0.0,
		"summary": dict(summary_counts),
		"run_id": run_id,
		"generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
		"problems": [],
	}
	
	save_result(summary_path, summary_payload)
	
	# Prepare worker arguments
	args_dict = vars(args)
	dataset_loader_kwargs = {
		"dataset_name": DEFAULT_DATASET_NAME,
		"datasets_path": str(DEFAULT_DATASETS_PATH),
	}
	
	worker_args = [
		(pid, args_dict, run_id, problems_dir)
		for pid in problem_ids
	]
	
	# Track processed count for incremental saves
	processed_count = 0
	
	# Timing statistics (for all processed problems)
	timing_total = 0.0
	timing_solved_total = 0.0
	timing_solved_count = 0
	timing_problem_count = 0  # Count of problems with timing data
	
	# Use Pool with initializer to load data once per worker
	# Create shared queue for worker ID assignment
	manager = Manager()
	worker_id_queue = manager.Queue()
	# Pre-populate queue with worker IDs (0, 1, 2, 3, ...)
	for i in range(args.num_workers):
		worker_id_queue.put(i)

	pool = None
	if args.show_solution:
		init_worker(dataset_loader_kwargs, args.retrieval_mode, args.embedding_model, problems_dir, worker_id_queue, True, args.quick_start, args.quick_start_file)
		results_iter = (process_problem_worker(w_arg) for w_arg in worker_args)
	else:
		pool = Pool(processes=args.num_workers, 
			 initializer=init_worker, 
			 initargs=(dataset_loader_kwargs, args.retrieval_mode, args.embedding_model, problems_dir, worker_id_queue, False, args.quick_start, args.quick_start_file))

		# Use imap_unordered - timeout is now enforced by func_timeout inside worker
		logger.info(
			"[Pipeline] Using func_timeout with strict timeout=%ds per problem",
			args.time_limit
		)
		results_iter = pool.imap_unordered(process_problem_worker, worker_args)
		
	try:
		# Process results one by one with REAL-TIME SAVING
		for result in tqdm(results_iter, total=len(problem_ids), desc="Processing problems", unit="problem", disable=args.quick_start):
			processed_count += 1
			
			# New result from solver - accumulate timing and create summary
			timing_stats = result.get("timing_stats", {})
			total_time = timing_stats.get("total_time", 0)
			timing_total += total_time
			timing_problem_count += 1  # Count this problem for timing average
			
			# Update counts based on status (mutually exclusive - 5 categories only)
			status = result.get("status", "")
			if status == "solved":
				summary_counts["solved"] += 1
				timing_solved_total += total_time
				timing_solved_count += 1
			elif status == "timeout":
				# Timeout is separate category - do NOT count as failed
				summary_counts["timeout"] += 1
				# Log timeout details
				timeout_info = result.get("timeout_info", {})
				if logger:
					logger.warning(
						"[Timeout] problem=%s type=%s elapsed=%.2fs steps=%s calls=%s",
						result.get("problem_id"),
						timeout_info.get("timeout_type", "unknown"),
						timeout_info.get("elapsed_time", total_time),
						timeout_info.get("steps_completed", "?"),
						timeout_info.get("calls_attempted", "?"),
					)
			elif status == "error":
				summary_counts["errors"] += 1
			else:  # failed
				summary_counts["failed"] += 1
			
			# Create summary from full result
			problem_summary = summarize_problem_outcome(result)
			
			# Collect result summary
			summary_payload["problems"].append(problem_summary)
			
			# REAL-TIME SAVE: Save after EVERY problem to ensure data is persisted
			# even if the process is interrupted
			_update_and_save_summary(
				summary_path, summary_payload, summary_counts,
				timing_total, timing_problem_count, timing_solved_total, timing_solved_count,
				processed_count, len(problem_ids),
				include_timing=not args.quick_start,
			)
			
			# Log progress every 10 problems
			if processed_count % 10 == 0 or processed_count == len(problem_ids):
				if logger:
					total = summary_counts["total"]
					accuracy = summary_counts["solved"] / total if total > 0 else 0.0
					logger.info(
						"[Progress] Processed %d/%d problems, accuracy=%.4f, solved=%d, failed=%d, errors=%d, timeout=%d (saved)",
						processed_count,
						len(problem_ids),
						accuracy,
						summary_counts["solved"],
						summary_counts["failed"],
						summary_counts["errors"],
						summary_counts["timeout"]
					)
	finally:
		if pool is not None:
			pool.close()
			pool.join()
	
	# Final summary (already saved in last batch)
	if not args.quick_start:
		logger.info("[Pipeline] All problems processed, summary saved to %s", summary_path)
	
	# Calculate final accuracy: solved / total
	# Status is mutually exclusive: solved OR failed OR error OR timeout
	total_processed = summary_counts["solved"] + summary_counts["failed"] + summary_counts["errors"] + summary_counts["timeout"]
	total = summary_counts["total"]
	accuracy = summary_counts["solved"] / total if total else 0.0
	
	# Calculate average times (only for problems that were actually run with timing data)
	avg_time_all = timing_total / timing_problem_count if timing_problem_count > 0 else 0.0
	avg_time_solved = timing_solved_total / timing_solved_count if timing_solved_count > 0 else 0.0
	
	# total = original problem count, should equal processed
	if args.quick_start:
		logger.info("[Summary] accuracy=%.4f", accuracy)
	else:
		logger.info(
			"[Summary] total=%s processed=%s solved=%s failed=%s errors=%s timeout=%s accuracy=%.4f",
			summary_counts["total"],
			total_processed,
			summary_counts["solved"],
			summary_counts["failed"],
			summary_counts["errors"],
			summary_counts["timeout"],
			accuracy,
		)
		logger.info(
			"[Timing] avg_time_all=%.2fs avg_time_solved=%.2fs total_time=%.2fs (based on %s problems with timing data, %s solved)",
			avg_time_all,
			avg_time_solved,
			timing_total,
			timing_problem_count,
			timing_solved_count,
		)
	
	if not args.show_solution:
		print(f"[Result] Summary saved to {summary_path}")
		print(
			f"[Result] total={summary_counts['total']} processed={total_processed} "
			f"solved={summary_counts['solved']} failed={summary_counts['failed']} "
			f"errors={summary_counts['errors']} timeout={summary_counts['timeout']} "
			f"accuracy={accuracy:.4f}"
		)
	
	summary_payload["summary"] = dict(summary_counts)
	summary_payload["accuracy"] = round(accuracy, 6)
	if not args.quick_start:
		summary_payload["timing"] = {
			"avg_time_all": round(avg_time_all, 2),
			"avg_time_solved": round(avg_time_solved, 2),
			"total_time": round(timing_total, 2),
			"problem_count": timing_problem_count,
			"solved_count": timing_solved_count
		}
	else:
		summary_payload.pop("timing", None)
	
	summary_payload["completed_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
	save_result(summary_path, summary_payload)


if __name__ == "__main__":
	main()
