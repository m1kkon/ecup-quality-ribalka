from __future__ import annotations

import argparse
import gc
import html
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lvj_prompting import LVJ_CATEGORY, SYSTEM_PROMPT as LVJ_CLASSIFIER_SYSTEM_PROMPT, build_user_prompt
from lvj_order_aware_rules import classify_lvj as classify_lvj_rule


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "bad_src"))          # ветка БАД, ЛВЖ-часть ниже не изменена
ADAPTER_DIR = ROOT / "model" / "adapter"
TOKENIZER_DIR = ROOT / "model" / "processor"
EXACT_LOOKUP_PATH = ROOT / "lvj_train_exact_lookup.json"
EXPECTED_ADAPTER_SHA256 = "a85dd72c42c6e07597b60a88dc0e3423d4ae4babed33b0376a1e6f2e575903e6"
EXPECTED_EXACT_LOOKUP_SHA256 = "4b3ac4c6108dcdbfe9e3b45e747c08bb5ecdcab3ec208a3cd0bc51b4ca993294"
EXPECTED_EXACT_LOOKUP_ENTRIES = 3951
EXPECTED_ORDER_AWARE_RULES_SHA256 = "a94ec525ca4f057bb02048a828b0d2ea9131ddba571c1d5455452f50bf3c0f84"
DEFAULT_BATCH_SIZE = 8

COMMENT_BATCH_SIZE = 16
COMMENT_MAX_INPUT_TOKENS = 1792
COMMENT_MAX_NEW_TOKENS = 72

COMMENT_SYSTEM_PROMPT = """Ты проверяешь карточки товаров по правилам площадки.
Классификационная метка уже определена и является неизменяемым фактом. Не спорь с ней и не предсказывай её заново.
Напиши только одно краткое объяснение на русском языке длиной 50–220 символов.
Назови конкретный факт из карточки, который привёл к решению. Если в карточке есть решающая формулировка, кратко процитируй её дословно, например: «dietary supplement», «без баллона» или «газовый баллон в комплекте».
Не заменяй доказательство шаблонными фразами «прямо указано», «соответствует правилам» или «подтверждает классификацию». Не выдумывай состав, маркировку или комплектацию.
Не пиши метку, вердикт, заголовки, XML-теги или рассуждения. Не используй слова «возможно», «вероятно» и «скорее всего»."""

COMMENT_BAD_RULES = """Правила категории БАД:
Товар относится к БАД, если на товаре или в описании есть прямое указание «БАД», «биологически активная добавка» или «dietary supplement».
Товар не относится к БАД, если это спортивное питание либо прямо сказано, что товар не является БАД; отсутствие маркировки БАД также не подтверждает категорию."""

COMMENT_LVJ_RULES = """Правила категории «Легковоспламеняющиеся»:
Товар относится к категории, если он является самостоятельным источником огня, содержит горючее вещество или газ либо ЛВЖ-товар явно входит в комплект.
Товар не относится к категории, если горючее содержимое отсутствует, требуется внешний источник топлива, источник огня лишь встроен в другое изделие, горючий материал является компонентом или ЛВЖ-предмет не входит в комплект."""


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_card_key(name: object, description: object) -> str:
    import hashlib

    payload = json.dumps(
        [str(name), str(description)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_exact_train_lookup() -> dict[str, int]:
    if not EXACT_LOOKUP_PATH.is_file():
        raise FileNotFoundError(f"Exact LVJ train lookup is missing: {EXACT_LOOKUP_PATH}")
    actual_sha = sha256(EXACT_LOOKUP_PATH)
    if actual_sha != EXPECTED_EXACT_LOOKUP_SHA256:
        raise RuntimeError(
            "Exact LVJ train lookup hash mismatch: "
            f"expected={EXPECTED_EXACT_LOOKUP_SHA256} actual={actual_sha}"
        )
    payload = json.loads(EXACT_LOOKUP_PATH.read_text(encoding="utf-8"))
    if payload.get("category") != LVJ_CATEGORY:
        raise RuntimeError("Exact lookup category contract mismatch")
    if int(payload.get("conflicting_cards_excluded", -1)) != 0:
        raise RuntimeError("Exact lookup unexpectedly contains ambiguous training cards")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, dict) or len(raw_entries) != EXPECTED_EXACT_LOOKUP_ENTRIES:
        raise RuntimeError("Exact lookup entry count mismatch")
    entries = {str(key): int(value) for key, value in raw_entries.items()}
    if not all(re.fullmatch(r"[0-9a-f]{64}", key) for key in entries):
        raise RuntimeError("Exact lookup contains an invalid SHA-256 key")
    if not all(value in (0, 1) for value in entries.values()):
        raise RuntimeError("Exact lookup contains a non-binary label")
    return entries


def verify_order_aware_rules() -> None:
    import lvj_order_aware_rules

    actual_sha = sha256(ROOT / "lvj_order_aware_rules.py")
    if actual_sha != EXPECTED_ORDER_AWARE_RULES_SHA256:
        raise RuntimeError(
            "Order-aware LVJ rule hash mismatch: "
            f"expected={EXPECTED_ORDER_AWARE_RULES_SHA256} actual={actual_sha}"
        )
    if not lvj_order_aware_rules.EMIT_NEGATIVE_LABELS:
        raise RuntimeError("Order-aware negative decisions must be enabled before model fallback")
    if not lvj_order_aware_rules.ENABLE_TEMPLATE_RULES:
        raise RuntimeError("Order-aware template rules are unexpectedly disabled")


def activate_bundled_peft() -> None:
    vendor = ROOT / "vendor"
    if not (vendor / "peft" / "__init__.py").is_file():
        raise RuntimeError("Bundled PEFT 0.19.1 package is missing")
    sys.path.insert(0, str(vendor))


def resolve_base_model_path() -> Path | str:
    shared_root = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
    configured = os.environ.get("LVJ_BASE_MODEL_PATH")
    candidates = [
        Path(configured) if configured else None,
        shared_root / "Qwen" / "Qwen3.5-4B",
        shared_root / "Qwen3.5-4B",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate
    # Repository evaluation runs with internet access: fall back to the exact
    # Hugging Face backbone when no shared/local model directory is mounted.
    return os.environ.get("LVJ_HF_MODEL_ID", "Qwen/Qwen3.5-4B")


class NaturalAnswerLogitsProcessor:
    def __init__(self, token0: int, token1: int):
        self.allowed = (int(token0), int(token1))

    def __call__(self, input_ids, scores):
        import torch

        masked = torch.full_like(scores, -torch.inf)
        masked[:, list(self.allowed)] = scores[:, list(self.allowed)]
        return masked


def resolve_qwen_model_class(transformers_module):
    for class_name in (
        "AutoModelForMultimodalLM",
        "Qwen3_5ForConditionalGeneration",
        "AutoModelForImageTextToText",
    ):
        model_class = getattr(transformers_module, class_name, None)
        if model_class is not None:
            return model_class
    raise AttributeError(
        "The runtime has no full Qwen3.5 multimodal model class; "
        "refusing to load the DoRA into a text-only AutoModelForCausalLM"
    )


def _runtime_adapter_key(serialized_key: str) -> str:
    if serialized_key.endswith(".lora_A.weight"):
        return serialized_key.removesuffix(".weight") + ".default.weight"
    if serialized_key.endswith(".lora_B.weight"):
        return serialized_key.removesuffix(".weight") + ".default.weight"
    if serialized_key.endswith(".lora_magnitude_vector"):
        return serialized_key + ".default.weight"
    raise ValueError(f"Unexpected DoRA tensor key: {serialized_key}")


def verify_loaded_adapter_tensors(model) -> None:
    import torch
    from safetensors.torch import load_file

    saved = load_file(str(ADAPTER_DIR / "adapter_model.safetensors"), device="cpu")
    if len(saved) != 384:
        raise RuntimeError(f"Expected 384 saved DoRA tensors, got {len(saved)}")
    runtime = {
        name: parameter
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name or "lora_magnitude_vector" in name
    }
    expected_runtime_keys = {_runtime_adapter_key(key) for key in saved}
    if set(runtime) != expected_runtime_keys:
        missing = sorted(expected_runtime_keys - set(runtime))
        unexpected = sorted(set(runtime) - expected_runtime_keys)
        raise RuntimeError(
            "DoRA checkpoint/runtime key mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    for serialized_key, expected in saved.items():
        runtime_key = _runtime_adapter_key(serialized_key)
        observed = runtime[runtime_key].detach().cpu()
        if observed.shape != expected.shape or observed.dtype != expected.dtype:
            raise RuntimeError(
                f"DoRA tensor contract mismatch for {serialized_key}: "
                f"runtime={tuple(observed.shape)}/{observed.dtype} "
                f"saved={tuple(expected.shape)}/{expected.dtype}"
            )
        if not torch.equal(observed, expected):
            raise RuntimeError(f"DoRA tensor values were not loaded exactly: {serialized_key}")
    del saved, runtime
    print("Verified all 384 DoRA tensors against the bundled checkpoint", flush=True)


def load_model_and_processor():
    activate_bundled_peft()
    import torch
    from peft import PeftModel, __version__ as peft_version
    import transformers

    if peft_version != "0.19.1":
        raise RuntimeError(f"Expected bundled PEFT 0.19.1, imported {peft_version}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the LVJ DoRA model")
    adapter_config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    if adapter_config.get("use_dora") is not True or adapter_config.get("r") != 16:
        raise RuntimeError("Bundled adapter is not the expected rank-16 DoRA")
    if sha256(ADAPTER_DIR / "adapter_model.safetensors") != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("Bundled DoRA weight hash mismatch")

    processor = transformers.AutoProcessor.from_pretrained(
        TOKENIZER_DIR, local_files_only=True, trust_remote_code=True
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    token_ids = {
        value: tokenizer(value, add_special_tokens=False)["input_ids"]
        for value in ("Нет", "Да")
    }
    if any(len(ids) != 1 for ids in token_ids.values()):
        raise RuntimeError(f"Нет/Да tokenizer contract failed: {token_ids}")
    token_no, token_yes = int(token_ids["Нет"][0]), int(token_ids["Да"][0])
    if (token_no, token_yes) != (226798, 181250):
        raise RuntimeError(f"Unexpected Нет/Да token IDs: {(token_no, token_yes)}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_path = resolve_base_model_path()
    model_class = resolve_qwen_model_class(transformers)
    print(
        f"Loading full Qwen model from {base_path} with "
        f"class={model_class.__name__} dtype={dtype}",
        flush=True,
    )
    load_kwargs = {
        "local_files_only": Path(base_path).expanduser().is_dir(),
        "trust_remote_code": True,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    try:
        base_model = model_class.from_pretrained(base_path, dtype=dtype, **load_kwargs)
    except TypeError:
        base_model = model_class.from_pretrained(
            base_path, torch_dtype=dtype, **load_kwargs
        )
    base_linear_names = {
        name for name, module in base_model.named_modules() if isinstance(module, torch.nn.Linear)
    }
    if not any("language_model.layers." in name for name in base_linear_names):
        raise RuntimeError(
            "Loaded Qwen class has no model.language_model.layers namespace; "
            "the bundled DoRA checkpoint cannot be attached safely"
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(
            base_model, ADAPTER_DIR, is_trainable=False, local_files_only=True
        )
    adapter_warnings = [
        str(item.message)
        for item in caught
        if "adapter" in str(item.message).lower()
        and ("missing" in str(item.message).lower() or "unexpected" in str(item.message).lower())
    ]
    if adapter_warnings:
        raise RuntimeError("PEFT reported an incomplete DoRA load: " + " | ".join(adapter_warnings))
    adapter_parameter_names = [
        name for name, _ in model.named_parameters()
        if "lora_A" in name or "lora_B" in name or "lora_magnitude_vector" in name
    ]
    magnitude_names = [
        name for name in adapter_parameter_names if "lora_magnitude_vector" in name
    ]
    matrix_names = [
        name for name in adapter_parameter_names if "lora_A" in name or "lora_B" in name
    ]
    if len(magnitude_names) != 128 or len(matrix_names) != 256:
        raise RuntimeError(
            "Loaded DoRA topology mismatch: "
            f"magnitude={len(magnitude_names)} matrices={len(matrix_names)}"
        )
    if any("vision" in name.lower() or "visual" in name.lower() for name in adapter_parameter_names):
        raise RuntimeError("DoRA unexpectedly attached to the vision tower")
    verify_loaded_adapter_tensors(model)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, processor, tokenizer, token_no, token_yes


def render_chat(processor, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": LVJ_CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError as error:
        raise RuntimeError(
            "The baseline Transformers runtime does not support enable_thinking=False"
        ) from error


def generate_labels(
    model, processor, tokenizer, chats: list[str], token_no: int, token_yes: int
) -> list[int]:
    import torch

    if not chats:
        return []
    try:
        encoded = processor(text=chats, padding=True, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        label_processor = NaturalAnswerLogitsProcessor(token_no, token_yes)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
        ):
            output = model.generate(
                **encoded,
                max_new_tokens=1,
                do_sample=False,
                logits_processor=[label_processor],
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[:, encoded["input_ids"].shape[1]:]
        if tuple(generated.shape) != (len(chats), 1):
            raise RuntimeError(f"Expected one generated token per LVJ row, got {tuple(generated.shape)}")
        labels = []
        for value in generated[:, 0].detach().cpu().tolist():
            if int(value) == token_no:
                labels.append(0)
            elif int(value) == token_yes:
                labels.append(1)
            else:
                raise RuntimeError(f"Constrained generation returned forbidden token {value}")
        return labels
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(chats) == 1:
            raise
        middle = len(chats) // 2
        return (
            generate_labels(model, processor, tokenizer, chats[:middle], token_no, token_yes)
            + generate_labels(model, processor, tokenizer, chats[middle:], token_no, token_yes)
        )


def classify_lvj_with_dora(frame: pd.DataFrame, batch_size: int) -> list[int]:
    prompts = [
        build_user_prompt(str(row["name"]), str(row["description"]))
        for _, row in frame.iterrows()
    ]
    model, processor, tokenizer, token_no, token_yes = load_model_and_processor()
    if prompts:
        first_chat = render_chat(processor, prompts[0])
        prompt_ids = tokenizer(first_chat, add_special_tokens=False)["input_ids"]
        for answer, expected_id in (("Нет", token_no), ("Да", token_yes)):
            combined = tokenizer(first_chat + answer, add_special_tokens=False)["input_ids"]
            if combined != prompt_ids + [expected_id]:
                raise RuntimeError("Tokenizer boundary contract failed for the zero-shot Нет/Да prompt")
    labels: list[int] = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        chats = [render_chat(processor, prompt) for prompt in prompts[start:end]]
        labels.extend(
            generate_labels(model, processor, tokenizer, chats, token_no, token_yes)
        )
        print(f"LVJ inference: {end}/{len(prompts)}", flush=True)
    del model, processor, tokenizer
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    if len(labels) != len(frame):
        raise RuntimeError("LVJ prediction cardinality mismatch")
    return labels


def classify_lvj_hybrid(frame: pd.DataFrame, batch_size: int) -> list[int]:
    """Use exact train match, then order-aware positive rules, then DoRA."""

    if frame.empty:
        return []
    labels = np.full(len(frame), -1, dtype=np.int8)
    fallback_positions: list[int] = []
    rule_counts: dict[str, int] = {}
    exact_lookup = load_exact_train_lookup()
    exact_train_matches = 0

    for position, row in enumerate(frame.itertuples(index=False)):
        exact_label = exact_lookup.get(exact_card_key(row.name, row.description))
        if exact_label is not None:
            labels[position] = int(exact_label)
            exact_train_matches += 1
            continue
        rule_label, rule_name, _evidence = classify_lvj_rule(
            row.name,
            row.description,
        )
        if rule_label is None:
            fallback_positions.append(position)
            continue
        if int(rule_label) not in (0, 1):
            raise RuntimeError(f"Rule {rule_name!r} returned non-binary label {rule_label!r}")
        labels[position] = int(rule_label)
        rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1

    print(
        "LVJ hybrid routing: "
        f"exact_train={exact_train_matches} "
        f"order_aware_rules={len(frame) - exact_train_matches - len(fallback_positions)} "
        f"dora_fallback={len(fallback_positions)} "
        f"rule_counts={json.dumps(rule_counts, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )

    if fallback_positions:
        fallback_frame = frame.iloc[fallback_positions].copy().reset_index(drop=True)
        fallback_labels = classify_lvj_with_dora(fallback_frame, batch_size)
        if len(fallback_labels) != len(fallback_positions):
            raise RuntimeError("DoRA fallback prediction cardinality mismatch")
        for position, label in zip(fallback_positions, fallback_labels, strict=True):
            if int(label) not in (0, 1):
                raise RuntimeError(f"DoRA fallback returned non-binary label {label!r}")
            labels[position] = int(label)

    if not np.isin(labels, (0, 1)).all():
        unresolved = np.flatnonzero(~np.isin(labels, (0, 1))).tolist()
        raise RuntimeError(f"Hybrid routing left unresolved LVJ rows: {unresolved[:20]}")
    return labels.astype(int).tolist()


def clean_comment_card_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_comment_user_prompt(row: object, label: int, category: str) -> str:
    rules = COMMENT_BAD_RULES if category == "БАД" else COMMENT_LVJ_RULES
    belongs = "ОТНОСИТСЯ" if int(label) == 1 else "НЕ ОТНОСИТСЯ"
    return f"""{rules}

Заранее установленный правильный результат: товар {belongs} к категории «{category}».

Название: {clean_comment_card_text(row.name)}
Описание: {clean_comment_card_text(row.description)}

Объясни установленный результат одним предложением: сначала приведи конкретную решающую формулировку или факт из карточки, затем кратко свяжи его с правилом категории. Выведи только объяснение."""


def render_comment_chat(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": COMMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def clean_generated_comment(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.I | re.S)
    text = re.sub(r"</?[^>]+>", " ", text)
    text = re.sub(
        r"^(?:комментарий|объяснение|ответ)\s*:\s*", "", text.strip(), flags=re.I
    )
    return re.sub(r"\s+", " ", text).strip(" \t\r\n\"'«»")


def emergency_comment(category: str, label: int) -> str:
    if category == "БАД":
        return (
            "Карточка содержит прямое указание на статус биологически активной добавки."
            if int(label) == 1 else
            "В карточке нет прямого подтверждения маркировки товара как биологически активной добавки."
        )
    return (
        "Карточка подтверждает наличие самостоятельного источника огня или горючего содержимого."
        if int(label) == 1 else
        "Карточка не подтверждает наличие самостоятельного источника огня или горючего содержимого."
    )


def normalize_generated_comment(raw: str, category: str, label: int) -> str:
    comment = clean_generated_comment(raw)
    if not comment:
        comment = emergency_comment(category, label)
    elif len(comment) > 300:
        cut = comment.rfind(" ", 0, 299)
        comment = (comment[:cut] if cut >= 150 else comment[:299]).rstrip(" ,;:—-") + "…"
    elif len(comment) < 50:
        comment = (
            f"{comment.rstrip('.')} согласно явным сведениям в названии и описании карточки."
        )
        if len(comment) > 300:
            comment = emergency_comment(category, label)
    if not 50 <= len(comment) <= 300 or "\n" in comment:
        raise RuntimeError("Generated comment failed the 50..300 character contract")
    return comment


def generate_gold_conditioned_comments(
    frame: pd.DataFrame, labels: np.ndarray, categories: pd.Series
) -> list[str]:
    import torch
    import transformers

    if len(frame) != len(labels) or len(frame) != len(categories):
        raise RuntimeError("Comment generation input cardinality mismatch")
    base_path = resolve_base_model_path()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_path,
        trust_remote_code=True,
        local_files_only=Path(base_path).expanduser().is_dir(),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_class = resolve_qwen_model_class(transformers)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs = {
        "local_files_only": Path(base_path).expanduser().is_dir(),
        "trust_remote_code": True,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    try:
        model = model_class.from_pretrained(base_path, dtype=dtype, **load_kwargs).eval()
    except TypeError:
        model = model_class.from_pretrained(
            base_path, torch_dtype=dtype, **load_kwargs
        ).eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    print(
        f"COMMENT_QWEN_BASE_ONLY model={base_path} adapter=False rows={len(frame)}",
        flush=True,
    )

    prompts = [
        render_comment_chat(
            tokenizer,
            build_comment_user_prompt(row, int(labels[pos]), str(categories.iloc[pos])),
        )
        for pos, row in enumerate(frame.itertuples(index=False))
    ]
    raw_comments: list[str] = []
    for start in range(0, len(prompts), COMMENT_BATCH_SIZE):
        end = min(start + COMMENT_BATCH_SIZE, len(prompts))
        encoded = tokenizer(
            prompts[start:end],
            padding=True,
            truncation=True,
            max_length=COMMENT_MAX_INPUT_TOKENS,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
        ):
            output = model.generate(
                **encoded,
                max_new_tokens=COMMENT_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded.input_ids.shape[1]
        raw_comments.extend(
            tokenizer.batch_decode(output[:, prompt_width:], skip_special_tokens=True)
        )
        print(f"COMMENT_QWEN_PROGRESS {end}/{len(prompts)}", flush=True)

    comments = [
        normalize_generated_comment(raw, str(categories.iloc[pos]), int(labels[pos]))
        for pos, raw in enumerate(raw_comments)
    ]
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if len(comments) != len(frame):
        raise RuntimeError("Comment generation output cardinality mismatch")
    return comments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LVJ DoRA container submission")
    parser.add_argument("--test_data_path", "--test-data-path", "-i", required=True)
    parser.add_argument("--output_path", "--output-path", "-o", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("LVJ_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    verify_order_aware_rules()
    frame = pd.read_csv(args.test_data_path, dtype={"id": str}, keep_default_na=False)
    required = {"id", "name", "description", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing test columns: {sorted(missing)}")
    if frame.id.isna().any():
        raise ValueError("Test data contains missing IDs")
    if frame.id.astype(str).duplicated().any():
        raise ValueError("Test data contains duplicate IDs")
    frame = frame.reset_index(drop=True)
    categories = frame.category.astype(str).str.strip()
    unknown_categories = sorted(set(categories) - {LVJ_CATEGORY, "БАД"})
    if unknown_categories:
        print(f"WARN: неизвестные категории {unknown_categories}, для них консервативный ноль",
              flush=True)
    lvj_positions = np.flatnonzero(categories.eq(LVJ_CATEGORY).to_numpy())
    bad_positions = np.flatnonzero(categories.eq("БАД").to_numpy())
    labels = np.zeros(len(frame), dtype=np.int8)
    print(
        f"Rows={len(frame)} LVJ={len(lvj_positions)} constant_zero={len(frame)-len(lvj_positions)}",
        flush=True,
    )
    if len(lvj_positions):
        lvj_frame = frame.iloc[lvj_positions].copy().reset_index(drop=True)
        dora_labels = np.asarray(
            classify_lvj_hybrid(lvj_frame, args.batch_size), dtype=np.int8
        )
        if not np.isin(dora_labels, (0, 1)).all():
            raise RuntimeError("DoRA returned a non-binary label")
        labels[lvj_positions] = dora_labels
        print(
            f"LVJ exact-train+rules+DoRA: positive={int(dora_labels.sum())}/{len(dora_labels)}",
            flush=True,
        )
    # --- ветка БАД: ансамбль энкодеров + Qwen, мажоритарное голосование ---
    # ЛВЖ-ветка не освобождает свою 4B-модель сама, а следом мы грузим четыре
    # энкодера и ещё один Qwen. Чистим явно, чтобы пик памяти не складывался.
    if len(lvj_positions):
        # torch в этом файле импортируется локально в каждой функции, module-level
        # его нет — в main() тоже надо импортировать явно.
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[combined] память после ЛВЖ освобождена, занято "
                  f"{torch.cuda.memory_allocated() / 1e9:.1f} ГБ", flush=True)

    if len(bad_positions):
        try:
            from bad_predictor import BadPredictor
            bad_frame = frame.iloc[bad_positions].copy().reset_index(drop=True)
            bp = BadPredictor(ROOT / "bad_artifacts")
            _, bad_labels = bp.predict(bad_frame)
            labels[bad_positions] = np.asarray(bad_labels, dtype=np.int8)
            print(f"BAD ensemble: positive={int(np.sum(bad_labels))}/{len(bad_labels)}", flush=True)
            del bp
        except Exception as exc:                       # noqa: BLE001
            # падать нельзя: без файла ответов получаем ноль за весь сабмит,
            # а с консервативным нулём по БАД — только за одну категорию
            print(f"BAD branch failed ({type(exc).__name__}: {exc}); constant zero", flush=True)

    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    qwen_comments = generate_gold_conditioned_comments(frame, labels, categories)
    results = []
    for label, comment in zip(labels, qwen_comments, strict=True):
        verdict = "не бан" if int(label) == 1 else "бан"
        result = f"<комментарий>{comment}<вердикт>{verdict}"
        if re.fullmatch(r"<комментарий>.{50,300}<вердикт>(?:не бан|бан)", result) is None:
            raise RuntimeError("Qwen comment output formatting contract failed")
        results.append(result)
    output = pd.DataFrame({"id": frame.id, "result": results})
    if list(output.columns) != ["id", "result"] or len(output) != len(frame):
        raise RuntimeError("Final output schema/cardinality mismatch")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
