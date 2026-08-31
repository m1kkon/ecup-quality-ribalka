from __future__ import annotations

"""Standalone Kaggle full-train DoRA for LVJ with data-agreed compact rules.

Uses all 3951 exact-deduplicated rows for training. The prompt contains only the
data-agreed classification rules plus the current product name/description: no family
field, no static examples, no dynamic/retrieval examples, and no TF-IDF prompt prior.
This controlled variant differs from BEST_DORA_3E_SYSTEM_RULES_YESNO_FULLTRAIN.py
only in the system-prompt wording and experiment/output identifiers. Training runs
for exactly 3.0 epochs and saves DoRA checkpoints at 0.5, 1.0, 1.5, 2.0, 2.5, and
3.0 epochs. The loss remains restricted to the two natural answer tokens
{"Нет", "Да"}, with negative row weight 1.0 and positive row weight 6.0.
"""

import base64
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


PROFILES = {
    "official_128_r16": {
        "rank": 16,
        "alpha": 16,
        "dropout": 0.0,
        "expected_modules": 128,
        "target_family": "official_attention_plus_mlp",
    },
    "all_248_r8": {
        "rank": 8,
        "alpha": 8,
        "dropout": 0.0,
        "expected_modules": 248,
        "target_family": "all_text_attention_delta_mlp",
    },
}


CONFIG = {
    "experiment_version": "h067_lvj_dora_fulltrain_data_agreed_rules_yesno_positive6_3ep",
    "active_profile": "official_128_r16",
    "data_path_candidates": (
        Path("/kaggle/input/datasets/m1ck0n/ozon-ecup/data.csv"),
        Path("/kaggle/input/ozon-ecup/data.csv"),
        Path("data.csv"),
    ),
    "model_id": (
        "/kaggle/input/datasets/m1ck0n/qwen3-5-4b-huggingface"
        if Path("/kaggle/input/datasets/m1ck0n/qwen3-5-4b-huggingface").is_dir()
        else "Qwen/Qwen3.5-4B"
    ),
    "output_root": Path("/kaggle/working/qwen35_lvj_dora_fulltrain_data_agreed_rules_yesno_3ep"),
    "expected_data_sha256": "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510",
    "expected_clean_rows": 3951,
    "expected_positives": 143,
    "expected_groups": 2651,
    "positive_class_multiplier": 6.0,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "scheduler_horizon_epochs": 3.0,
    "checkpoint_epochs": (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    "generation_batch_size": 8,
    "max_new_tokens": 1,
    "learning_rate": 5.0e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "max_grad_norm": 1.0,
    "peft_version": "0.19.1",
    "seed": 20260810,
    "training_dtype": "bfloat16",
}

# Optional overrides used by the repository's single entry point. Kaggle's
# original defaults remain byte-for-byte equivalent when the variables are unset.
if os.environ.get("LVJ_TRAIN_DATA_PATH"):
    CONFIG["data_path_candidates"] = (Path(os.environ["LVJ_TRAIN_DATA_PATH"]),)
if os.environ.get("LVJ_BASE_MODEL_PATH"):
    CONFIG["model_id"] = os.environ["LVJ_BASE_MODEL_PATH"]
if os.environ.get("LVJ_OUTPUT_ROOT"):
    CONFIG["output_root"] = Path(os.environ["LVJ_OUTPUT_ROOT"])

SYSTEM_PROMPT = """Ты классифицируешь товары по категории «Легковоспламеняющиеся».
Определи по названию, описанию и комплектности, относится ли фактически продаваемый
товар к этой категории. Учитывай только содержимое самого товара или комплекта:
назначение, совместимость и упомянутые отдельно расходники не являются содержимым.

Правила классификации:
1. Сухое горючее в таблетках, кубики, палочки, роллы и растопочные брикеты без настоящих
   спичек не относятся к категории. Обычные дрова, щепа и древесные топливные брикеты
   также не относятся. Если в тот же комплект явно входят настоящие спички или
   зажигалка, товар относится к категории.
2. Сам древесный или каменный уголь, антрацит и древесно-угольные брикеты для гриля
   относятся к категории. Пустой мангал, угольный гриль, стартер или другой аксессуар
   без угля не относятся. Одноразовый мангал с явно вложенным пакетом угля относится.
   Слово «угольный» или «брикет» само по себе наличие нужного товара не доказывает.
3. Заполненный баллон с бутаном, пропаном, изобутаном или MAPP-газом относится к
   категории. Пустой баллон, переходник, клапан, чехол, баллон с CO2 или сжатым воздухом
   не относятся. Жидкий бензин, нефрас и отдельная жидкость для заправки зажигалок по
   границе этой разметки также не относятся.
4. Автономная ручная мини-горелка, зажигалка-горелка или ювелирная горелка со встроенным
   перезаправляемым резервуаром относится к категории, даже если продаётся незаправленной.
   Горелка-насадка, плита, система приготовления, гриль или обогреватель, которым нужен
   отдельный внешний баллон или топливо, не относятся, если этот расходник не входит.
   Пьезоподжиг, регулировка пламени и совместимость с баллоном наличие топлива не доказывают.
5. Настоящие спички и настоящая зажигалка с пламенем относятся к категории. Пустая
   спичечница, чехол, фитиль, кремни, электрический розжиг без пламени, «вечная спичка»
   без топлива, игрушка, шокер или сувенир в форме зажигалки не относятся.
6. Активное пиротехническое изделие с пороховым или пиротехническим зарядом, фитилём,
   петардой, активной чекой или другим инициатором относится к категории. Сюда входят
   цветной дым, дымовые шашки, салюты, фейерверки, бенгальские огни, свечи-фонтаны для
   торта и страйкбольные пиротехнические гранаты. Рисунок, бренд или рекламные слова
   «салют», «огонь» и «фейерверк» без реального пиротехнического изделия не учитывай.
7. Обычная восковая, ароматическая или тортовая свеча с обычным фитилём не относится.
   Настоящая хлопушка с пиротехническим зарядом либо срабатывающая от вытягивания кольца
   или шнура и выбрасывающая конфетти относится. Пневмохлопушка на сжатом воздухе,
   отдельные конфетти, бумажный серпантин и декоративная лента без самой хлопушки не
   относятся.
8. Пустой биокамин не относится. Биокамин относится только когда в его продаваемый
   комплект явно входят и биотопливо, и настоящие спички. Краска Холи, благовония и
   ароматические палочки без пиротехнического заряда не относятся.
9. Если продаётся только аксессуар, корпус, насадка, чехол, держатель, упаковка, декор,
   игрушка, посуда или пустое устройство без перечисленного выше содержимого, товар не
   относится к категории. В наборе проверяй каждый явно перечисленный предмет по тем же
   правилам; одно лишь название набора ничего не доказывает. При противоречии опирайся
   на явный состав и комплектность, а не на рекламное название или способы применения.

На вопрос о принадлежности товара к категории отвечай только одним словом: «Да» или «Нет».
"""

VERBALIZERS = {0: "Нет", 1: "Да"}



# Exact clean LVJ fold-0 membership from the accepted OOF control.  It is
# embedded because sklearn's StratifiedGroupKFold assignment has changed
# across versions.  Static source 2462 belongs to this fold and is therefore
# removed (without replacement) in fold0_sweep mode.
_FOLD0_IDS_B85 = """c-k$OiLvA`2nAD_5Uhhk`6ssB{GBSlk}*QuXx49iUtF<2aIF^4K#K9ia=ky-dxdzGc;(zZlvm(c3Oh&oxzf*b*BzPn557T9x_8zw_&g+*V&b?pN_w=M>DEjMIojl$TCv6QYkyK&#d;6M9}fA+__pf3t4i-73<k`*;uh4wpSY#0kg`81d;MXFch+&lmh=2UiyAai?~Xx*+&`+it~*=qa~-oM^fag4L63+Ux(2QMl#qMYiX2z09G2`F{97v~pX(YGY6SN;=y^Q?x`y!9pzLofct>IH&f0)s`w|Vw{?u%*^F(>TS;x>-YVRd|D`%fS)bf2*VNzz_lx55ZWr@#q%ux|7&QU7Aa}S!4JLkOlR$H-k<*qrR(vPKc2}%x1>2t;5=@`0^-eWA83;+2{pYs`vqcUe{F$sM|Q|YW>Tt@JWm1hsnRKPi3iR+0Pi1F?{8hCWpi8H9PPK>ZHnNNP-p!8e=v)qGcD4GCDDQoc5h^ZW$lggMLa{?H$?^ZUTknzdsQM&I?YCu(b^n6S9H&(yV<&C|j2*AhtX&m^(H<SnaT}r(BbG?nt)%n!B>_@rx82)?oCOCAhpc}Zpo_Zq&YtUnF&aO%Jo5MLNPZ5Ra$HuVxbl`n!P_}1(p|K3wh~iVb_zYfr^@p&B+68h4hhSR8R7`9$FLn;vlwzy_L)58p%*&WekBNSq5qx@hdU#BDP!+DJ9(T^V!x`2Gxi4p!o{^tAq^_LCKuIVp-yh0SHr6OpgjDia`$hDbrPcE%ux+3*fu^AjdDM6w3PVm8%+Ka*)JZ@N$!(ja@V-Ggj>pdGS|u5#Zib%AFs|ShbI|P`omqZMb1dhX^XksJ2GyjkzwA+?O$XYVqNhcfaY%q#pT~kFu0hyprdG}S_OajBbm_8wY9doJg6J4ID0BQ#h9``CjV%>sTcyPVhG<PzI^ins(6zlXh7%2|%&wV9RW_TNz+j}@J&N}pv=D2?RJA^>$>4f=y`~21)vR?6TT?W1+MxCFV=t=`oV9%C582*69P;$bh<#R<;`VQU1b=@RL^BJ3q@kZ5AptTSSOB7F>wK^uZ(1%u1@8U)Kw<5JYTK#+kV4Y~IEvCi@=?l_W{FSx-<Iqq^#S68VD_d8C7H2tq?N|A1Q?1kn+Cs|UJCHq^ZI;M6TJtx!ae)B!xq0EcYta0+h&Hs`$xcO`92F+5-{-Sjl;bJNTH>n=|MBiJ6@3h-xPtzf5$r!qW~Jp|BlV<T(k#VVVQm5c((yy3lGrt<E*0vN~1LB`)o*pf!m-Jp+2V5sqN|t*fRZ|9*<sbEsUqbZ-Li#Lwi^&y{X>gxK}}aa-8k}a-Mp<gv&g;-xA18U<n0WUkMTLfaJk{7iaLKz$R}?SC0TCNW0RA|8-;neDL<fa6tH$Mz!;>i`3(}0A9H^QznCeZcGRGkp##W6ynu!H^}cRTc}ac7jNQ*s~edYd`vDl0I#KOwU1B&+e+zp=6Yeg79s$LF1t+xul~Bv53d9c3{wUjg!*x1IYI(x9afDo=ILK}Q3-Aa!Uixd>^>$y{0DYg0TFl#V3++Sy!3>#KkTBBWENKjSiJK_GskRB;SCOe{v<s-8diI#G~g@N037|um~;f{CHh$i4<OB{z5b<_p2V1>%6}$eI`+^71~MMt0PZ3J<(<TtLEr=X#k<bxyqpb4Qald)rBO@!Y#KU+A{ubFjg$%Tn<>GN_%Z8+dDzxUZ~DT_g!D?E4!3la<-@CKi8B)U>czf(rJ40%f!(9DwTu#l#AhdZk#41R<$ss5mz8904Z{1Cv#lJ<Wq|1AOMnFfDyhv<=~7SIJ$)YVj#V}$u3&s@a<?9i0eF4aSNA{qay?Eql>HH-_D>k#D>k%+)%Kg|^X>ny-lP!#%bv$K$cLJ3>!%q^ZTF@9^LDWSVE0vye?;0EqI-rQjWTKfaJ5D@Hg^62yxy~#n6#a!y==5@>+;|VX>QUX?UZYKPaCOj%x%Yck4*n{bo(aQsma_BW`)A<`XQtd#(t7;&m&cz*Yfuf9X4L!{hrZ${3QX%#OH+^KlAE;v$s)+;^hM+#1MejzWlt20qplS_WV;~CnSfUJNlpL`K&Fkzt2hkQaNNHK$g5B9M1qeM=F2}dn)Sw17bSl;{"""
FOLD0_IDS = frozenset(zlib.decompress(base64.b85decode(_FOLD0_IDS_B85)).decode().split())
FOLD0_IDS_SHA256 = "cfde8133f7539c2459032cbb7614433db48436fba0b558ee8fb20199ce837024"
FOLD0_STATIC_EXCLUDED_ID = "2462"

REQUIRED_COLUMNS = ["id", "name", "description", "category", "label"]
_NON_ALNUM = re.compile(r"[^a-zа-яё0-9]+")


def json_sha(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_hash(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def dry_run() -> bool:
    flag = os.environ.get("LVJ_DORA_FULLTRAIN_DRY_RUN", os.environ.get("LVJ_DORA_RESTRICTED2_DRY_RUN", "")).strip().lower()
    return flag not in {"", "0", "false", "no"} or "--dry-run" in sys.argv


def canonical_id_sha256(values) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


def run_plan() -> dict:
    horizon = float(CONFIG["scheduler_horizon_epochs"])
    checkpoints = tuple(map(float, CONFIG["checkpoint_epochs"]))
    if not math.isclose(horizon, 3.0, abs_tol=1e-15):
        raise AssertionError(f"Full-train horizon must be exactly 3.0 epochs, got {horizon}")
    if checkpoints != (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        raise AssertionError(f"Invalid checkpoint epochs: {checkpoints}")
    return {
        "mode": "fulltrain_data_agreed_rules_system_yesno",
        "train_until_epoch": 3.0,
        "scheduler_horizon_epochs": 3.0,
        "checkpoint_epochs": checkpoints,
        "evaluate_fold0": False,
        "prompt_policy": "data_agreed_rules_in_system_plus_natural_user_question_zero_fewshots",
    }


def active_profile() -> dict:
    name = str(CONFIG["active_profile"])
    if name not in PROFILES:
        raise AssertionError(f"Unknown active_profile={name!r}")
    return {"name": name, **PROFILES[name]}


def validate_config() -> dict:
    profile = active_profile()
    plan = run_plan()
    if profile["name"] != "official_128_r16":
        print(f"WARNING: non-default profile is active: {profile['name']}", flush=True)
    if CONFIG["training_dtype"] != "bfloat16":
        raise AssertionError("The right-padding trainer is BF16-only")
    if not math.isclose(float(CONFIG["positive_class_multiplier"]), 6.0, abs_tol=1e-15):
        raise AssertionError("Full-train run requires positive x6 / negative x1")
    return {
        "profile": profile,
        "target": "exactly one natural answer token: Нет/Да",
        "mode_plan": plan,
        "static_examples": 0,
        "dynamic_examples": 0,
        "retrieval": False,
        "query_representation": "raw name+description",
        "prompt": "system=classification rules; user=current name/description + natural category question",
        "training_dtype": "bfloat16",
        "adapter_master_dtype": "float32",
        "cross_entropy_dtype": "float32 over only the Нет/Да logits",
    }


def resolve_data_path() -> Path:
    for path in map(Path, CONFIG["data_path_candidates"]):
        if path.is_file():
            print(f"data_path={path}", flush=True)
            return path
    raise FileNotFoundError(f"data.csv not found: {CONFIG['data_path_candidates']}")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower().replace("ё", "е")
    return _NON_ALNUM.sub(" ", value).strip()


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def load_clean_frame(path: Path) -> tuple[pd.DataFrame, dict]:
    actual_sha = file_sha(path)
    if actual_sha != CONFIG["expected_data_sha256"]:
        raise AssertionError(f"data.csv SHA drift: {actual_sha}")
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[c for c in frame if c.startswith("Unnamed:")], errors="ignore")
    if set(REQUIRED_COLUMNS) - set(frame):
        raise ValueError(f"Missing columns: {sorted(set(REQUIRED_COLUMNS) - set(frame))}")
    raw_rows = len(frame)
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["id"] = frame["id"].astype(str)
    for column in ("name", "description", "category"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame.category.eq("Легковоспламеняющиеся")].copy()
    source_category_rows = len(frame)
    exact_stats = frame.groupby(["name", "description"], dropna=False).label.agg(["size", "sum"])
    exact_label_conflicts = int(
        (exact_stats["sum"].gt(0) & exact_stats["sum"].lt(exact_stats["size"])).sum()
    )
    frame = frame.drop_duplicates(["name", "description"], keep="first").copy()

    # Leakage grouping is deliberately decoupled from the class-only loss.
    dsu = DisjointSet(len(frame))
    for column in ("name", "description"):
        seen: dict[str, int] = {}
        for position, value in enumerate(frame[column]):
            key = normalize_text(value)
            if not key:
                continue
            if key in seen:
                dsu.union(position, seen[key])
            else:
                seen[key] = position
    members: dict[int, list[str]] = defaultdict(list)
    for position, item_id in enumerate(frame.id):
        members[dsu.find(position)].append(str(item_id))
    group_names = {
        root: hashlib.sha1("\x1f".join(sorted(ids)).encode()).hexdigest()[:16]
        for root, ids in members.items()
    }
    frame["product_group"] = [group_names[dsu.find(position)] for position in range(len(frame))]
    frame = frame.sort_values("id", kind="mergesort").reset_index(drop=True)
    observed = (len(frame), int(frame.label.sum()), int(frame.product_group.nunique()))
    expected = (CONFIG["expected_clean_rows"], CONFIG["expected_positives"], CONFIG["expected_groups"])
    if observed != expected:
        raise AssertionError(f"Exact-deduplicated LVJ contract drift: observed={observed} expected={expected}")

    actual_fold0 = set(frame.loc[frame.id.isin(FOLD0_IDS), "id"])
    if (
        len(FOLD0_IDS) != 739
        or canonical_id_sha256(FOLD0_IDS) != FOLD0_IDS_SHA256
        or actual_fold0 != set(FOLD0_IDS)
    ):
        raise AssertionError("Frozen fold-0 ID contract drifted")
    fold0 = frame.loc[frame.id.isin(FOLD0_IDS)]
    if len(fold0) != 739 or fold0.label.value_counts().to_dict() != {0: 719, 1: 20}:
        raise AssertionError("Frozen fold-0 row/label contract drifted")
    if frame.duplicated(["name", "description"]).any() or not frame.id.is_unique:
        raise AssertionError("LVJ exact duplicates survived or IDs are not unique")
    return frame, {
        "data_csv_sha256": actual_sha,
        "raw_rows": raw_rows,
        "source_lvj_rows": source_category_rows,
        "rows_removed_as_exact_name_description_duplicates": source_category_rows - len(frame),
        "exact_keys_with_conflicting_labels": exact_label_conflicts,
        "deduplication_policy": "drop_duplicates(name,description,keep=first)",
        "product_group_policy": "normalized name OR description; retrieval leakage only; excluded from loss",
        "rows": len(frame),
        "positives": int(frame.label.sum()),
        "product_groups": int(frame.product_group.nunique()),
        "fold0_rows": len(fold0),
        "fold0_positives": int(fold0.label.sum()),
        "fold0_ids_sha256": FOLD0_IDS_SHA256,
    }


def apply_loss_weights(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Use only class weights: negative x1, positive x6."""
    result = frame.copy().reset_index(drop=True)
    result["row_loss_weight"] = np.where(result.label.eq(1), 6.0, 1.0)
    class_counts = result.label.value_counts().sort_index().to_dict()
    observed_positive_share = float(
        result.loc[result.label.eq(1), "row_loss_weight"].sum() / result.row_loss_weight.sum()
    )
    report = {
        "train_rows": len(result),
        "train_positives": int(result.label.sum()),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "row_weight_sum": float(result.row_loss_weight.sum()),
        "positive_weight_mass": float(result.loc[result.label.eq(1), "row_loss_weight"].sum()),
        "nominal_positive_loss_share": observed_positive_share,
        "weighting": "class-only: negative=1.0, positive=6.0; no duplicate/group-size weighting",
    }
    return result, report


















def make_prompt(row) -> str:
    """Natural zero-shot user message. All persistent classification rules live in system."""
    return f"""Название: {row.name}
Описание: {row.description}

Относится ли этот товар к категории «Легковоспламеняющиеся»?"""


def build_prompts(frame: pd.DataFrame, payload_name: str = "fulltrain") -> tuple[pd.DataFrame, dict]:
    frame = frame.copy()
    prompts = []
    for row in tqdm(
        frame.itertuples(index=False), total=len(frame),
        desc=f"{payload_name} zero-shot prompts", unit="product",
    ):
        prompt = make_prompt(row)
        # Hard guard against accidental few-shot leakage.
        forbidden = (
            "Примеры из обучающей выборки",
            "Динамические примеры",
            "Пример 1",
            "similarity=",
            "Сигнал отдельной TF-IDF",
        )
        if any(token in prompt for token in forbidden):
            raise AssertionError("Few-shot/retrieval text leaked into a zero-shot prompt")
        if prompt.count(f"Название: {row.name}") != 1 or prompt.count(f"Описание: {row.description}") != 1:
            raise AssertionError(f"Current product must appear exactly once in prompt id={row.id}")
        prompts.append(prompt)

    frame["prompt"] = prompts
    frame["target_text"] = frame.label.map(VERBALIZERS)
    frame["prompt_sha256"] = frame.prompt.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    frame["target_sha256"] = frame.target_text.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    prompt_sha = hashlib.sha256(
        "\x1e".join(
            f"{item_id}\x1f{prompt}"
            for item_id, prompt in zip(frame.id, frame.prompt, strict=True)
        ).encode()
    ).hexdigest()
    target_sha = hashlib.sha256(
        "\x1e".join(
            f"{item_id}\x1f{target}"
            for item_id, target in zip(frame.id, frame.target_text, strict=True)
        ).encode()
    ).hexdigest()
    return frame, {
        "payload_name": payload_name,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "system_prompt_words": len(SYSTEM_PROMPT.split()),
        "prompt_payload_sha256": prompt_sha,
        "target_payload_sha256": target_sha,
        "static_examples_per_query": 0,
        "dynamic_examples_per_query": 0,
        "retrieval_enabled": False,
        "tfidf_prompt_prior": False,
        "query_representation": "raw name+description",
        "prompt_policy": "rules in system; current product + natural question in user",
    }


def balanced_windows(frame: pd.DataFrame, epoch: int) -> tuple[list[list[int]], pd.DataFrame, dict]:
    effective_batch = int(CONFIG["batch_size"] * CONFIG["gradient_accumulation"])
    steps = math.ceil(len(frame) / effective_batch)
    # Spread the short tail over all optimizer windows.  With uniform weights
    # this keeps every denominator-equivalent window at 15 or 16 rows instead
    # of leaving one pathological 7-row final window.
    base_capacity, extra_capacity = divmod(len(frame), steps)
    capacities = [base_capacity + 1] * extra_capacity + [base_capacity] * (
        steps - extra_capacity
    )
    bins: list[list[int]] = [[] for _ in range(steps)]
    masses = np.zeros(steps, dtype=np.float64)
    target_mass = float(frame.row_loss_weight.sum() / steps)

    def needed_average(slot: int) -> float:
        remaining = capacities[slot] - len(bins[slot])
        return (target_mass - float(masses[slot])) / remaining if remaining else -math.inf

    positives = sorted(
        frame.index[frame.label.eq(1)],
        key=lambda index: (
            -float(frame.at[index, "row_loss_weight"]),
            stable_hash(CONFIG["seed"], epoch, "positive", frame.at[index, "id"]),
        ),
    )
    if len(positives) > steps:
        raise AssertionError("More positive rows than optimizer windows; distinct placement impossible")
    # Place each positive in a distinct window.  The small final-capacity bin
    # gets first access to a high-weight row via the required-average key.
    for index in positives:
        eligible = [slot for slot in range(steps) if not bins[slot]]
        slot = max(
            eligible,
            key=lambda candidate: (
                needed_average(candidate),
                stable_hash(CONFIG["seed"], epoch, "positive-slot", candidate),
            ),
        )
        bins[slot].append(int(index))
        masses[slot] += float(frame.at[index, "row_loss_weight"])

    negatives = sorted(
        frame.index[frame.label.eq(0)],
        key=lambda index: (
            -float(frame.at[index, "row_loss_weight"]),
            stable_hash(CONFIG["seed"], epoch, "negative", frame.at[index, "id"]),
        ),
    )
    # Capacity-aware LPT: give the next largest row to the bin whose remaining
    # slots need the largest average weight to land on the common target mass.
    for index in negatives:
        eligible = [slot for slot in range(steps) if len(bins[slot]) < capacities[slot]]
        if not eligible:
            raise AssertionError("Balanced-window allocator exhausted capacity early")
        slot = max(
            eligible,
            key=lambda candidate: (
                needed_average(candidate),
                target_mass - float(masses[candidate]),
                stable_hash(CONFIG["seed"], epoch, "negative-slot", candidate),
            ),
        )
        bins[slot].append(int(index))
        masses[slot] += float(frame.at[index, "row_loss_weight"])

    # Randomize optimizer-window position and within-window microbatch order
    # without changing the balanced membership.
    slot_order = sorted(range(steps), key=lambda slot: stable_hash(CONFIG["seed"], epoch, "window", slot))
    windows = []
    for slot in slot_order:
        members = sorted(
            bins[slot],
            key=lambda index: stable_hash(CONFIG["seed"], epoch, "member", frame.at[index, "id"]),
        )
        windows.append(members)

    flattened = [index for window in windows for index in window]
    if len(flattened) != len(frame) or len(set(flattened)) != len(frame):
        raise AssertionError(f"Epoch {epoch} balanced windows do not cover every row exactly once")
    window_masses = np.asarray([float(frame.loc[window, "row_loss_weight"].sum()) for window in windows])
    positive_counts = np.asarray([int(frame.loc[window, "label"].sum()) for window in windows])
    # With only 123 positive rows across 201 optimizer windows, exact x6 class
    # weights cannot make every window equal-mass without oversampling.  Keep
    # every row exactly once and accept the bounded 16..21 natural mass range.
    if window_masses.max() / window_masses.min() > 1.35:
        raise AssertionError(f"Epoch {epoch} window mass imbalance: {window_masses.min()}..{window_masses.max()}")
    expected_positives = int(frame.label.sum())
    if int(positive_counts.max()) > 1 or int((positive_counts > 0).sum()) != expected_positives:
        raise AssertionError(f"Epoch {epoch} positives are not evenly distributed")

    records = []
    for window_position, window in enumerate(windows):
        for member_position, index in enumerate(window):
            records.append({
                "epoch": epoch,
                "window": window_position,
                "member_position": member_position,
                "id": str(frame.at[index, "id"]),
                "label": int(frame.at[index, "label"]),
                "row_loss_weight": float(frame.at[index, "row_loss_weight"]),
                "window_mass": float(window_masses[window_position]),
            })
    report = {
        "epoch": epoch,
        "optimizer_windows": len(windows),
        "effective_batch_size": effective_batch,
        "window_mass_min": float(window_masses.min()),
        "window_mass_p50": float(np.quantile(window_masses, .5)),
        "window_mass_max": float(window_masses.max()),
        "window_mass_ratio_max_min": float(window_masses.max() / window_masses.min()),
        "target_window_mass": target_mass,
        "windows_with_positive": int((positive_counts > 0).sum()),
        "max_positives_per_window": int(positive_counts.max()),
        "fixed_denominator": float(frame.row_loss_weight.sum() / len(windows)),
    }
    return windows, pd.DataFrame(records), report


def build_all_windows(frame: pd.DataFrame) -> tuple[dict[int, list[list[int]]], pd.DataFrame, dict]:
    windows, records, reports = {}, [], []
    for epoch in range(1, math.ceil(float(CONFIG["scheduler_horizon_epochs"])) + 1):
        windows[epoch], audit, report = balanced_windows(frame, epoch)
        records.append(audit)
        reports.append(report)
    schedule = pd.concat(records, ignore_index=True)
    schedule_sha = hashlib.sha256(schedule.to_csv(index=False).encode()).hexdigest()
    schedule_key = f"{run_plan()['mode']}:{active_profile()['name']}"
    return windows, schedule, {
        "schedule_key": schedule_key,
        "schedule_sha256": schedule_sha,
        "epochs": reports,
    }


def accumulation_equivalence_preflight(
    frame: pd.DataFrame, windows: dict[int, list[list[int]]]
) -> dict:
    """Prove microbatch accumulation equals the intended fixed-D window objective."""
    steps = len(windows[1])
    denominator = float(frame.row_loss_weight.sum() / steps)
    synthetic_loss = {
        int(index): (int(stable_hash("ga-proof", frame.at[index, "id"])[:12], 16) + 1) / 16**12
        for index in frame.index
    }
    max_abs_error = 0.0
    for epoch_windows in windows.values():
        for window in epoch_windows:
            whole = sum(
                synthetic_loss[index] * float(frame.at[index, "row_loss_weight"])
                for index in window
            ) / denominator
            accumulated = 0.0
            for offset in range(0, len(window), int(CONFIG["batch_size"])):
                accumulated += sum(
                    synthetic_loss[index] * float(frame.at[index, "row_loss_weight"])
                    for index in window[offset:offset + int(CONFIG["batch_size"])]
                ) / denominator
            max_abs_error = max(max_abs_error, abs(whole - accumulated))
    if max_abs_error > 1e-12:
        raise AssertionError(f"Gradient-accumulation objective algebra drift: {max_abs_error}")
    proof_window = next(
        window for window in windows[1]
        if len(window) == int(CONFIG["batch_size"] * CONFIG["gradient_accumulation"])
    )

    def gradient_proof(dtype: torch.dtype) -> float:
        features = torch.tensor([
            [
                int(stable_hash("feature-a", frame.at[index, "id"])[:8], 16) / 16**8,
                int(stable_hash("feature-b", frame.at[index, "id"])[:8], 16) / 16**8,
                1.0,
            ]
            for index in proof_window
        ], dtype=dtype)
        targets = torch.tensor(
            [int(frame.at[index, "label"]) for index in proof_window], dtype=torch.long
        )
        weights = torch.tensor(
            [float(frame.at[index, "row_loss_weight"]) for index in proof_window], dtype=dtype
        )
        initial = torch.tensor(
            [[0.17, -0.11], [-0.07, 0.13], [0.03, -0.02]], dtype=dtype
        )
        whole_parameter = initial.clone().requires_grad_(True)
        whole_losses = torch.nn.functional.cross_entropy(
            features @ whole_parameter, targets, reduction="none"
        )
        ((whole_losses * weights).sum() / denominator).backward()
        whole_gradient = whole_parameter.grad.detach().clone()

        micro_parameter = initial.clone().requires_grad_(True)
        batch_size = int(CONFIG["batch_size"])
        for offset in range(0, len(proof_window), batch_size):
            micro_losses = torch.nn.functional.cross_entropy(
                features[offset:offset + batch_size] @ micro_parameter,
                targets[offset:offset + batch_size],
                reduction="none",
            )
            (
                (micro_losses * weights[offset:offset + batch_size]).sum() / denominator
            ).backward()
        return float((whole_gradient - micro_parameter.grad).abs().max())

    fp64_gradient_error = gradient_proof(torch.float64)
    fp32_gradient_error = gradient_proof(torch.float32)
    if fp64_gradient_error > 1e-10 or fp32_gradient_error > 1e-6:
        raise AssertionError(
            f"True autograd accumulation proof failed: fp64={fp64_gradient_error} fp32={fp32_gradient_error}"
        )
    return {
        "checked_windows": int(sum(map(len, windows.values()))),
        "fixed_denominator": denominator,
        "max_abs_error_microbatch_sum_vs_whole_window": max_abs_error,
        "autograd_proof_window_rows": len(proof_window),
        "autograd_fp64_max_gradient_error": fp64_gradient_error,
        "autograd_fp32_max_gradient_error": fp32_gradient_error,
        "autograd_thresholds": {"float64": 1e-10, "float32": 1e-6},
        "passed": True,
    }


def prepare_contracts(
    root: Path,
) -> tuple[pd.DataFrame, None, dict[int, list[list[int]]], dict]:
    config_report = validate_config()
    plan = run_plan()
    all_frame, data_report = load_clean_frame(resolve_data_path())

    # TRUE FULL-TRAIN: every exact-deduplicated LVJ row is used for optimization.
    train = all_frame.copy().reset_index(drop=True)
    if (len(train), int(train.label.sum())) != (CONFIG["expected_clean_rows"], CONFIG["expected_positives"]):
        raise AssertionError(
            f"Full-train cardinality drift: rows={len(train)} positives={int(train.label.sum())}"
        )
    train, weight_report = apply_loss_weights(train)
    train, train_prompt_report = build_prompts(train, "fulltrain")

    # Explicitly prove the prompt has zero examples for every training row.
    forbidden = ("Примеры из обучающей выборки", "Динамические примеры", "Пример 1", "similarity=")
    if train.prompt.map(lambda p: any(token in p for token in forbidden)).any():
        raise AssertionError("A few-shot marker survived in full-train prompts")

    windows, schedule_frame, schedule_report = build_all_windows(train)
    accumulation_report = accumulation_equivalence_preflight(train, windows)
    root.mkdir(parents=True, exist_ok=False)

    train[[
        "id", "label", "product_group", "row_loss_weight", "prompt_sha256", "target_sha256",
    ]].to_csv(root / "train_manifest.csv", index=False)
    schedule_frame.to_csv(root / "balanced_optimizer_schedule.csv", index=False)
    train[["id", "prompt", "target_text"]].to_json(
        root / "training_payload.jsonl.gz", orient="records", lines=True,
        force_ascii=False, compression="gzip",
    )

    contract = {
        "experiment_version": CONFIG["experiment_version"],
        "config": config_report,
        "data": data_report,
        "training_pool": weight_report,
        "prompt": train_prompt_report,
        "fulltrain": {
            "enabled": True,
            "validation_rows_held_out": 0,
            "rows_used_per_epoch": len(train),
            "positives_used_per_epoch": int(train.label.sum()),
            "all_clean_rows_used": True,
        },
        "objective": {
            "prompt_tokens_masked": True,
            "active_tokens_per_row": 1,
            "active_tokens": ["natural answer token Нет/Да"],
            "allowed_softmax_tokens": ["Нет", "Да"],
            "all_other_vocabulary_logits_excluded_before_softmax": True,
            "row_weight_formula": "class-only weights: negative 1.0, positive 6.0",
            "positive_multiplier": CONFIG["positive_class_multiplier"],
            "fixed_optimizer_denominator": "sum(row_loss_weight) / optimizer_steps_per_epoch",
            "cross_entropy_dtype": "float32",
            "gradient_accumulation_equivalence_preflight": accumulation_report,
        },
        "training": {
            "all_ids_once_per_epoch": True,
            "rows_per_epoch": len(train),
            "positives_per_epoch": int(train.label.sum()),
            "train_until_epoch": plan["train_until_epoch"],
            "scheduler_horizon_epochs": plan["scheduler_horizon_epochs"],
            "checkpoint_epochs": list(plan["checkpoint_epochs"]),
            "batch_size": CONFIG["batch_size"],
            "gradient_accumulation": CONFIG["gradient_accumulation"],
            "learning_rate": CONFIG["learning_rate"],
            "scheduler": "cosine_to_zero",
            "warmup_ratio": CONFIG["warmup_ratio"],
            "weight_decay": CONFIG["weight_decay"],
            "max_grad_norm": CONFIG["max_grad_norm"],
            "gradient_checkpointing": True,
            "seed": CONFIG["seed"],
        },
        "schedule": schedule_report,
    }
    contract["contract_sha256"] = json_sha(contract)
    write_json(root / "contract_manifest.json", contract)
    print("\n===== LVJ DORA FULLTRAIN ZERO-FEWSHOT CONTRACT =====", flush=True)
    print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)
    return train, None, windows, contract


def render_chat(processor, prompt: str) -> str:
    return processor.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def label_token_ids(tokenizer) -> dict[int, int]:
    """Resolve the two single-token natural verbalizers in class order: Нет=0, Да=1."""
    result: dict[int, int] = {}
    for label, value in VERBALIZERS.items():
        ids = list(map(int, tokenizer(value, add_special_tokens=False)["input_ids"]))
        if len(ids) != 1:
            raise AssertionError(
                f"Natural answer {value!r} for label={label} must be exactly one tokenizer token, got {ids}"
            )
        result[int(label)] = ids[0]
    if len(set(result.values())) != 2:
        raise AssertionError(f"Нет and Да must be two distinct token IDs: {result}")
    if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) in result.values():
        raise AssertionError(f"An answer token unexpectedly equals EOS: {result}")
    return result


class LabelValueDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, processor):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.allowed_token_ids = label_token_ids(self.tokenizer)
        self.items, lengths, token_by_label = [], [], defaultdict(set)
        for row in tqdm(self.frame.itertuples(index=False), total=len(self.frame), desc="tokenize Нет/Да", unit="product"):
            chat = render_chat(processor, str(row.prompt))
            prompt_ids = list(map(int, self.tokenizer(chat, add_special_tokens=False)["input_ids"]))
            combined = list(map(int, self.tokenizer(chat + str(row.target_text), add_special_tokens=False)["input_ids"]))
            value_token_id = self.allowed_token_ids[int(row.label)]
            if combined != prompt_ids + [value_token_id]:
                raise AssertionError(f"Tokenizer boundary merge for id={row.id}")
            if not prompt_ids:
                raise AssertionError(f"Empty causal prompt for id={row.id}")
            token_by_label[int(row.label)].add(value_token_id)
            self.items.append({
                "id": str(row.id), "input_ids": prompt_ids,
                "gold_class_index": int(row.label),
                "row_loss_weight": float(row.row_loss_weight),
            })
            lengths.append(len(prompt_ids))
        if set(token_by_label) != {0, 1} or any(len(tokens) != 1 for tokens in token_by_label.values()):
            raise AssertionError(f"Label verbalizer token IDs are not stable: {dict(token_by_label)}")
        if token_by_label[0] == token_by_label[1]:
            raise AssertionError("Natural answers Нет and Да map to the same token")
        values = np.asarray(lengths)
        self.length_report = {
            "min": int(values.min()), "p50": float(np.quantile(values, .5)),
            "p95": float(np.quantile(values, .95)), "max": int(values.max()),
            "allowed_token_ids_in_class_order": [
                self.allowed_token_ids[0], self.allowed_token_ids[1]
            ],
            "label_token_ids": {str(label): next(iter(tokens)) for label, tokens in token_by_label.items()},
            "active_tokens_per_row": 1,
            "causal_shift_equivalence": (
                "the selected predecessor position is the final prompt token; its target is the natural answer token"
            ),
            "loss_softmax_domain": "exactly [token('Нет'), token('Да')]; all other logits excluded",
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[int(index)]


def collate(items: list[dict], pad_token_id: int) -> dict:
    length = int(math.ceil(max(len(item["input_ids"]) for item in items) / 8) * 8)
    def padded(key, fill):
        return [item[key] + [fill] * (length - len(item[key])) for item in items]
    return {
        "ids": [item["id"] for item in items],
        "input_ids": torch.tensor(padded("input_ids", pad_token_id), dtype=torch.long),
        "attention_mask": torch.tensor([
            [1] * len(item["input_ids"]) + [0] * (length - len(item["input_ids"])) for item in items
        ], dtype=torch.long),
        "gold_class_indices": torch.tensor(
            [item["gold_class_index"] for item in items], dtype=torch.long
        ),
        "row_loss_weights": torch.tensor([item["row_loss_weight"] for item in items], dtype=torch.float32),
    }


def masked_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Explicit logical Qwen3.5 positions for either padded batch direction."""
    positions = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
    return positions.masked_fill(attention_mask.eq(0), 0)




def sparse_label_logits(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[object, torch.Tensor, dict]:
    """Gather each row's final-prompt logits that predict its sole label token."""
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise AssertionError("Sparse label inputs require matching [batch, sequence] tensors")
    real_lengths = attention_mask.to(dtype=torch.long).sum(dim=-1)
    if (real_lengths < 1).any():
        raise AssertionError("Every row needs a nonempty prompt")
    selected_positions = real_lengths - 1
    unique_positions, inverse = torch.unique(
        selected_positions, sorted=True, return_inverse=True
    )
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=masked_position_ids(attention_mask),
        use_cache=False,
        logits_to_keep=unique_positions,
    )
    expected_prefix = (int(input_ids.shape[0]), int(unique_positions.numel()))
    if outputs.logits.ndim != 3 or tuple(outputs.logits.shape[:2]) != expected_prefix:
        raise AssertionError(
            f"Qwen tensor logits_to_keep contract drifted: {tuple(outputs.logits.shape)}; "
            f"expected prefix {expected_prefix}"
        )
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    logits = outputs.logits[rows, inverse, :]
    if logits.ndim != 2 or logits.shape[0] != input_ids.shape[0]:
        raise AssertionError(f"Sparse label gather produced invalid shape {tuple(logits.shape)}")
    return outputs, logits, {
        "selected_positions": selected_positions.detach().cpu().tolist(),
        "unique_positions": unique_positions.detach().cpu().tolist(),
        "lm_head_positions_computed": int(unique_positions.numel()),
    }




def right_padding_sparse_logits_preflight(model, dataset: LabelValueDataset, pad_token_id: int) -> dict:
    """Hard-gate the one-position right-pad gather and restricted logit slice."""
    lengths = np.asarray([len(item["input_ids"]) for item in dataset.items])
    short_index, long_index = int(lengths.argmin()), int(lengths.argmax())
    if lengths[short_index] == lengths[long_index]:
        raise AssertionError("Padding preflight needs two different sequence lengths")
    items = [dataset[short_index], dataset[long_index]]
    batch = collate(items, pad_token_id)
    batch_mask = batch["attention_mask"]
    if bool((batch_mask[:, 1:] > batch_mask[:, :-1]).any().item()):
        raise AssertionError("Training collate is not contiguous right padding")
    expected = [int(length) - 1 for length in batch_mask.sum(dim=-1).tolist()]
    allowed = list(map(int, dataset.length_report["allowed_token_ids_in_class_order"]))
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = batch["input_ids"].to(model.device)
            attention = batch["attention_mask"].to(model.device)
            _, full_logits, report = sparse_label_logits(model, input_ids, attention)
            restricted = full_logits[:, allowed].float()
        if report["selected_positions"] != expected:
            raise AssertionError(
                f"Sparse label positions drifted: {report['selected_positions']} != {expected}"
            )
        if tuple(restricted.shape) != (2, 2) or not torch.isfinite(restricted).all():
            raise AssertionError(f"Invalid restricted-logit preflight tensor {tuple(restricted.shape)}")
        return {
            "short_length": int(lengths[short_index]),
            "long_length": int(lengths[long_index]),
            "padded_length": int(batch["input_ids"].shape[1]),
            "padding_direction": "right",
            "selected_causal_positions": "final prompt token",
            "target_sequence": "exactly one gold Нет/Да token",
            "allowed_token_ids_in_class_order": allowed,
            "restricted_logit_shape": list(restricted.shape),
            "loss_softmax_width": 2,
            "full_vocabulary_logits_excluded_before_log_softmax": True,
            "passed": True,
        }
    finally:
        model.train(was_training)
        torch.cuda.empty_cache()


def resolve_model_class(transformers_module):
    for name in ("AutoModelForMultimodalLM", "Qwen3_5ForConditionalGeneration", "AutoModelForImageTextToText"):
        cls = getattr(transformers_module, name, None)
        if cls is not None:
            return cls
    raise AttributeError("No Qwen3.5 multimodal model class is available")


def ensure_transformers(model_id: str):
    try:
        torchao = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao = None
    if torchao is not None and tuple(int(x) for x in re.findall(r"\d+", torchao)[:3]) < (0, 16, 0):
        if any(name == "transformers" or name.startswith("transformers.") or name == "torchao" or name.startswith("torchao.") for name in sys.modules):
            raise RuntimeError("Old torchao/Transformers already imported; restart Kaggle once")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
        importlib.invalidate_caches()
    required = {"transformers": "5.14.1", "safetensors": "0.8.0"}
    installed = {}
    for package in required:
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed[package] = None
    if installed != required:
        wheels = next((
            path for path in (
                Path("/kaggle/input/datasets/m1ck0n/qwen35-runtime-wheels"),
                Path("/kaggle/input/qwen35-runtime-wheels"), Path(model_id) / "wheels",
            ) if path.is_dir() and any(path.glob("*.whl"))
        ), None)
        if any(name == "transformers" or name.startswith("transformers.") for name in sys.modules):
            raise RuntimeError("Transformers already imported before runtime pin; restart Kaggle once")
        command = [sys.executable, "-m", "pip", "install"]
        if wheels is not None:
            command += ["--find-links", str(wheels)]
        command += [
            "transformers==5.14.1", "safetensors==0.8.0", "accelerate", "sentencepiece",
        ]
        subprocess.check_call(command)
        importlib.invalidate_caches()
    module = importlib.import_module("transformers")
    local_only = Path(model_id).expanduser().is_dir()
    module.AutoConfig.from_pretrained(model_id, local_files_only=local_only)
    module.AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
    resolve_model_class(module)
    print(f"Using Transformers {module.__version__}, Safetensors {importlib.metadata.version('safetensors')}", flush=True)
    return module


def ensure_peft():
    target = CONFIG["peft_version"]
    try:
        current = importlib.metadata.version("peft")
    except importlib.metadata.PackageNotFoundError:
        current = None
    if current != target:
        if "peft" in sys.modules:
            raise RuntimeError(f"PEFT {current} already imported; restart before pinning {target}")
        wheel_roots = [
            Path(os.environ["LVJ_WHEEL_DIR"]) if os.environ.get("LVJ_WHEEL_DIR") else None,
            Path("/kaggle/input/datasets/m1ck0n/qwen35-runtime-wheels"),
            Path("/kaggle/input/qwen35-runtime-wheels"),
            Path(CONFIG["model_id"]) / "wheels" if Path(CONFIG["model_id"]).is_dir() else None,
        ]
        wheels = sorted(
            wheel
            for root in wheel_roots if root is not None and root.is_dir()
            for wheel in root.rglob(f"peft-{target}-*.whl")
        )
        command = (
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheels[0])]
            if wheels else [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", f"peft=={target}"]
        )
        subprocess.check_call(command)
        importlib.invalidate_caches()
    if importlib.metadata.version("peft") != target:
        raise RuntimeError("PEFT pin failed")
    return importlib.import_module("peft")


def text_decoder_linear_names(model) -> list[str]:
    suffixes = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    }
    names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", 1)[-1] in suffixes
        and "vision" not in name.lower() and "visual" not in name.lower() and "lm_head" not in name
    ]
    explicit = [name for name in names if "language_model" in name or "text_model" in name]
    return explicit or [name for name in names if re.search(r"(?:^|\.)layers\.\d+\.", name)]


def select_lora_targets(model) -> tuple[list[str], dict]:
    profile = active_profile()
    names = text_decoder_linear_names(model)
    by_layer, found = defaultdict(list), defaultdict(set)
    for name in names:
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if match:
            layer = int(match.group(1))
            by_layer[layer].append(name)
            found[layer].add(name.rsplit(".", 1)[-1])
    if sorted(found) != list(range(32)):
        raise AssertionError(f"Expected decoder layers 0..31, got {sorted(found)}")
    standard = {"q_proj", "k_proj", "v_proj", "o_proj"}
    delta = {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
    mlp = {"gate_proj", "up_proj", "down_proj"}
    standard_layers, delta_layers = [], []
    for layer in range(32):
        if not mlp <= found[layer]:
            raise AssertionError(f"Layer {layer} misses MLP projections")
        if standard <= found[layer]:
            standard_layers.append(layer)
        elif delta <= found[layer]:
            delta_layers.append(layer)
        else:
            raise AssertionError(f"Layer {layer} has unknown attention family: {sorted(found[layer])}")
    expected_standard_layers = [3, 7, 11, 15, 19, 23, 27, 31]
    if standard_layers != expected_standard_layers or len(delta_layers) != 24:
        raise AssertionError(f"Expected 8 full-attention + 24 DeltaNet layers, got {standard_layers}/{delta_layers}")

    allowed = standard | mlp if profile["target_family"] == "official_attention_plus_mlp" else standard | delta | mlp
    selected = sorted(name for name in names if name.rsplit(".", 1)[-1] in allowed)
    if len(selected) != profile["expected_modules"]:
        raise AssertionError(f"Profile {profile['name']} selected {len(selected)} modules, expected {profile['expected_modules']}")
    if profile["target_family"] == "official_attention_plus_mlp" and any(
        name.rsplit(".", 1)[-1] in delta for name in selected
    ):
        raise AssertionError("Official profile must leave all DeltaNet projections frozen")
    report = {
        "profile": profile,
        "target_modules": selected,
        "target_module_count": len(selected),
        "full_attention_layers": standard_layers,
        "delta_net_layers": delta_layers,
        "vision_frozen": True,
        "lm_head_frozen": True,
    }
    return selected, report


def precision_preflight(model) -> dict:
    frozen, trainable = defaultdict(int), defaultdict(int)
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        dtype_name = str(parameter.dtype).removeprefix("torch.")
        (trainable if parameter.requires_grad else frozen)[dtype_name] += int(parameter.numel())
        if parameter.requires_grad and parameter.dtype != torch.float32:
            raise AssertionError(f"DoRA master parameter is not FP32: {name} {parameter.dtype}")
        if not parameter.requires_grad and parameter.dtype == torch.float16:
            raise AssertionError(f"Hidden FP16 frozen parameter: {name}")
    if frozen.get("bfloat16", 0) <= 0 or trainable.get("float32", 0) <= 0:
        raise AssertionError(f"Precision contract failed: frozen={dict(frozen)} trainable={dict(trainable)}")
    return {
        "base_and_forward_dtype": "bfloat16",
        "adapter_master_dtype": "float32",
        "cross_entropy_dtype": "float32",
        "gradient_scaler_enabled": False,
        "frozen_parameter_dtype_counts": dict(sorted(frozen.items())),
        "trainable_parameter_dtype_counts": dict(sorted(trainable.items())),
        "device_name": torch.cuda.get_device_name(0),
        "cuda_capability": list(map(int, torch.cuda.get_device_capability(0))),
    }


def load_model(transformers_module, root: Path):
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected CUDA device does not support BF16")
    peft = ensure_peft()
    model = resolve_model_class(transformers_module).from_pretrained(
        CONFIG["model_id"], local_files_only=Path(CONFIG["model_id"]).expanduser().is_dir(), dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    targets, target_report = select_lora_targets(model)
    profile = active_profile()
    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=profile["rank"], lora_alpha=profile["alpha"], lora_dropout=profile["dropout"],
        target_modules=targets, bias="none", use_rslora=False, use_dora=True,
        init_lora_weights=True,
    )
    model = peft.get_peft_model(model, lora_config)
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or any(
        "lora_" not in name or "vision" in name.lower() or "visual" in name.lower() or "lm_head" in name
        for name in trainable_names
    ):
        raise AssertionError("Only text-decoder DoRA parameters may be trainable")
    magnitude_names = [name for name in trainable_names if "lora_magnitude_vector" in name]
    matrix_names = [name for name in trainable_names if "lora_A" in name or "lora_B" in name]
    if len(magnitude_names) != len(targets) or len(matrix_names) != 2 * len(targets):
        raise AssertionError(
            "DoRA trainable tensor contract failed: "
            f"magnitude={len(magnitude_names)} matrices={len(matrix_names)} targets={len(targets)}"
        )
    target_report.update({
        "peft_version": getattr(peft, "__version__", "unknown"),
        "trainable_parameter_names": trainable_names,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "adapter_method": "DoRA",
        "use_dora": True,
        "use_rslora": False,
        "magnitude_vector_tensor_count": len(magnitude_names),
        "lora_matrix_tensor_count": len(matrix_names),
    })
    standard_lora_trainable = 21_233_664 if profile["name"] == "official_128_r16" else 16_232_448
    if target_report["trainable_parameters"] <= standard_lora_trainable:
        raise AssertionError(
            "DoRA must add magnitude parameters beyond standard LoRA: "
            f"{target_report['trainable_parameters']} <= {standard_lora_trainable}"
        )
    write_json(root / "dora_target_manifest.json", target_report)
    precision = precision_preflight(model)
    write_json(root / "precision_manifest.json", precision)
    print(
        f"DoRA profile={profile['name']} modules={len(targets)} "
        f"trainable={target_report['trainable_parameters']:,}", flush=True,
    )
    return model, target_report, precision


def cosine_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def binary_metrics(truth, prediction, sample_weight=None) -> dict:
    truth = np.asarray(truth, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    weights = np.ones(len(truth), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != truth.shape or not np.isfinite(weights).all() or (weights <= 0).any():
        raise AssertionError("Invalid binary-metric sample weights")
    tp = float(weights[(truth == 1) & (prediction == 1)].sum())
    fp = float(weights[(truth == 0) & (prediction == 1)].sum())
    fn = float(weights[(truth == 1) & (prediction == 0)].sum())
    tn = float(weights[(truth == 0) & (prediction == 0)].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "weighted_errors": fp + fn,
    }


def weighted_average_precision(truth, scores, sample_weight) -> float:
    truth = np.asarray(truth, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    truth, scores, weights = truth[order], scores[order], weights[order]
    total_positive = float(weights[truth == 1].sum())
    if total_positive <= 0:
        return 0.0
    cumulative_tp = np.cumsum(weights * (truth == 1))
    cumulative_total = np.cumsum(weights)
    group_ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    previous_recall, average_precision = 0.0, 0.0
    for end in group_ends:
        recall = float(cumulative_tp[end] / total_positive)
        precision = float(cumulative_tp[end] / cumulative_total[end])
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision




def parse_yes_no_response(text: str) -> dict:
    nonempty = [line.strip() for line in str(text).splitlines() if line.strip()]
    first = nonempty[0] if nonempty else ""
    inverse = {value: label for label, value in VERBALIZERS.items()}
    if first not in inverse or len(nonempty) != 1:
        raise RuntimeError(f"Constrained Да/Нет generation contract failed without fallback: {text!r}")
    return {
        "predicted_label": int(inverse[first]),
        "parse_ok": 1,
        "strict_one_line": 1,
        "first_line": first,
        "nonempty_lines": len(nonempty),
    }


class BinaryLabelLogitsProcessor:
    """Hard-mask the sole generation step to the two label token IDs."""
    def __init__(self, token0: int, token1: int):
        self.answer_ids = (int(token0), int(token1))

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        masked = torch.full_like(scores, -torch.inf)
        masked[:, list(self.answer_ids)] = scores[:, list(self.answer_ids)]
        return masked


def generate_chat_batch(processor, model, chats: list[str]) -> list[dict]:
    if not chats:
        return []
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    try:
        inputs = processor(text=chats, padding=True, return_tensors="pt").to(model.device)
        input_length = int(inputs["input_ids"].shape[1])
        allowed = label_token_ids(tokenizer)
        logits_processor = BinaryLabelLogitsProcessor(allowed[0], allowed[1])
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                logits_processor=[logits_processor],
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        generated = output[:, input_length:]
        if generated.shape[1] != 1:
            raise RuntimeError(f"Expected exactly one label token, got shape {tuple(generated.shape)}")
        valid_answers = generated[:, 0].eq(allowed[0]) | generated[:, 0].eq(allowed[1])
        if not bool(valid_answers.all().item()):
            raise RuntimeError("Hard-constrained generator emitted a token outside {Нет, Да}")
        raw = processor.batch_decode(generated, skip_special_tokens=False)
        clean = processor.batch_decode(generated, skip_special_tokens=True)
        return [
            {
                "raw_response": raw_item,
                "clean_response": clean_item,
                "generated_tokens": 1,
                "generated_answer_tokens": 1,
                "seconds": float(elapsed / len(chats)),
            }
            for position, (raw_item, clean_item) in enumerate(zip(raw, clean, strict=True))
        ]
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(chats) == 1:
            raise
        middle = len(chats) // 2
        return generate_chat_batch(processor, model, chats[:middle]) + generate_chat_batch(
            processor, model, chats[middle:]
        )


def score_label_value_batch(
    processor, model, chats: list[str], truth_labels: list[int]
) -> list[dict]:
    """Score the sole label token using the same restricted binary objective as training."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    allowed = label_token_ids(tokenizer)
    allowed_ids = [allowed[0], allowed[1]]
    sequences = []
    for chat in chats:
        prompt_ids = list(map(int, tokenizer(chat, add_special_tokens=False)["input_ids"]))
        for label, answer in VERBALIZERS.items():
            combined = list(map(int, tokenizer(chat + answer, add_special_tokens=False)["input_ids"]))
            if combined != prompt_ids + [allowed[int(label)]]:
                raise AssertionError("Validation Да/Нет scorer hit a tokenizer boundary merge")
        sequences.append(prompt_ids)
    max_length = max(map(len, sequences))
    padded = [
        sequence + [int(tokenizer.pad_token_id)] * (max_length - len(sequence))
        for sequence in sequences
    ]
    attention = [
        [1] * len(sequence) + [0] * (max_length - len(sequence))
        for sequence in sequences
    ]
    input_ids = torch.tensor(padded, dtype=torch.long, device=model.device)
    attention_mask = torch.tensor(attention, dtype=torch.long, device=model.device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs, full_logits, _ = sparse_label_logits(model, input_ids, attention_mask)
    restricted_logits = full_logits[:, allowed_ids].float()
    probabilities = torch.softmax(restricted_logits, dim=-1)[:, 1]
    gold_classes = torch.tensor(
        [int(label) for label in truth_labels],
        dtype=torch.long,
        device=model.device,
    )
    restricted_ce = torch.nn.functional.cross_entropy(
        restricted_logits, gold_classes, reduction="none"
    )
    clipped = probabilities.clamp(1e-7, 1 - 1e-7)
    truth = torch.tensor(truth_labels, dtype=torch.float32, device=model.device)
    binary_ce = -(truth * clipped.log() + (1 - truth) * (1 - clipped).log())
    records = [
        {
            "forced_p1": float(probabilities[index].detach().cpu()),
            "forced_binary_no_yes_nll": float(binary_ce[index].detach().cpu()),
            "gold_label_restricted2_ce": float(restricted_ce[index].detach().cpu()),
        }
        for index in range(len(chats))
    ]
    del outputs, full_logits, restricted_logits, probabilities, gold_classes, restricted_ce
    del binary_ce, truth, input_ids, attention_mask
    return records


def evaluate_fold0_checkpoint(
    model, processor, validation: pd.DataFrame, milestone: float, root: Path, contract_sha: str,
) -> dict:
    """Optional fold-0 evaluator (not used in full-train mode)."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    records = []
    ordered = validation.sort_values("id", kind="mergesort").reset_index(drop=True)
    progress = tqdm(total=len(ordered), desc=f"fold0 eval epoch={milestone:g}", unit="product")
    try:
        rows = list(ordered.itertuples(index=False))
        for start in range(0, len(rows), int(CONFIG["generation_batch_size"])):
            batch_rows = rows[start:start + int(CONFIG["generation_batch_size"])]
            chats = [render_chat(processor, str(row.prompt)) for row in batch_rows]
            generations = generate_chat_batch(processor, model, chats)
            forced_scores = score_label_value_batch(
                processor, model, chats, [int(row.label) for row in batch_rows]
            )
            for row, generation, scores in zip(batch_rows, generations, forced_scores, strict=True):
                parsed = parse_yes_no_response(generation["clean_response"])
                records.append({
                    "id": str(row.id), "label": int(row.label),
                    "checkpoint_epoch": float(milestone), "contract_sha256": contract_sha,
                    **scores, "forced_label": int(scores["forced_p1"] >= .5),
                    **parsed, **generation,
                })
            progress.update(len(batch_rows))
    finally:
        progress.close()
        tokenizer.padding_side = previous_padding_side
        model.train()
        torch.cuda.empty_cache()

    predictions = pd.DataFrame(records)
    if len(predictions) != 739 or predictions.id.nunique() != 739:
        raise AssertionError("Fold-0 prediction cardinality drifted")
    score_columns = [
        "forced_p1", "forced_binary_no_yes_nll", "gold_label_restricted2_ce",
    ]
    if not np.isfinite(predictions[score_columns].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Non-finite fold0 forced-answer scores")
    checkpoint_dir = root / "checkpoints" / f"epoch_{milestone:05.3f}".replace(".", "p")
    predictions.to_csv(checkpoint_dir / "fold0_predictions.csv", index=False)

    parse_failures = int(predictions.parse_ok.eq(0).sum())
    generated_metrics = binary_metrics(predictions.label, predictions.predicted_label)
    forced_metrics = binary_metrics(predictions.label, predictions.forced_label)
    class_weights = np.where(predictions.label.to_numpy(dtype=int) == 1, 6.0, 1.0)
    metrics = {
        "checkpoint_epoch": float(milestone),
        "rows": len(predictions),
        "positives": int(predictions.label.sum()),
        "parse_failures": parse_failures,
        "strict_one_line_rate": float(predictions.strict_one_line.mean()),
        "generated": generated_metrics,
        "forced_answer": {
            "at_0p5": forced_metrics,
            "average_precision": weighted_average_precision(
                predictions.label, predictions.forced_p1, np.ones(len(predictions))
            ),
            "mean_binary_no_yes_nll": float(predictions.forced_binary_no_yes_nll.mean()),
            "class6_weighted_restricted2_label_ce": float(np.average(
                predictions.gold_label_restricted2_ce.to_numpy(dtype=float), weights=class_weights
            )),
            "greedy_forced_consistency_rate": float(
                predictions.predicted_label.eq(predictions.forced_label).mean()
            ),
        },
        "selection_eligibility": {
            "eligible": parse_failures == 0,
            "failed_gates": [] if parse_failures == 0 else ["parse_failures"],
            "policy": "hard-constrained single-token Да/Нет generation; no fallback",
        },
        "prompt_type": "rules_in_system_plus_natural_user_question_zero_fewshots",
    }
    write_json(checkpoint_dir / "fold0_metrics.json", metrics)
    generated = metrics["generated"]
    forced = metrics["forced_answer"]
    print(
        "CHECKPOINT_SCORE "
        f"epoch={milestone:.2f} "
        f"F1={generated['f1']:.6f} "
        f"P={generated['precision']:.6f} "
        f"R={generated['recall']:.6f} "
        f"TN={generated['tn']} FP={generated['fp']} "
        f"FN={generated['fn']} TP={generated['tp']} "
        f"forced_F1={forced['at_0p5']['f1']:.6f} "
        f"AP={forced['average_precision']:.6f} "
        f"NLL={forced['mean_binary_no_yes_nll']:.6f} "
        f"restricted2_CE={forced['class6_weighted_restricted2_label_ce']:.6f} "
        f"parse_failures={metrics['parse_failures']} "
        f"eligible={metrics['selection_eligibility']['eligible']}",
        flush=True,
    )
    return metrics




def select_fold0_checkpoint(ranking: list[dict]) -> dict:
    """Select one eligible epoch using fixed noise bands, not decimal chasing."""
    eligible = [item for item in ranking if bool(item["selection_eligible"])]
    if not eligible:
        raise RuntimeError("No fold0 checkpoint passed the pre-registered promotion gates")

    row_band = float(CONFIG["selection_row_f1_band"])
    ap_band = float(CONFIG["selection_ap_band"])
    best_row = max(item["row_f1"] for item in eligible)
    row_candidates = [item for item in eligible if item["row_f1"] >= best_row - row_band]
    best_ap = max(item["weighted_average_precision"] for item in row_candidates)
    ap_candidates = [
        item for item in row_candidates
        if item["weighted_average_precision"] >= best_ap - ap_band
    ]
    # Earlier epoch is the explicit final robustness preference. Objective-matched
    # restricted binary CE is only a duplicate-epoch tie breaker.
    selected = min(
        ap_candidates,
        key=lambda item: (item["checkpoint_epoch"], item["objective_matched_restricted2_ce"]),
    )
    return {
        "selected_profile": str(selected["profile"]),
        "selected_epoch": float(selected["checkpoint_epoch"]),
        "selected_checkpoint_metrics": selected,
        "policy": {
            "eligible_only": True,
            "stage_1": f"within {row_band:.3f} of best validation F1",
            "stage_2": f"within {ap_band:.3f} of best AP among stage-1 candidates",
            "final": "earliest epoch; objective-matched restricted-2 CE only breaks duplicate-epoch ties",
        },
        "best_values": {
            "row_f1": float(best_row),
            "average_precision_after_f1_band": float(best_ap),
        },
        "candidate_epochs": {
            "eligible": [float(item["checkpoint_epoch"]) for item in eligible],
            "after_row_band": [float(item["checkpoint_epoch"]) for item in row_candidates],
            "after_ap_band": [float(item["checkpoint_epoch"]) for item in ap_candidates],
        },
    }




def verify_saved_adapter_loadability(adapter: Path) -> dict:
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise AssertionError(f"Missing PEFT adapter config: {config_path}")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise AssertionError("Saved adapter_config is not a LoRA adapter")
    if adapter_config.get("use_dora") is not True or adapter_config.get("use_rslora") is not False:
        raise AssertionError("Saved adapter is not the required DoRA (use_dora=true, use_rslora=false)")
    weight_files = sorted(adapter.glob("*.safetensors"))
    if len(weight_files) != 1 or weight_files[0].stat().st_size <= 0:
        raise AssertionError("Expected one non-empty safetensors adapter weight file")
    from safetensors import safe_open
    decoded_elements = 0
    tensor_shapes = {}
    with safe_open(str(weight_files[0]), framework="pt", device="cpu") as handle:
        tensor_keys = list(handle.keys())
        for key in tensor_keys:
            tensor = handle.get_tensor(key)
            if tensor.numel() <= 0 or not bool(torch.isfinite(tensor).all()):
                raise AssertionError(f"Invalid saved adapter tensor payload: {key}")
            decoded_elements += int(tensor.numel())
            tensor_shapes[key] = list(tensor.shape)
            del tensor
    if not tensor_keys:
        raise AssertionError("Saved adapter safetensors contains no tensors")
    magnitude_keys = [key for key in tensor_keys if "lora_magnitude_vector" in key]
    if len(magnitude_keys) != int(active_profile()["expected_modules"]):
        raise AssertionError(
            f"Saved DoRA magnitude tensor count drifted: {len(magnitude_keys)}"
        )
    return {
        "peft_type": "LORA",
        "adapter_method": "DoRA",
        "use_dora": True,
        "adapter_config_parse_ok": True,
        "safetensors_header_open_ok": True,
        "all_tensor_payloads_decoded_and_finite": True,
        "safetensors_file": weight_files[0].name,
        "tensor_count": len(tensor_keys),
        "magnitude_vector_tensor_count": len(magnitude_keys),
        "decoded_element_count": decoded_elements,
        "tensor_shape_manifest_sha256": json_sha(tensor_shapes),
        "passed": True,
    }


def trainable_tensor_sha256(model) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    tensor_count = 0
    element_count = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        tensor_count += 1
        element_count += int(tensor.numel())
        del tensor
    if tensor_count == 0 or element_count == 0:
        raise AssertionError("No trainable tensors available for serialization invariant")
    return digest.hexdigest(), tensor_count, element_count


def save_checkpoint(model, root: Path, milestone: float, step: int, steps_per_epoch: int,
                    contract_sha: str, history: list[dict]) -> dict:
    name = f"epoch_{milestone:05.3f}".replace(".", "p")
    directory = root / "checkpoints" / name
    directory.mkdir(parents=True, exist_ok=False)
    adapter = directory / "adapter"
    training_before = bool(model.training)
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    tensors_sha_before, trainable_tensor_count, trainable_element_count = trainable_tensor_sha256(model)
    model.save_pretrained(adapter, safe_serialization=True)
    tensors_sha_after, after_tensor_count, after_element_count = trainable_tensor_sha256(model)
    cpu_rng_unchanged = torch.equal(cpu_rng_before, torch.random.get_rng_state())
    cuda_rng_after = torch.cuda.get_rng_state_all()
    cuda_rng_unchanged = len(cuda_rng_before) == len(cuda_rng_after) and all(
        torch.equal(before, after) for before, after in zip(cuda_rng_before, cuda_rng_after, strict=True)
    )
    serialization_invariance = {
        "model_training_before": training_before,
        "model_training_after": bool(model.training),
        "cpu_rng_unchanged": cpu_rng_unchanged,
        "cuda_rng_unchanged": cuda_rng_unchanged,
        "trainable_tensor_sha256_before": tensors_sha_before,
        "trainable_tensor_sha256_after": tensors_sha_after,
        "trainable_tensor_count": trainable_tensor_count,
        "trainable_element_count": trainable_element_count,
        "passed": (
            training_before and bool(model.training)
            and cpu_rng_unchanged and cuda_rng_unchanged
            and tensors_sha_before == tensors_sha_after
            and trainable_tensor_count == after_tensor_count
            and trainable_element_count == after_element_count
        ),
    }
    if not serialization_invariance["passed"]:
        raise AssertionError(f"Checkpoint serialization changed training state: {serialization_invariance}")
    hashes = {
        str(path.relative_to(directory)): file_sha(path)
        for path in sorted(adapter.rglob("*")) if path.is_file()
    }
    if not hashes:
        raise AssertionError("No adapter files saved")
    loadability_report = verify_saved_adapter_loadability(adapter)
    metadata = {
        "checkpoint_id": f"{active_profile()['name']}_epoch_{milestone:.3f}_step_{step}".replace(".", "p"),
        "loadability_report": loadability_report,
        "serialization_invariance": serialization_invariance,
        "checkpoint_relative_dir": str(directory.relative_to(root)),
        "requested_epoch": milestone,
        "actual_optimizer_step": step,
        "actual_epoch": step / steps_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "contract_sha256": contract_sha,
        "adapter_files_sha256": hashes,
        "adapter_bundle_sha256": json_sha(hashes),
        "history_so_far": history,
    }
    write_json(directory / "checkpoint_metadata.json", metadata)
    write_json(directory / "checkpoint_manifest.json", metadata)
    return metadata


def train(
    frame: pd.DataFrame,
    validation: pd.DataFrame | None,
    windows: dict[int, list[list[int]]],
    contract: dict,
    root: Path,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; enable Kaggle GPU")
    transformers = ensure_transformers(CONFIG["model_id"])
    processor = transformers.AutoProcessor.from_pretrained(
        CONFIG["model_id"],
        local_files_only=Path(CONFIG["model_id"]).expanduser().is_dir(),
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = LabelValueDataset(frame, processor)
    write_json(root / "token_length_manifest.json", dataset.length_report)
    (root / "processor").mkdir(parents=True, exist_ok=False)
    processor.save_pretrained(root / "processor")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    model, target_report, precision = load_model(transformers, root)
    padding_report = right_padding_sparse_logits_preflight(
        model, dataset, int(tokenizer.pad_token_id)
    )
    write_json(root / "right_padding_sparse_logits_preflight.json", padding_report)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    plan = run_plan()
    steps_per_epoch = len(windows[1])
    scheduler_total_steps = round(steps_per_epoch * plan["scheduler_horizon_epochs"])
    train_until_steps = math.ceil(steps_per_epoch * plan["train_until_epoch"])
    warmup_steps = max(1, round(scheduler_total_steps * CONFIG["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_lambda(step, scheduler_total_steps, warmup_steps)
    )
    milestone_steps = defaultdict(list)
    for milestone in plan["checkpoint_epochs"]:
        milestone_steps[math.ceil(milestone * steps_per_epoch)].append(milestone)

    fixed_denominator = float(frame.row_loss_weight.sum() / steps_per_epoch)
    history, checkpoints, global_step = [], [], 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, math.ceil(plan["train_until_epoch"]) + 1):
        remaining_steps = train_until_steps - global_step
        epoch_windows = windows[epoch][:min(steps_per_epoch, remaining_steps)]
        if not epoch_windows:
            break
        seen, numerator_epoch, denominator_epoch = [], 0.0, 0.0
        progress = tqdm(epoch_windows, desc=f"{active_profile()['name']} epoch={epoch}", unit="step")
        for window in progress:
            window_mass = float(frame.loc[window, "row_loss_weight"].sum())
            for offset in range(0, len(window), CONFIG["batch_size"]):
                indices = window[offset:offset + CONFIG["batch_size"]]
                batch = collate([dataset[index] for index in indices], int(tokenizer.pad_token_id))
                seen.extend(batch["ids"])
                input_ids = batch["input_ids"].to(model.device, non_blocking=True)
                attention = batch["attention_mask"].to(model.device, non_blocking=True)
                gold_class_indices = batch["gold_class_indices"].to(model.device, non_blocking=True)
                row_weights = batch["row_loss_weights"].to(model.device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs, logits, sparse_report = sparse_label_logits(model, input_ids, attention)
                allowed_ids = dataset.length_report["allowed_token_ids_in_class_order"]
                restricted_logits = logits[:, allowed_ids].float()
                if sparse_report["lm_head_positions_computed"] > len(indices):
                    raise AssertionError("Sparse lm_head computed more than one position per row")
                losses = torch.nn.functional.cross_entropy(
                    restricted_logits, gold_class_indices, reduction="none",
                )
                if losses.shape != row_weights.shape:
                    raise AssertionError(f"One label-token loss per row required: {losses.shape}")
                numerator = (losses * row_weights.float()).sum()
                if not torch.isfinite(numerator):
                    raise RuntimeError(f"Non-finite loss epoch={epoch} step={global_step}")
                (numerator / fixed_denominator).backward()
                numerator_epoch += float(numerator.detach().cpu())
                del outputs, logits, restricted_logits, losses, numerator
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, CONFIG["max_grad_norm"], error_if_nonfinite=True)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm epoch={epoch} step={global_step}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            denominator_epoch += window_mass
            progress.set_postfix(
                loss=f"{numerator_epoch / denominator_epoch:.5f}",
                mass=f"{window_mass:.2f}", step=global_step,
            )
            for milestone in milestone_steps.get(global_step, []):
                checkpoint = save_checkpoint(
                    model, root, milestone, global_step, steps_per_epoch,
                    contract["contract_sha256"], history,
                )
                if validation is not None:
                    metrics = evaluate_fold0_checkpoint(
                        model, processor, validation, milestone, root, contract["contract_sha256"]
                    )
                    checkpoint["fold0_evaluation"] = metrics
                    directory = root / checkpoint["checkpoint_relative_dir"]
                    write_json(directory / "checkpoint_metadata.json", checkpoint)
                    write_json(directory / "checkpoint_manifest.json", checkpoint)
                checkpoints.append(checkpoint)
        expected_ids = [str(frame.at[index, "id"]) for window in epoch_windows for index in window]
        if seen != expected_ids or len(set(seen)) != len(seen):
            raise AssertionError(f"Epoch {epoch} ID order/uniqueness contract failed")
        if len(epoch_windows) == steps_per_epoch and len(seen) != len(frame):
            raise AssertionError(f"Complete epoch {epoch} did not see every train ID once")
        history.append({
            "epoch": epoch,
            "epoch_fraction_completed": len(epoch_windows) / steps_per_epoch,
            "optimizer_step": global_step,
            "seen_ids": len(seen),
            "seen_positives": int(frame.loc[
                [index for window in epoch_windows for index in window], "label"
            ].sum()),
            "weighted_restricted2_label_loss": numerator_epoch / denominator_epoch,
            "row_weight_sum": denominator_epoch,
            "fixed_denominator": fixed_denominator,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        })
        pd.DataFrame(history).to_csv(root / "training_history.csv", index=False)
    expected_steps = train_until_steps
    expected_checkpoint_count = len(plan["checkpoint_epochs"])
    if global_step != expected_steps or len(checkpoints) != expected_checkpoint_count:
        raise AssertionError("Optimizer-step/checkpoint contract failed")
    # Full-train has no held-out fold and therefore no checkpoint selection.
    # Every requested milestone is retained for downstream submission testing.
    fold0_ranking = []
    selected_epoch_manifest = None

    summary = {
        "status": "complete",
        "mode_plan": plan,
        "profile": active_profile(),
        "rows_per_epoch": len(dataset),
        "train_until_epoch": plan["train_until_epoch"],
        "steps_per_epoch": steps_per_epoch,
        "scheduler_total_steps": scheduler_total_steps,
        "actual_optimizer_steps": global_step,
        "actual_train_until_epoch": global_step / steps_per_epoch,
        "warmup_steps": warmup_steps,
        "history": history,
        "checkpoints": checkpoints,
        "dora_target_manifest_sha256": json_sha(target_report),
        "precision": precision,
        "right_padding_sparse_logits_preflight": padding_report,
        "contract_sha256": contract["contract_sha256"],
        "fold0_checkpoint_ranking": fold0_ranking,
        "selected_epoch_manifest": selected_epoch_manifest,
        "validation": None,
        "checkpoint_policy": "save all requested full-train milestones; no fold-based selection",
    }
    write_json(root / "training_summary.json", summary)
    return summary


def expose_download(archive: Path) -> None:
    try:
        from IPython.display import FileLink, display
        working = Path("/kaggle/working")
        link = working / archive.name
        if archive.resolve() != link.resolve():
            if link.exists():
                raise FileExistsError(f"Download hard-link already exists: {link}")
            os.link(archive, link)
        os.chdir(working)
        display(FileLink(link.name))
    except Exception as error:
        print(f"Download link unavailable: {error}", flush=True)


def run() -> None:
    plan = run_plan()
    if dry_run():
        root = Path(tempfile.mkdtemp(prefix="lvj_dora_fulltrain_preflight_")) / (
            f"{plan['mode']}_{active_profile()['name']}"
        )
    else:
        suffix = "fulltrain_3ep_checkpoints_every_0p5"
        root = Path(CONFIG["output_root"]) / plan["mode"] / active_profile()["name"] / suffix
        if root.exists():
            raise FileExistsError(f"Output already exists; move it aside before a fresh exact run: {root}")
    frame, validation, windows, contract = prepare_contracts(root)
    if dry_run():
        print(
            f"CPU preflight passed: root={root} schedule={contract['schedule']['schedule_sha256']}",
            flush=True,
        )
        return
    summary = train(frame, validation, windows, contract, root)
    archive = Path(shutil.make_archive(str(root), "zip", root_dir=root))
    summary["archive"] = str(archive)
    summary["archive_sha256"] = file_sha(archive)
    write_json(root / "training_summary.json", summary)
    print("\n===== LVJ DORA FULLTRAIN ZERO-FEWSHOT 3-EPOCH RUN COMPLETE =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    expose_download(archive)


if __name__ == "__main__":
    run()
