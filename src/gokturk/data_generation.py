"""Synthetic Turkish data generation using teacher models.

Two-step pipeline (configurable):

1. **Draft** -- a teacher is asked to invent a realistic Turkish user prompt for
   a given task family and topic. The draft is parsed as JSON; malformed drafts
   are rejected, never repaired silently.
2. **Solve** -- a teacher answers the drafted prompt. The answer becomes the
   training target. It is filtered by :mod:`gokturk.data_quality` before being
   accepted, and it is *not* assumed to be correct: teacher outputs still need
   human spot-checking (see ``docs/architecture.md``).

Professional pipeline (``generate_pro``)
----------------------------------------
For high-quality, diverse corpora the module adds above the basic loop:

* **Curriculum planning** (:mod:`gokturk.curriculum`): a deterministic plan of
  N slots across ten task families with topic banks that never repeat the same
  topic twice in a row.
* **Multi-candidate sampling**: draw ``candidates`` answers per prompt and keep
  the one with the fewest validator failures, then the longest (a cheap proxy
  for completeness, recorded in metadata).
* **Self-critique + repair**: ask the teacher to audit its own answer against a
  rubric and to return a corrected version only if needed. The repair is kept
  only when it passes the same validators *and* no rubric finding remains.
* **Role routing**: reasoning/architecture tasks go to a reasoning-role teacher
  when one is configured, coding tasks to a code-role teacher, with fallback.
* **Provenance**: every record stores family, level, candidate index, scores,
  repairs and usage so the corpus is auditable.

Safety behaviour
----------------
* No API key -> the run stops before a single request is sent.
* Authentication failure (401/403) -> immediate abort.
* Repeated transport failures -> abort after
  ``provider.abort_after_consecutive_failures`` consecutive failures.
* Every accepted record is appended to disk immediately, so an aborted run
  keeps its partial data and can be resumed with ``--resume``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    ConfigError,
    ProviderConfig,
    QualityConfig,
    TeacherModelConfig,
    TeachersConfig,
    load_quality_config,
    load_teachers_config,
)
from .curriculum import Plan, Slot, build_plan, estimate_cost
from .data_quality import assess_record, read_jsonl
from .schema import SCHEMA_VERSION, TASK_TYPES
from .task_specs import TaskFamily, family_summary, get_family
from .teachers import (
    RETRYABLE_ERRORS,
    TeacherAuthError,
    TeacherClient,
    TeacherConfigError,
    TeacherError,
    TeacherPool,
    TeacherRequestError,
)

LOGGER = logging.getLogger("gokturk.data_generation")

PROMPT_VERSION = "v1"

DRAFTER_SYSTEM = (
    "Sen Türkçe bir veri seti tasarımcısısın. Görevin, verilen görev türü ve konu "
    "için gerçek bir kullanıcının yazacağı tek turlu, doğal ve net bir Türkçe "
    "istek (prompt) üretmektir. Kullanıcı isteği kısa ama yeterince "
    "belirgin olmalı, ek bağlam gerektirmemeli ve modelin yanıtlayabileceği "
    "somut bir çıktı istemelidir. Yalnızca JSON döndür."
)

DRAFTER_USER = """Görev türü: {task_type}
Konu: {topic}
Zorluk: {level}
Varyasyon numarası: {variant}

Şu şemada SADECE JSON döndür (açıklama, markdown, kod bloğu yok):
{{"prompt": "<Türkçe kullanıcı isteği>"}}

Kurallar:
- İstek Türkçe olacak ve {task_type} görev türüne uygun olacak.
- İstek, cevabı verilecek asıl görevi içerecek; "sen bir uzmansın" gibi rol
  yönergeleri kullanıcı isteğinde yer almayacak.
- 10 ile 90 kelime arasında olacak.
- Daha önce üretilmiş örneklerden farklı olacak."""

ANSWER_SYSTEM = (
    "Sen GökTürk-Ultra için eğitim verisi üreten, alanında uzman bir Türkçe "
    "asistansın. Yanıtlarını her zaman Türkçe, doğru, eksiksiz ve doğrudan "
    "kullanıcının istediği biçimde verirsin. Gereksiz giriş cümlesi kurmazsın. "
    "Kod istendiğinde çalışır kod bloğu verirsin. Araç çağrısı istendiğinde "
    "istenen JSON şemasına birebir uyarsın. Bilmediğin bir şeyi uydurmazsın."
)

ANSWER_USER = """Kullanıcı isteği:
\"\"\"
{prompt}
\"\"\""""

TOOL_FORMALISM = (
    "\n\nBu örnek bir araç çağrısı görevidir. Yanıt, yalnızca aşağıdaki şemaya "
    "uyan bir JSON nesnesi olsun:\n"
    '{{"tool": "<araç_adı>", "arguments": {{...}}}}\n'
    "Açıklama, markdown veya ek metin ekleme."
)


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------
@dataclass
class SeedTask:
    id: str
    task_type: str
    topic: str
    level: str = "orta"
    prompt: str | None = None  # fixed prompt skips the draft step
    n_variants: int = 1
    notes: str | None = None

    def validate(self, index: int = 0) -> None:
        if not self.id.strip():
            raise ConfigError(f"seed[{index}]: id must not be empty")
        if self.task_type not in TASK_TYPES:
            raise ConfigError(
                f"seed[{index}] ({self.id}): task_type {self.task_type!r} not in {list(TASK_TYPES)}"
            )
        if not self.topic.strip() and not (self.prompt and self.prompt.strip()):
            raise ConfigError(f"seed[{index}] ({self.id}): need a topic or a fixed prompt")
        if self.n_variants < 1:
            raise ConfigError(f"seed[{index}] ({self.id}): n_variants must be >= 1")


def load_seeds(path: str | Path) -> list[SeedTask]:
    records, issues = read_jsonl(path)
    if issues:
        raise ConfigError(
            f"{path}: {len(issues)} invalid line(s); first: line {issues[0].line}: {issues[0].reason}"
        )
    known = {f for f in SeedTask.__dataclass_fields__}
    seeds: list[SeedTask] = []
    for i, rec in enumerate(records):
        unknown = set(rec) - known
        if unknown:
            raise ConfigError(f"{path}: seed[{i}] has unknown field(s) {sorted(unknown)}")
        seed = SeedTask(**{k: v for k, v in rec.items() if k in known})
        seed.validate(i)
        seeds.append(seed)
    if not seeds:
        raise ConfigError(f"{path}: no seeds found")
    ids = [s.id for s in seeds]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ConfigError(f"{path}: duplicate seed id(s) {sorted(dupes)}")
    return seeds


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------
def build_draft_request(seed: SeedTask, variant: int) -> tuple[str, str]:
    return (
        DRAFTER_SYSTEM,
        DRAFTER_USER.format(
            task_type=seed.task_type,
            topic=seed.topic,
            level=seed.level,
            variant=variant,
        ),
    )


def build_answer_request(seed: SeedTask, prompt: str) -> tuple[str, str]:
    user = ANSWER_USER.format(prompt=prompt)
    if seed.task_type == "tool_calling":
        user += TOOL_FORMALISM
    return ANSWER_SYSTEM, user


def parse_draft(text: str) -> str | None:
    """Extract ``prompt`` from a drafter response. Returns None if unusable."""
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = [ln for ln in candidate.splitlines() if not ln.strip().startswith("```")]
        candidate = "\n".join(lines).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, Mapping):
        return None
    prompt = obj.get("prompt")
    if not isinstance(prompt, str) or len(prompt.strip()) < 10:
        return None
    return prompt.strip()


def make_record_id(seed_id: str, teacher: str, variant: int, task_type: str) -> str:
    digest = hashlib.sha1(f"{seed_id}|{teacher}|{variant}".encode()).hexdigest()[:10]
    return f"gtu-{task_type[:3]}-{digest}"


def build_record(
    *,
    seed: SeedTask,
    prompt: str,
    response: str,
    teacher: TeacherModelConfig,
    provider: ProviderConfig,
    variant: int,
    response_meta: Mapping[str, Any],
    draft_teacher: str | None = None,
) -> dict[str, Any]:
    return {
        "id": make_record_id(seed.id, teacher.name, variant, seed.task_type),
        "prompt": prompt,
        "response": response,
        "source": {
            "kind": "teacher",
            "teacher": teacher.name,
            "teacher_model": teacher.model,
            "endpoint": provider.base_url,
            "task_type": seed.task_type,
            "template_version": PROMPT_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed_id": seed.id,
            "draft_teacher": draft_teacher,
            "finish_reason": response_meta.get("finish_reason"),
            "truncated": response_meta.get("finish_reason") == "length",
        },
        "meta": {
            "lang": "tr",
            "level": seed.level,
            "variant": variant,
            "latency_s": response_meta.get("latency_s"),
            "usage": response_meta.get("usage", {}),
        },
        "schema_version": SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
@dataclass
class GenerationStats:
    requested: int = 0
    accepted: int = 0
    rejected: int = 0
    failed: int = 0
    skipped_existing: int = 0
    aborted: bool = False
    abort_reason: str | None = None
    reject_reasons: dict[str, int] = field(default_factory=dict)
    teachers_used: dict[str, int] = field(default_factory=dict)
    critique_failures: int = 0
    rejected_repairs: int = 0
    plan_summary: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected += 1
        key = reason.split("(")[0].split(":")[0].strip()[:80]
        self.reject_reasons[key] = self.reject_reasons.get(key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationAborted(RuntimeError):
    """Raised when generation stops safely; carries the partial statistics."""

    def __init__(self, message: str, stats: GenerationStats | None = None) -> None:
        super().__init__(message)
        self.stats = stats


def _existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    records, _ = read_jsonl(path)
    return {str(r.get("id", "")) for r in records if r.get("id")}


def generate(
    teachers: TeachersConfig,
    seeds: Sequence[SeedTask],
    *,
    out_path: str | Path,
    per_seed: int = 1,
    resume: bool = True,
    quality: QualityConfig | None = None,
    client: TeacherClient | None = None,
    pool: TeacherPool | None = None,
    sleep_between: float = 0.0,
    max_records: int | None = None,
) -> GenerationStats:
    """Run the pipeline. Returns stats; raises on fatal teacher errors."""
    qcfg = quality or QualityConfig()
    qcfg.validate()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = out.with_name(out.stem + ".rejects.jsonl")

    teacher_pool = pool or TeacherPool(teachers, client=client)
    stats = GenerationStats()
    existing = _existing_ids(out) if resume else set()
    if not resume and out.exists():
        out.unlink()

    abort_after = teachers.provider.abort_after_consecutive_failures

    with out.open("a", encoding="utf-8") as out_fh, rejects_path.open(
        "a", encoding="utf-8"
    ) as rej_fh:
        for seed in seeds:
            variants = max(per_seed, seed.n_variants)
            for variant in range(variants):
                if max_records and stats.accepted >= max_records:
                    LOGGER.info("max_records sınırına ulaşıldı (%d)", max_records)
                    return stats
                stats.requested += 1
                try:
                    _generate_one(
                        seed=seed,
                        variant=variant,
                        teachers=teachers,
                        pool=teacher_pool,
                        qcfg=qcfg,
                        stats=stats,
                        existing=existing,
                        out_fh=out_fh,
                        rej_fh=rej_fh,
                    )
                except (TeacherAuthError, TeacherConfigError) as exc:
                    stats.aborted = True
                    stats.abort_reason = f"{type(exc).__name__}: {exc}"
                    LOGGER.error("Öğretmen erişimi kritik hata verdi: %s", exc)
                    raise GenerationAborted(stats.abort_reason, stats) from exc
                except RETRYABLE_ERRORS as exc:
                    stats.failed += 1
                    consecutive = teacher_pool.register_failure()
                    LOGGER.warning("İstek başarısız: %s", exc)
                    if consecutive >= abort_after:
                        stats.aborted = True
                        stats.abort_reason = (
                            f"{consecutive} ardışık hata; öğretmen kullanılamıyor gibi görünüyor"
                        )
                        raise GenerationAborted(stats.abort_reason, stats) from exc
                except TeacherRequestError as exc:
                    # Bad request (e.g. wrong model id): fatal, retrying is pointless.
                    stats.aborted = True
                    stats.abort_reason = f"{type(exc).__name__}: {exc}"
                    raise GenerationAborted(stats.abort_reason, stats) from exc
                except TeacherError as exc:  # pragma: no cover - defensive
                    stats.failed += 1
                    LOGGER.warning("Beklenmeyen öğretmen hatası: %s", exc)
                if sleep_between:
                    time.sleep(sleep_between)
    return stats


def _generate_one(
    *,
    seed: SeedTask,
    variant: int,
    teachers: TeachersConfig,
    pool: TeacherPool,
    qcfg: QualityConfig,
    stats: GenerationStats,
    existing: set[str],
    out_fh,
    rej_fh,
) -> None:
    # ---- draft ---------------------------------------------------------
    if seed.prompt:
        prompt = seed.prompt
        draft_teacher = None
    else:
        sys_prompt, user_prompt = build_draft_request(seed, variant)
        draft_resp, draft_model = pool.ask(sys_prompt, user_prompt)
        prompt = parse_draft(draft_resp.text)
        if prompt is None:
            stats.reject("drafter returned unusable JSON")
            rej_fh.write(
                json.dumps(
                    {"seed_id": seed.id, "variant": variant, "reason": "bad_draft",
                     "raw_preview": draft_resp.text[:200]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            return
        draft_teacher = draft_model.name

    # ---- answer --------------------------------------------------------
    sys_prompt, user_prompt = build_answer_request(seed, prompt)
    answer_resp, teacher = pool.ask(sys_prompt, user_prompt)
    pool.reset_failures()

    record = build_record(
        seed=seed,
        prompt=prompt,
        response=answer_resp.text,
        teacher=teacher,
        provider=teachers.provider,
        variant=variant,
        response_meta=answer_resp.as_metadata(),
        draft_teacher=draft_teacher,
    )
    if record["id"] in existing:
        stats.skipped_existing += 1
        return

    problems = assess_record(record, qcfg)
    if problems:
        stats.reject(problems[0])
        rej_fh.write(
            json.dumps(
                {"id": record["id"], "reason": problems, "prompt": prompt[:300]},
                ensure_ascii=False,
            )
            + "\n"
        )
        rej_fh.flush()
        return

    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_fh.flush()
    existing.add(record["id"])
    stats.accepted += 1
    stats.teachers_used[teacher.name] = stats.teachers_used.get(teacher.name, 0) + 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Öğretmen modellerle sentetik veri üretimi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  # 10.000 örneklik profesyonel plan (ağa çıkmaz)\n"
            "  python -m gokturk.data_generation --mode pro --target 10000 \\\n"
            "      --plan-only --plan-out data/raw/plan.json\n\n"
            "  # gerçek üretim (EVREN_API_KEY + doğrulanmış model kimlikleri gerekir)\n"
            "  python -m gokturk.data_generation --mode pro --target 10000 \\\n"
            "      --out data/raw/pro.jsonl --candidates 2\n\n"
            "  # aile envanteri\n"
            "  python -m gokturk.data_generation --list-families\n"
        ),
    )
    parser.add_argument("--mode", choices=["basic", "pro"], default="basic")
    parser.add_argument("--teachers-config", default="configs/teachers.yaml")
    parser.add_argument("--seeds", default="data/seeds/turkish_seed_tasks.jsonl")
    parser.add_argument("--out", default="data/raw/generated.jsonl")
    parser.add_argument("--per-seed", type=int, default=1)
    parser.add_argument("--quality-config", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--health-check", action="store_true", help="Yalnızca bağlantıyı doğrula")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Sağlayıcının /models listesini al ve yapılandırılmış kimliklerle karşılaştır",
    )
    parser.add_argument("--dry-run", action="store_true", help="Ağa çıkmadan planı yazdır")
    parser.add_argument("--list-families", action="store_true", help="Görev ailelerini listele")
    # --- pro mode ---
    parser.add_argument("--target", type=int, default=1000, help="pro modunda hedef örnek sayısı")
    parser.add_argument("--families", default=None, help="Virgülle ayrılmış aile listesi")
    parser.add_argument("--levels", default=None, help="temel,orta,ileri")
    parser.add_argument("--plan-only", action="store_true", help="Yalnızca plan + maliyet tahmini")
    parser.add_argument("--plan-out", default=None, help="Plan JSON yolu")
    parser.add_argument("--plan-in", default=None, help="Mevcut planı yeniden kullan")
    parser.add_argument("--candidates", type=int, default=2, help="İstem başına aday yanıt sayısı")
    parser.add_argument("--no-critique", action="store_true", help="Öz-eleştiri/düzeltme adımını kapat")
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0, help="İstekler arası bekleme (sn)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list_families:
        print(json.dumps(family_summary(), ensure_ascii=False, indent=2))
        return 0

    families = [f.strip() for f in (args.families or "").split(",") if f.strip()] or None
    levels = [lvl.strip() for lvl in (args.levels or "").split(",") if lvl.strip()] or None

    if args.mode == "pro" and (args.plan_only or args.dry_run):
        try:
            summary = plan_only(
                target=args.target,
                seed=args.seed,
                families=families,
                levels=levels,
                candidates=args.candidates,
                plan_path=args.plan_out,
            )
        except (ValueError, KeyError) as exc:
            print(f"Plan hatası: {exc}", file=sys.stderr)
            return 4
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(
            "\nUYARI: plan yalnızca dağılımı gösterir; öğretmen model kimlikleri ve "
            "uç nokta doğrulanmadan üretim başlamaz.",
            file=sys.stderr,
        )
        return 0

    try:
        teachers = load_teachers_config(args.teachers_config)
        seeds = load_seeds(args.seeds)
        quality = load_quality_config(args.quality_config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Yapılandırma hatası: {exc}", file=sys.stderr)
        return 4

    provider = teachers.provider
    key_present = provider.api_key() is not None
    total_requests = sum(max(args.per_seed, s.n_variants) for s in seeds)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "provider": provider.name,
                    "base_url": provider.base_url or "(boş — doğrulanmalı)",
                    "api_key_env": provider.api_key_env,
                    "api_key_present": key_present,
                    "teachers": [t.to_public_dict() for t in teachers.models()],
                    "seeds": len(seeds),
                    "planned_records": total_requests,
                    "tasks": sorted({s.task_type for s in seeds}),
                    "out": args.out,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if not provider.base_url:
            print(
                "\nUYARI: provider.base_url boş. "
                "configs/teachers.yaml içine sağlayıcının doğrulanmış adresini yazın.",
                file=sys.stderr,
            )
        if not key_present:
            print(
                f"\nUYARI: {provider.api_key_env} tanımlı değil; gerçek üretim "
                "bu ortam değişkeni olmadan başlamaz.",
                file=sys.stderr,
            )
        return 0

    if not provider.base_url:
        print(
            "Durduruldu: provider.base_url boş. Doğrulanmamış bir adrese istek "
            "gönderilmez. configs/teachers.yaml dosyasını doldurun.",
            file=sys.stderr,
        )
        return 3
    if not key_present:
        print(
            f"Durduruldu: {provider.api_key_env} ortam değişkeni yok. "
            "Anahtarı dosyaya yazmayın; ortam değişkeni veya Kaggle Secrets kullanın.",
            file=sys.stderr,
        )
        return 3

    pool = TeacherPool(teachers)
    if args.list_models:
        try:
            report = pool.list_models()
        except TeacherError as exc:
            print(f"/models alınamadı: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["missing"]:
            print(
                "\nEKSİK: yapılandırmadaki şu kimlikler sağlayıcıda yok: "
                + ", ".join(report["missing"])
                + ". configs/teachers.yaml içindeki `model` alanlarını "
                "yukarıdaki listeden düzeltin.",
                file=sys.stderr,
            )
        return 0 if report["ok"] else 2
    if args.health_check:
        report = pool.health_check()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 2

    started = time.time()
    if args.mode == "pro":
        try:
            teachers.require_ready()
        except ConfigError as exc:
            print(f"Durduruldu: {exc}", file=sys.stderr)
            return 3
        settings = ProSettings(
            candidates=args.candidates,
            critique=not args.no_critique,
            max_repair_attempts=args.max_repairs,
        )
        existing_plan = None
        if args.plan_in:
            existing_plan = _load_plan(args.plan_in)
        try:
            stats = generate_pro(
                teachers,
                target=args.target,
                out_path=args.out,
                seed=args.seed,
                families=families,
                levels=levels,
                settings=settings,
                quality=quality,
                pool=pool,
                plan=existing_plan,
                resume=not args.no_resume,
                sleep_between=args.sleep,
                write_plan=args.plan_out,
            )
            exit_code = 0
        except GenerationAborted as exc:
            print(f"\nÜretim güvenli biçimde durduruldu: {exc}", file=sys.stderr)
            stats = exc.stats or GenerationStats()
            stats.aborted = True
            stats.abort_reason = str(exc)
            exit_code = 2
        summary = stats.to_dict()
        summary["duration_s"] = round(time.time() - started, 1)
        summary["out"] = args.out
        summary["mode"] = "pro"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return exit_code

    try:
        stats = generate(
            teachers,
            seeds,
            out_path=args.out,
            per_seed=args.per_seed,
            resume=not args.no_resume,
            quality=quality,
            pool=pool,
            max_records=args.max_records,
        )
        exit_code = 0
    except GenerationAborted as exc:
        print(f"\nÜretim güvenli biçimde durduruldu: {exc}", file=sys.stderr)
        stats = exc.stats or GenerationStats()
        stats.aborted = True
        stats.abort_reason = str(exc)
        exit_code = 2

    summary = stats.to_dict()
    summary["duration_s"] = round(time.time() - started, 1)
    summary["out"] = args.out
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code




# ===========================================================================
# Professional pipeline
# ===========================================================================
CRITIQUE_SYSTEM = (
    "Sen bir Türkçe eğitim verisi denetçisisin. Sana bir kullanıcı isteği ve "
    "aday yanıt verilir. Yanıtı verilen değerlendirme ölçütlerine göre denetler, "
    "kusurları kısa ve somut biçimde listelersin. Yalnızca JSON döndürürsün."
)

CRITIQUE_USER = """Görev türü: {task_type}
Kullanıcı isteği:
\"\"\"
{prompt}
\"\"\"

Aday yanıt:
\"\"\"
{response}
\"\"\"

Ölçütler:
{criteria}

Yalnızca şu şemada JSON döndür:
{{"sorunlar": ["<somut kusur>", "..."], "duzeltilmis_yanit": "<düzeltilmiş tam yanıt veya boş>"}}

Kurallar:
- Yanıt tüm ölçütleri karşılıyorsa "sorunlar" boş liste olsun ve "duzeltilmis_yanit" boş string kalsın.
- Uygun ve doğru yanıtı gereksiz yere değiştirme.
- Yeni bilgi uydurma; yalnızca kusurları düzelt.
- "duzeltilmis_yanit" doluysa yanıtın TAMAMINI içermeli, kısmi parça verme."""

CRITIQUE_CRITERIA = {
    "coding": [
        "Kod çalışır ve istenen dili kullanır",
        "Kod bloğu dil etiketli ve eksiksiz",
        "Kenar durumlar belirtilmiş",
        "Yanıt tamamen Türkçe açıklama içerir",
    ],
    "coding_debug": [
        "Hatanın nedeni doğru açıklanmış",
        "Düzeltilmiş kod tam ve çalışır",
        "Kenar durumlar belirtilmiş",
    ],
    "reasoning": [
        "Tüm veriler kullanılmış, hesap adımları denetlenebilir",
        "Sonuç net ve sağlaması yapılmış",
        "Varsayımlar açıkça yazılmış",
    ],
    "system_architecture": [
        "Varsayımlar (ölçek, bütçe, kısıt) belirtilmiş",
        "Bileşenler ve veri akışı somut",
        "Riskler ve ödünleşimler gerçekçi",
        "Doğrulama/ölçüm adımları var",
    ],
    "turkish_advanced": [
        "Dil bilgisi ve yazım kurallarına uygun",
        "Üslup hedef kitleye uygun",
        "Uydurma kaynak veya kural numarası yok",
    ],
    "instruction_following": [
        "İstekteki tüm kısıtlar (sayı/uzunluk/biçim) karşılanmış",
        "Fazladan istenmeyen içerik yok",
    ],
    "summarization": [
        "Özet kaynak metnin kapsamında, uydurma bilgi yok",
        "İstenen uzunlukta",
    ],
    "rewrite": [
        "Anlam korunmuş",
        "Hedef ton ve uzunluk uygulanmış",
    ],
    "translation": [
        "Çeviri eksiksiz ve akıcı",
        "Terimler tutarlı, özgün terim ilk kullanımda verilmiş",
    ],
    "tool_calling": [
        "Yanıt yalnızca tek bir JSON nesnesi",
        "Araç adı ve argümanlar istekteki imzayla birebir uyumlu",
    ],
}


@dataclass
class ProSettings:
    """Knobs for the professional pipeline."""

    candidates: int = 2
    critique: bool = True
    max_repair_attempts: int = 1
    temperature_jitter: float = 0.15
    reviewer_role: str = "reasoning"

    def validate(self) -> None:
        if self.candidates < 1:
            raise ConfigError("pro.candidates must be >= 1")
        if not 0 <= self.max_repair_attempts <= 3:
            raise ConfigError("pro.max_repair_attempts must be in [0, 3]")
        if self.temperature_jitter < 0:
            raise ConfigError("pro.temperature_jitter must be >= 0")


def parse_critique(text: str) -> dict[str, Any] | None:
    """Parse the JSON critique object; ``None`` when the shape is wrong."""
    candidate = text.strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, Mapping):
        return None
    problems = obj.get("sorunlar", obj.get("problems", []))
    repaired = obj.get("duzeltilmis_yanit", obj.get("corrected", ""))
    if not isinstance(problems, list):
        problems = [str(problems)]
    problems = [str(p).strip() for p in problems if str(p).strip()]
    if not isinstance(repaired, str):
        repaired = ""
    return {"problems": problems, "repaired": repaired.strip()}


def build_record_pro(
    *,
    slot: Slot,
    family: TaskFamily,
    prompt: str,
    response: str,
    teacher: TeacherModelConfig,
    provider: ProviderConfig,
    candidate_index: int,
    candidates_tried: int,
    critique: Mapping[str, Any] | None,
    draft_teacher: str | None,
    response_meta: Mapping[str, Any],
    soft_notes: Mapping[str, list[str]],
    template_version: str = "pro-v1",
) -> dict[str, Any]:
    """Build a schema-valid record with full, non-secret provenance."""
    record_id = make_record_id(slot.seed_id, teacher.name, slot.variant, family.id)
    return {
        "id": record_id,
        "prompt": prompt,
        "response": response,
        "source": {
            "kind": "teacher",
            "teacher": teacher.name,
            "teacher_model": teacher.model,
            "teacher_roles": teacher.roles(),
            "endpoint": provider.base_url,
            "task_type": family.id,
            "family_label": family.label,
            "template_version": template_version,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed_id": slot.seed_id,
            "draft_teacher": draft_teacher,
            "finish_reason": response_meta.get("finish_reason"),
            "truncated": response_meta.get("finish_reason") == "length",
        },
        "meta": {
            "lang": "tr",
            "level": slot.level,
            "topic": slot.topic,
            "variant": slot.variant,
            "candidate_index": candidate_index,
            "candidates_tried": candidates_tried,
            "critique_problems": list(critique.get("problems", [])) if critique else [],
            "repaired": bool(critique and critique.get("repaired")),
            "soft_notes": {k: v for k, v in soft_notes.items()},
            "latency_s": response_meta.get("latency_s"),
            "usage": response_meta.get("usage", {}),
            "review": {"status": "unreviewed", "note": "insan incelemesi bekliyor"},
        },
        "schema_version": SCHEMA_VERSION,
    }


def _criteria_for(family: TaskFamily) -> str:
    items = CRITIQUE_CRITERIA.get(family.id, ["Yanıt doğru, eksiksiz ve Türkçe"])
    return "\n".join(f"- {item}" for item in items)


def _score_candidate(problems: Sequence[str], response: str) -> tuple[int, int]:
    """Rank candidates: fewer validator failures first, then longer."""
    return (len(problems), -len(response))


def generate_pro(
    teachers: TeachersConfig,
    *,
    target: int,
    out_path: str | Path,
    seed: int = 0,
    families: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    settings: ProSettings | None = None,
    quality: QualityConfig | None = None,
    pool: TeacherPool | None = None,
    client: TeacherClient | None = None,
    plan: Plan | None = None,
    resume: bool = True,
    sleep_between: float = 0.0,
    write_plan: str | Path | None = None,
) -> GenerationStats:
    """Professional corpus generation (see module docstring).

    Stops safely on authentication errors, request errors and repeated
    transport failures -- exactly like :func:`generate` -- and never writes a
    placeholder answer.
    """
    pro = settings or ProSettings()
    pro.validate()
    qcfg = quality or QualityConfig()
    qcfg.validate()

    plan = plan or build_plan(
        target, seed=seed, families=families, levels=levels, weights=weights
    )
    if write_plan:
        plan.write(write_plan)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = out.with_name(out.stem + ".rejects.jsonl")
    reviewed_path = out.with_name(out.stem + ".critique.jsonl")

    teacher_pool = pool or TeacherPool(teachers, client=client)
    stats = GenerationStats()
    existing = _existing_ids(out) if resume else set()
    if not resume and out.exists():
        out.unlink()

    abort_after = teachers.provider.abort_after_consecutive_failures

    with (
        out.open("a", encoding="utf-8") as out_fh,
        rejects_path.open("a", encoding="utf-8") as rej_fh,
        reviewed_path.open("a", encoding="utf-8") as crit_fh,
    ):
        for slot in plan.slots:
            stats.requested += 1
            family = get_family(slot.family)
            try:
                _generate_slot(
                    slot=slot,
                    family=family,
                    pro=pro,
                    qcfg=qcfg,
                    teachers=teachers,
                    pool=teacher_pool,
                    stats=stats,
                    existing=existing,
                    out_fh=out_fh,
                    rej_fh=rej_fh,
                    crit_fh=crit_fh,
                )
            except (TeacherAuthError, TeacherConfigError) as exc:
                stats.aborted = True
                stats.abort_reason = f"{type(exc).__name__}: {exc}"
                LOGGER.error("Öğretmen erişimi kritik hata verdi: %s", exc)
                raise GenerationAborted(stats.abort_reason, stats) from exc
            except RETRYABLE_ERRORS as exc:
                stats.failed += 1
                consecutive = teacher_pool.register_failure()
                LOGGER.warning("İstek başarısız: %s", exc)
                if consecutive >= abort_after:
                    stats.aborted = True
                    stats.abort_reason = (
                        f"{consecutive} ardışık hata; öğretmen kullanılamıyor gibi görünüyor"
                    )
                    raise GenerationAborted(stats.abort_reason, stats) from exc
            except TeacherRequestError as exc:
                stats.aborted = True
                stats.abort_reason = f"{type(exc).__name__}: {exc}"
                raise GenerationAborted(stats.abort_reason, stats) from exc
            except TeacherError as exc:  # pragma: no cover - defensive
                stats.failed += 1
                LOGGER.warning("Beklenmeyen öğretmen hatası: %s", exc)
            if sleep_between:
                time.sleep(sleep_between)

    stats.plan_summary = plan.by_family()
    return stats


def _generate_slot(
    *,
    slot: Slot,
    family: TaskFamily,
    pro: ProSettings,
    qcfg: QualityConfig,
    teachers: TeachersConfig,
    pool: TeacherPool,
    stats: GenerationStats,
    existing: set[str],
    out_fh,
    rej_fh,
    crit_fh,
) -> None:
    # ---- 1) draft the user request --------------------------------------
    draft_resp, draft_model = pool.ask_with_fallback(
        family.drafter_system(), family.drafter_user(slot.topic, slot.level, slot.variant)
    )
    prompt = parse_draft(draft_resp.text)
    if prompt is None:
        stats.reject("drafter returned unusable JSON")
        rej_fh.write(
            json.dumps(
                {"slot": slot.to_dict(), "reason": "bad_draft",
                 "raw_preview": draft_resp.text[:200]},
                ensure_ascii=False,
            )
            + "\n"
        )
        rej_fh.flush()
        return

    # ---- 2) sample N candidates -----------------------------------------
    candidates: list[tuple[str, Any, Any, list[str]]] = []
    tried = max(1, pro.candidates)
    for index in range(tried):
        temperature = None
        if pro.temperature_jitter and family.id not in ("tool_calling", "instruction_following"):
            base = 0.6
            temperature = round(
                min(1.0, max(0.1, base + index * pro.temperature_jitter)), 2
            )
        resp, teacher = pool.ask_with_fallback(
            family.answer_system(),
            family.answer_user(prompt),
            role=slot.role,
            temperature=temperature,
        )
        problems = family.problems(prompt, resp.text)
        candidates.append((resp.text, resp, teacher, problems))

    best_text, best_resp, best_teacher, best_problems = min(
        candidates, key=lambda c: _score_candidate(c[3], c[0])
    )

    # ---- 3) self-critique + repair --------------------------------------
    critique: dict[str, Any] | None = None
    response_text = best_text
    if pro.critique:
        critique = _critique_and_repair(
            family=family,
            prompt=prompt,
            response=best_text,
            pool=pool,
            role=pro.reviewer_role,
            attempts=pro.max_repair_attempts,
            qcfg=qcfg,
            stats=stats,
        )
        if critique and critique.get("repaired"):
            response_text = critique["repaired"]
        crit_fh.write(
            json.dumps(
                {
                    "id": make_record_id(slot.seed_id, best_teacher.name, slot.variant, family.id),
                    "slot": slot.to_dict(),
                    "candidate_problems": [c[3] for c in candidates],
                    "critique": critique or {},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        crit_fh.flush()

    # ---- 4) final validation --------------------------------------------
    final_problems = family.problems(prompt, response_text)
    if final_problems:
        stats.reject(final_problems[0])
        rej_fh.write(
            json.dumps(
                {"slot": slot.to_dict(), "reason": final_problems, "prompt": prompt[:300]},
                ensure_ascii=False,
            )
            + "\n"
        )
        rej_fh.flush()
        return

    record = build_record_pro(
        slot=slot,
        family=family,
        prompt=prompt,
        response=response_text,
        teacher=best_teacher,
        provider=teachers.provider,
        candidate_index=candidates.index(
            (best_text, best_resp, best_teacher, best_problems)
        ),
        candidates_tried=len(candidates),
        critique=critique,
        draft_teacher=draft_model.name,
        response_meta=best_resp.as_metadata(),
        soft_notes=family.soft_notes(prompt, response_text),
    )

    if record["id"] in existing:
        stats.skipped_existing += 1
        return

    # Shared schema/quality gate (lengths, refusals, control chars, ...).
    problems = assess_record(record, qcfg)
    if problems:
        stats.reject(problems[0])
        rej_fh.write(
            json.dumps(
                {"id": record["id"], "reason": problems, "prompt": prompt[:300]},
                ensure_ascii=False,
            )
            + "\n"
        )
        rej_fh.flush()
        return

    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_fh.flush()
    existing.add(record["id"])
    stats.accepted += 1
    stats.teachers_used[best_teacher.name] = stats.teachers_used.get(best_teacher.name, 0) + 1


def _critique_and_repair(
    *,
    family: TaskFamily,
    prompt: str,
    response: str,
    pool: TeacherPool,
    role: str,
    attempts: int,
    qcfg: QualityConfig,
    stats: GenerationStats,
) -> dict[str, Any] | None:
    """Ask the teacher to audit its own answer; keep a repair only if it passes."""
    critique: dict[str, Any] | None = None
    current = response
    for _ in range(max(1, attempts + 1)):
        try:
            resp, _teacher = pool.ask_with_fallback(
                CRITIQUE_SYSTEM,
                CRITIQUE_USER.format(
                    task_type=family.id,
                    prompt=prompt,
                    response=current,
                    criteria=_criteria_for(family),
                ),
                role=role,
            )
        except RETRYABLE_ERRORS as exc:
            LOGGER.warning("eleştiri adımı başarısız: %s", exc)
            stats.critique_failures += 1
            return critique
        parsed = parse_critique(resp.text)
        if parsed is None:
            stats.critique_failures += 1
            return critique
        repaired = parsed["repaired"]
        if not parsed["problems"] or not repaired:
            # Nothing to fix: keep the answer untouched.
            critique = {**parsed, "repaired": ""}
            return critique

        # A repair is only accepted when it clears the *same* hard validators
        # and the shared quality gate. Otherwise the original answer is kept and
        # the rejected repair is recorded for auditing.
        repair_problems = family.problems(prompt, repaired)
        if not repair_problems:
            repair_problems = assess_record(
                {
                    "id": "critique-check",
                    "prompt": prompt,
                    "response": repaired,
                    "source": {"kind": "teacher", "teacher": "critique"},
                },
                qcfg,
            )
        if repair_problems:
            stats.rejected_repairs += 1
            critique = {
                **parsed,
                "repaired": "",
                "rejected_repair_problems": repair_problems,
            }
            continue
        critique = {**parsed, "repaired": repaired}
        return critique
    return critique


def _load_plan(path: str | Path) -> Plan:
    """Load a plan written by :meth:`gokturk.curriculum.Plan.write`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    slots = [
        Slot(
            index=item["index"],
            family=item["family"],
            topic=item["topic"],
            level=item["level"],
            role=item["role"],
            variant=item["variant"],
            seed_id=item["seed_id"],
        )
        for item in data["slots"]
    ]
    return Plan(
        target=data.get("target", len(slots)),
        seed=data.get("seed", 0),
        slots=slots,
        families=data.get("families", []),
        notes=data.get("notes", []),
    )


def plan_only(
    *,
    target: int,
    seed: int = 0,
    families: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
    candidates: int = 2,
    plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a plan + cost estimate without touching the network."""
    plan = build_plan(target, seed=seed, families=families, levels=levels)
    if plan_path:
        plan.write(plan_path)
    return {
        "target": target,
        "seed": seed,
        "families": family_summary() if not families else [
            s for s in family_summary() if s["id"] in set(families)
        ],
        "counts": {
            "family": plan.by_family(),
            "level": plan.by_level(),
            "role": plan.by_role(),
        },
        "cost": estimate_cost(plan, candidates=candidates),
        "plan_path": str(plan_path) if plan_path else None,
    }

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
