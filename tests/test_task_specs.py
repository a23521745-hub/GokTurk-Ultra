"""Task families: prompt construction, validators, diversity, no invented facts."""

from __future__ import annotations

import json

import pytest

from gokturk.schema import TASK_TYPES
from gokturk.task_specs import (
    FAMILIES,
    HARD_VALIDATORS,
    check_code_block,
    check_min_sections,
    check_no_reasoning_leak,
    check_not_meta_preamble,
    check_tool_json,
    check_turkish_language,
    family_ids,
    family_summary,
    get_family,
    resolve_families,
    validate_generated,
)

GOOD_CODE = (
    "Aşağıdaki fonksiyon bir metindeki kelime sıklığını hesaplar.\n\n"
    "```python\nfrom collections import Counter\n\n\n"
    "def kelime_sikligi(metin: str) -> dict[str, int]:\n"
    "    return dict(Counter(metin.split()))\n```\n\n"
    "Örnek: `kelime_sikligi(\"a b a\")` → `{'a': 2, 'b': 1}`. Boş metin için boş "
    "sözlük döner; çok büyük dosyalarda parça parça okuma tercih edilmelidir."
)

GOOD_ARCH = """## Varsayımlar
- Günlük 5 milyon istek, tepe anında saniyede 300 istek.
- Bütçe aylık 2.000 USD, ekip 3 kişi.

## Önerilen mimari
- Yönlendirme katmanı: yönetilen API geçidi ve hız sınırlama.
- Okuma yolu: dağıtık önbellek + birincil anahtar indeksi.
- Yazma yolu: kuyruk üzerinden idempotent işleyici.

## Riskler ve ödünleşimler
- Önbellek tutarlılığı gecikmeyle ödünleşir; TTL ve geçersizleştirme gerekir.
- Kuyruk gecikmesi artarsa kullanıcıya "işleniyor" durumu gösterilmeli.

## Doğrulama adımları
- Yük testi ile p95 gecikme ölçümü.
- Kaos denemesi: önbellek düğümü düşürülür ve hata oranı izlenir.
"""

GOOD_TOOL = '{"tool": "hava_durumu", "arguments": {"sehir": "Ankara", "gun": "yarin"}}'


def test_every_family_maps_to_a_known_task_type():
    for family_id in family_ids():
        assert family_id in TASK_TYPES, family_id
    assert len(family_ids()) >= 10


def test_every_family_has_a_real_topic_bank():
    for family in FAMILIES.values():
        assert len(family.topics) >= 6, family.id
        assert len(set(family.topics)) == len(family.topics), f"duplicate topics in {family.id}"
        assert family.levels, family.id


def test_required_families_exist():
    assert {"coding", "reasoning", "turkish_advanced", "system_architecture"} <= set(FAMILIES)


def test_prompts_contain_no_secrets_or_invented_endpoints():
    for family in FAMILIES.values():
        for text in (
            family.drafter_system(),
            family.drafter_user("konu", "orta", 1),
            family.answer_system(),
            family.answer_user("örnek istek"),
        ):
            lowered = text.lower()
            for token in ("sk-", "api_key", "http://", "https://", "bearer"):
                assert token not in lowered, (family.id, token)


def test_prompts_are_formatted_without_placeholders():
    for family in FAMILIES.values():
        user = family.drafter_user("Konu X", "ileri", 3)
        assert "{task_type}" not in user and "{topic}" not in user
        assert "Konu X" in user and "ileri" in user
        answer = family.answer_user("İstek Y")
        assert "İstek Y" in answer


def test_specialised_suffixes_are_applied():
    assert "kod bloğu" in get_family("coding").answer_user("x")
    assert "Varsayımlar" in get_family("system_architecture").answer_user("x")
    assert '"tool"' in get_family("tool_calling").answer_user("x")
    assert "JSON" in get_family("tool_calling").answer_user("x")


def test_topic_selection_is_deterministic_and_spreads():
    family = get_family("coding")
    picks = {family.pick_topic(f"{family.id}-{i}", i) for i in range(50)}
    assert len(picks) >= 8, "topic bank is under-used"
    assert family.pick_topic("same", 7) == family.pick_topic("same", 7)


def test_level_selection_respects_family_levels():
    for family in FAMILIES.values():
        for i in range(20):
            assert family.pick_level(f"s{i}", i) in family.levels


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------
def test_code_block_validator():
    assert check_code_block("p", GOOD_CODE) == []
    assert check_code_block("p", "kod bloğu yok burada") == ["yanıt kod bloğu içermiyor"]
    assert check_code_block("p", "```sql\nselect 1\n```", language="python")
    assert check_code_block("p", "```python\n\n```") == ["kod bloğu boş"]


@pytest.mark.parametrize(
    "response",
    [
        GOOD_TOOL,
        f"```json\n{GOOD_TOOL}\n```",
        f"Araç çağrısı: {GOOD_TOOL}",
    ],
)
def test_tool_json_accepts_common_shapes(response):
    assert check_tool_json("p", response) == []


def test_tool_json_rejects_bad_shapes():
    assert check_tool_json("p", "araç çağırmayacağım")[0].startswith("geçerli")
    assert check_tool_json("p", '{"tool": "x"}')[0].startswith("araç çağrısında eksik")
    assert check_tool_json("p", '{"tool": "x", "arguments": "metin"}')


def test_min_sections():
    assert check_min_sections("p", GOOD_ARCH, minimum=3) == []
    assert check_min_sections("p", "düz metin", minimum=2)


@pytest.mark.parametrize(
    "leak",
    [
        "<|channel|>final<|message|>",
        "<|start|>assistant",
        "Here's a thinking process:",
        "iç düşünce sürecim şöyle",
    ],
)
def test_reasoning_leak_is_rejected(leak):
    assert check_no_reasoning_leak("p", f"Yanıt: {leak} devam")


def test_turkish_language_check():
    english = "The quick brown fox is in the garden and the dog is with the cat near that tree."
    assert check_turkish_language("p", english)
    turkish = (
        "Bu yanıt Türkçe olarak yazılmıştır ve içinde birçok bağlaç ile edat "
        "bulunur; bu nedenle dil kontrolünden geçer ve daha uzun olması gerekir."
    )
    assert check_turkish_language("p", turkish) == []
    assert check_turkish_language("p", "kısa") == []


def test_meta_preamble_is_a_soft_note():
    assert check_not_meta_preamble("p", "Elbette, işte istediğiniz yanıt.")


def test_family_problems_combine_validators():
    family = get_family("coding")
    problems = family.problems("p", "kısa ve kodsuz yanıt")
    assert any("kod bloğu" in p for p in problems)
    assert any("çok kısa" in p for p in problems)


def test_validate_generated_reports_shape():
    ok = validate_generated("system_architecture", "p", GOOD_ARCH)
    assert ok["ok"] is True and ok["problems"] == []
    bad = validate_generated("system_architecture", "p", "Elbette! Kısa cevap.")
    assert bad["ok"] is False and bad["problems"]
    assert "no_meta_preamble" in bad["soft_notes"]


def test_unknown_family_raises_with_helpful_message():
    with pytest.raises(KeyError, match="unknown task family"):
        get_family("astrology")


def test_resolve_families_order_and_subset():
    assert [f.id for f in resolve_families(["reasoning", "coding"])] == ["reasoning", "coding"]
    assert len(resolve_families(None)) == len(FAMILIES)


def test_family_summary_is_json_serialisable():
    summary = family_summary()
    assert json.loads(json.dumps(summary))
    assert {row["id"] for row in summary} == set(FAMILIES)
    assert len(HARD_VALIDATORS) >= 6
