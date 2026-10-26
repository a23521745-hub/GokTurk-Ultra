"""Task specifications for professional Turkish synthetic data.

Each *family* bundles everything needed to produce, validate and score one kind
of example:

* the drafter prompt (what kind of user request to invent),
* the solver prompt (how to answer),
* a curated Turkish topic bank (diversity without guessing),
* hard validators (a response failing these is discarded),
* soft checks (recorded as metadata, never silently dropped),
* a rough quality target used for quota planning.

Design principles
-----------------
* **No invented facts.** Prompt templates never contain model ids, endpoints or
  claims about provider capabilities.
* **Deterministic.** Everything here is pure data plus pure functions; the only
  randomness comes from a seeded ``random.Random`` in the planner.
* **Teachers are not oracles.** Validators check *form* (JSON shape, code
  fences, language, leakage markers). Semantic correctness requires the human
  review loop described in ``docs/architecture.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# shared prompt fragments
# ---------------------------------------------------------------------------
_ROLE = (
    "Sen GökTürk-Ultra için profesyonel Türkçe eğitim verisi üreten kıdemli bir "
    "uzmansın. Yanıtların Türkçe, teknik olarak doğru, eksiksiz ve doğrudan "
    "kullanıcının istediği biçimde olur. Gereksiz giriş cümlesi kurmaz, "
    "kendini tanıtmaz, \"elbette/tabii ki\" gibi dolgu ifadeler kullanmazsın. "
    "Bilmediğin bir şeyi uydurmazsın; emin olmadığında varsayımlarını açıkça yazar, "
    "gerektiğinde ek bilgi istersin. Akıl yürütme adımlarını yanıt metnine "
    "dökmezsin; sonucu gerekçeleriyle birlikte, düzenli biçimde verirsin."
)

_SOLVER_RULES = (
    "Kurallar:\n"
    "- Yanıt tamamen Türkçe olacak (kod, komut ve özel adlar hariç).\n"
    "- İstekte belirtilen biçim, uzunluk ve madde sayısı kısıtlarına birebir uy.\n"
    "- Doğrudan yanıta başla; giriş/kapanış nezaket cümlesi yazma.\n"
    "- Uydurma kaynak, DOI, kanun numarası veya API imzası verme.\n"
    "- Emin olmadığın bilgiyi kesinmiş gibi sunma; varsayımı belirt.\n"
)

_DRAFTER_RULES = (
    "Kurallar:\n"
    "- İstek, gerçek bir kullanıcının yazacağı doğallıkta olacak.\n"
    "- İstek, cevabı verilecek asıl görevi içerecek; role-play yönergesi olmayacak.\n"
    "- Somut bir çıktı biçimi ve kapsam isteyecek (uzunluk, madde sayısı, dil, seviye).\n"
    "- Daha önce üretilmiş örneklerden belirgin biçimde farklı olacak.\n"
    "- Özetleme, yeniden yazma ve çeviri görevlerinde kaynak metnin tamamını <kaynak>...</kaynak> içinde ver; dışarıdan metin bekleme.\n"
    "- Yalnızca JSON döndür: {{\"prompt\": \"<Türkçe kullanıcı isteği>\"}}\n"
)

_DRAFTER_SYSTEM = (
    "Sen Türkçe bir veri seti tasarımcısısın. Görevin, verilen görev türü ve konu "
    "için tek turlu, net, gerçekçi bir Türkçe kullanıcı isteği yazmaktır. "
    "Yalnızca istenen JSON şemasında yanıt verirsin."
)

_DRAFTER_USER_TMPL = """Görev türü: {task_type}
Konu: {topic}
Zorluk: {level}
Varyasyon: {variant}
Hedef kitle: {audience}

Şu şemada SADECE JSON döndür:
{{"prompt": "<Türkçe kullanıcı isteği>"}}

{extra}
{rules}"""

_ANSWER_USER_TMPL = """Kullanıcı isteği:
\"\"\"
{prompt}
\"\"\""""


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------
_CODE_FENCE_RE = re.compile(r"```([A-Za-z0-9+#._-]*)\s*\n(.*?)```", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{2,4}\s+\S+|\d+[.)]\s+\S+)", re.MULTILINE)
_LEAK_MARKERS = (
    "<|channel|>",
    "<|start|>",
    "<|end|>",
    "<|assistant|>",
    "reasoning:",
    "analysis channel",
    "here's a thinking process",
    "iç düşünce sürecim",
)
_TR_STOPWORDS = {
    "ve", "bir", "bu", "için", "ile", "olarak", "da", "de", "çok", "daha",
    "gibi", "ancak", "veya", "kadar", "sonra", "önce", "gerekir", "olur",
}
_EN_STOPWORDS = {"the", "and", "is", "are", "of", "to", "in", "that", "for", "with"}


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def check_code_block(prompt: str, response: str, *, language: str | None = None) -> list[str]:
    blocks = _CODE_FENCE_RE.findall(response)
    if not blocks:
        return ["yanıt kod bloğu içermiyor"]
    if language:
        languages = {lang.lower() for lang, _ in blocks if lang}
        if languages and language.lower() not in languages:
            return [f"kod bloğu dili {language!r} değil: {sorted(languages)}"]
    for _, body in blocks:
        if not body.strip():
            return ["kod bloğu boş"]
    return []


TOOL_KEYS = ("tool", "arguments")


def check_tool_json(prompt: str, response: str, *, keys: Sequence[str] = TOOL_KEYS) -> list[str]:
    candidates = [response.strip()]
    candidates += [f"{{{body.strip()}}}" for _, body in _CODE_FENCE_RE.findall(response)]
    box = re.search(r"\{.*\}", response, re.DOTALL)
    if box:
        candidates.append(box.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, Mapping):
            continue
        missing = [k for k in keys if k not in obj]
        if missing:
            return [f"araç çağrısında eksik anahtar: {missing}"]
        if not isinstance(obj.get("arguments", {}), Mapping):
            return ["arguments bir nesne olmalı"]
        return []
    return ["geçerli araç çağrısı JSON'u bulunamadı"]


def check_min_sections(prompt: str, response: str, *, minimum: int = 2) -> list[str]:
    sections = _HEADING_RE.findall(response)
    if len(sections) < minimum:
        return [f"yanıtta en az {minimum} başlık/madde bekleniyordu ({len(sections)} bulundu)"]
    return []


def check_no_reasoning_leak(prompt: str, response: str) -> list[str]:
    lowered = response.casefold()
    for marker in _LEAK_MARKERS:
        if marker in lowered:
            return [f"yanıtta iç akıl yürütme/özel token sızıntısı: {marker!r}"]
    return []


def check_turkish_language(prompt: str, response: str, *, min_ratio: float = 0.12) -> list[str]:
    words = _words(response)
    if len(words) < 12:
        return []
    tr = sum(1 for w in words if w in _TR_STOPWORDS)
    en = sum(1 for w in words if w in _EN_STOPWORDS)
    if tr == 0 and en >= 3:
        return ["yanıt Türkçe görünmüyor"]
    if len(words) >= 40 and tr / len(words) < min_ratio and en > tr:
        return ["Türkçe oranı çok düşük"]
    return []


def check_not_meta_preamble(prompt: str, response: str) -> list[str]:
    head = response.strip().casefold()[:120]
    for opener in ("elbette", "tabii ki", "memnuniyetle", "işte istediğiniz", "sure,", "certainly"):
        if head.startswith(opener):
            return [f"dolgu girişiyle başlıyor: {opener!r}"]
    return []


def check_length_window(
    prompt: str, response: str, *, minimum: int = 200, maximum: int = 12000
) -> list[str]:
    if len(response) < minimum:
        return [f"yanıt çok kısa ({len(response)} < {minimum})"]
    if len(response) > maximum:
        return [f"yanıt çok uzun ({len(response)} > {maximum})"]
    return []


HardValidator = Callable[..., list[str]]

HARD_VALIDATORS: dict[str, HardValidator] = {
    "code_block": check_code_block,
    "tool_json": check_tool_json,
    "min_sections": check_min_sections,
    "no_leak": check_no_reasoning_leak,
    "turkish": check_turkish_language,
    "length_window": check_length_window,
}

SOFT_CHECKS: dict[str, HardValidator] = {
    "no_meta_preamble": check_not_meta_preamble,
}


# ---------------------------------------------------------------------------
# family definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskFamily:
    id: str
    label: str
    description: str
    topics: tuple[str, ...]
    levels: tuple[str, ...] = ("temel", "orta", "ileri")
    audience: str = "genel teknik kullanıcı"
    min_response_chars: int = 200
    max_response_chars: int = 12000
    validators: tuple[str, ...] = ("no_leak", "turkish")
    validator_kwargs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: teacher ``role`` values this family prefers (see configs/teachers.yaml)
    preferred_roles: tuple[str, ...] = ("general",)
    drafter_extra: str = ""
    answer_suffix: str = ""
    weight: float = 1.0

    # -- prompt construction ------------------------------------------------
    def drafter_system(self) -> str:
        return _DRAFTER_SYSTEM

    def drafter_user(self, topic: str, level: str, variant: int) -> str:
        return _DRAFTER_USER_TMPL.format(
            task_type=self.id,
            topic=topic,
            level=level,
            variant=variant,
            audience=self.audience,
            extra=self.drafter_extra.strip() + ("\n" if self.drafter_extra else ""),
            rules=_DRAFTER_RULES,
        )

    def answer_system(self) -> str:
        return _ROLE

    def answer_user(self, prompt: str) -> str:
        body = _ANSWER_USER_TMPL.format(prompt=prompt) + "\n" + _SOLVER_RULES
        if self.answer_suffix:
            body += "\n" + self.answer_suffix.strip() + "\n"
        return body

    # -- validation ---------------------------------------------------------
    def problems(self, prompt: str, response: str) -> list[str]:
        problems: list[str] = []
        for name in self.validators:
            fn = HARD_VALIDATORS[name]
            kwargs = dict(self.validator_kwargs.get(name, {}))
            if name == "length_window":
                kwargs.setdefault("minimum", self.min_response_chars)
                kwargs.setdefault("maximum", self.max_response_chars)
            problems.extend(fn(prompt, response, **kwargs))
        return problems

    def soft_notes(self, prompt: str, response: str) -> dict[str, list[str]]:
        notes: dict[str, list[str]] = {}
        for name, fn in SOFT_CHECKS.items():
            found = fn(prompt, response)
            if found:
                notes[name] = found
        return notes

    def pick_topic(self, seed_id: str, variant: int) -> str:
        digest = hashlib.sha1(f"{self.id}|{seed_id}|{variant}".encode()).hexdigest()
        index = int(digest[:8], 16) % len(self.topics)
        return self.topics[index]

    def pick_level(self, seed_id: str, variant: int) -> str:
        digest = hashlib.sha1(f"level|{self.id}|{seed_id}|{variant}".encode()).hexdigest()
        return self.levels[int(digest[:8], 16) % len(self.levels)]


# ---------------------------------------------------------------------------
# the families
# ---------------------------------------------------------------------------
CODING_SUFFIX = (
    "Bu bir kodlama görevidir. Yanıtında:\n"
    "1) Çalışır, kopyalanabilir kod bloğu (dil etiketiyle) ver.\n"
    "2) Kodun altında kısa bir açıklama ve en az bir örnek girdi/çıktı göster.\n"
    "3) Kenar durumlarını (boş girdi, hatalı tip vb.) belirt.\n"
    "Kod bloğu dışına uzun teorik anlatım koyma."
)

ARCH_SUFFIX = (
    "Bu bir sistem mimarisi senaryosudur. Yanıtı şu dört başlıkla yapılandır:\n"
    "## Varsayımlar\n## Önerilen mimari\n## Riskler ve ödünleşimler\n## Doğrulama adımları\n"
    "Somut bileşen adları, veri akışı ve ölçek varsayımları (RPS, veri boyutu, bütçe) belirt. "
    "Teknoloji önerirken gerekçe ver; moda terimleri gerekçesiz kullanma."
)

REASONING_SUFFIX = (
    "Bu bir akıl yürütme görevidir. Yanıtında:\n"
    "1) Problemin verilenlerini ve varsayımlarını yaz.\n"
    "2) Çözüm adımlarını numaralı ve denetlenebilir biçimde göster.\n"
    "3) Sonucu net biçimde belirt ve sağlamasını yap.\n"
    "Yüzeysel tahmin yürütme; her adım bir öncekine dayansın."
)

TURKISH_SUFFIX = (
    "Bu ileri düzey Türkçe görevidir. Yanıtında dil bilgisi kurallarına, yazım "
    "kılavuzuna (TDK) ve resmî yazışma geleneğine uy. Gerektiğinde terimin "
    "Türkçe karşılığını parantez içinde ver. Uydurma kaynak/kural numarası verme."
)

TOOL_SUFFIX = (
    "Bu bir araç çağırma görevidir. Yanıt SADECE şu şemaya uyan tek bir JSON "
    "nesnesi olsun:\n"
    '{"tool": "<araç_adı>", "arguments": {<argümanlar>}}\n'
    "JSON dışında hiçbir metin, markdown veya açıklama ekleme. "
    "Argüman adları istekteki araç imzasıyla birebir aynı olmalı."
)

INSTRUCTION_SUFFIX = (
    "Bu bir talimat takibi görevidir. İstekteki tüm kısıtları (madde sayısı, "
    "kelime sınırı, yasaklı kelime, biçim) tek tek kontrol et ve hepsine uy. "
    "Kısıtı ihlal etmek yerine kısa ve öz yaz."
)

SUMMARY_SUFFIX = (
    "Bu bir özetleme görevidir. Yanıt verilen metnin kapsamını aşmayacak, "
    "uydurma bilgi eklemeyecek ve istenen uzunluk sınırında olacak. "
    "Gerektiğinde madde işaretleri kullan."
)

REWRITE_SUFFIX = (
    "Bu bir yeniden yazma görevidir. Anlamı koru; cümle yapısını, tonu ve "
    "uzunluğu isteğe göre değiştir. Değiştirdiğin önemli ifadeleri kısaca belirt."
)

TRANSLATION_SUFFIX = (
    "Bu bir çeviri görevidir. Yanıt yalnızca çeviri metnini içersin. "
    "Teknik terimleri tutarlı çevir; ilk kullanımda özgün terimi parantez içinde ver. "
    "Uydurma çeviri notu veya gerekçe ekleme."
)


FAMILIES: dict[str, TaskFamily] = {}


def _register(family: TaskFamily) -> TaskFamily:
    FAMILIES[family.id] = family
    return family


_register(
    TaskFamily(
        id="coding",
        label="Kodlama",
        description=(
            "Algoritma, hata ayıklama, veri işleme, SQL ve API tasarımı içeren "
            "çalıştırılabilir kod görevleri."
        ),
        topics=(
            "Python'da büyük bir metin dosyasında kelime sıklığı çıkarma (bellek dostu)",
            "Python'da tarih aralığına göre gruplayıp toplam hesaplama",
            "SQL ile müşteri başına son siparişi bulma (pencere fonksiyonları)",
            "Python'da bir REST API çağrısını yeniden deneme (retry) ve zaman aşımı ile yazma",
            "Python'da iç içe sözlükten alan çıkarma ve varsayılan değer yönetimi",
            "Python'da büyük CSV dosyasını parça parça okuyup özet çıkarma",
            "Python'da bir metni normalize edip Türkçe karakterleri koruyarak arama yapma",
            "SQL'de N+1 sorgu problemini tek sorguya indirme",
            "Python'da asenkron toplu HTTP isteklerini sınırlı eşzamanlılıkla çalıştırma",
            "Python'da bir sözlük listesini birden çok anahtara göre sıralama",
            "Python'da önbellekleme (LRU) ile yavaş fonksiyonu hızlandırma",
            "Python'da JSON şemasına göre doğrulama ve anlaşılır hata mesajı üretme",
            "Python'da log dosyasından hata satırlarını ayıklayıp sayma",
            "Python'da bir sınıfı test edilebilir hale getirmek için bağımlılık enjeksiyonu",
            "SQL'de yavaş bir JOIN sorgusunu indekslerle iyileştirme",
            "Python'da birim testi yazma (sınır durumları dahil)",
        ),
        validators=("code_block", "no_leak", "turkish", "length_window"),
        validator_kwargs={"code_block": {"language": "python"}},
        min_response_chars=250,
        preferred_roles=("code",),
        answer_suffix=CODING_SUFFIX,
        drafter_extra="İstek, hangi dilde (Python/SQL) yazılacağını ve beklenen çıktıyı belirtsin.",
        weight=1.2,
    )
)

_register(
    TaskFamily(
        id="coding_debug",
        label="Hata ayıklama",
        description="Bozuk kod parçalarının tanılanması ve düzeltilmesi.",
        topics=(
            "Python'da kapalı değişken (closure) kaynaklı beklenmeyen sonuç",
            "Python'da değiştirilebilir varsayılan argüman hatası",
            "SQL'de NULL karşılaştırmasından kaynaklanan eksik satırlar",
            "Python'da dosya tanıtıcısı sızıntısı (open/close dengesi)",
            "Python'da paralel çalışan iş parçacıklarında yarış durumu (race condition)",
            "Python'da yuvarlama hatası ve Decimal kullanımı",
            "Python'da liste üzerinde dönerken eleman silme hatası",
            "SQL'de GROUP BY olmadan seçilen toplulaştırılmamış sütun",
            "Python'da zaman dilimi (timezone) karışıklığı",
            "Python'da sonsuz özyineleme ve önbellek çözümü",
        ),
        validators=("code_block", "no_leak", "turkish", "length_window"),
        validator_kwargs={"code_block": {"language": "python"}},
        min_response_chars=250,
        preferred_roles=("code",),
        answer_suffix=CODING_SUFFIX + "\nÖnce hatanın nedenini açıkla, sonra düzeltilmiş kodu ver.",
        drafter_extra="İstek, bozuk bir kod parçası içermeli ve hatanın ne olduğunu sormadan düzeltme istemeli.",
        weight=0.8,
    )
)

_register(
    TaskFamily(
        id="reasoning",
        label="Mantık ve akıl yürütme",
        description="Sayısal problemler, mantık çıkarımları ve ödünleşim analizleri.",
        topics=(
            "Bütçe ve mesafe verilerinden en ekonomik ulaşım seçeneğini bulma",
            "İşçi-havuz-verim problemi (birleşik oranlar)",
            "Verilen öncüllerden geçerli çıkarım kurma ve geçersiz olanı ayıklama",
            "Yüzde artış/azalış zincirlemesi ve gerçek getiri hesabı",
            "Sınırlı kaynakla iş önceliklendirme (kısıt altında sıralama)",
            "Olasılık: bağımlı ve bağımsız olay ayrımı",
            "Bir sistem arızasının kök nedenini zaman çizelgesinden çıkarma",
            "İki teklifin 3 yıllık toplam maliyet karşılaştırması (amortisman dahil)",
            "Grafikteki çelişkili verilerden tutarsızlığı bulma",
            "Oran-orantı ile ölçekleme ve birim dönüşümü",
            "Sıralama kısıtlarıyla (A, B'den önce vb.) uygun düzen bulma",
            "Ağırlıklı ortalama ile not/performans hesabı",
        ),
        validators=("no_leak", "turkish", "length_window", "min_sections"),
        validator_kwargs={"min_sections": {"minimum": 2}},
        min_response_chars=300,
        preferred_roles=("reasoning",),
        answer_suffix=REASONING_SUFFIX,
        drafter_extra="İstek, tüm sayısal verileri ve kısıtları içermeli; çözüm için ek bilgi gerekmemeli.",
        weight=1.2,
    )
)

_register(
    TaskFamily(
        id="turkish_advanced",
        label="İleri düzey Türkçe",
        description=(
            "Dil bilgisi, yazım, üslup, resmî yazışma, hukuk/idari dil ve "
            "terminoloji alanlarında uzman düzeyinde Türkçe görevler."
        ),
        topics=(
            "Resmî bir kuruma yazılacak itiraz dilekçesinin yapısı ve üslubu",
            "Bir sözleşme maddesindeki belirsiz ifadelerin netleştirilmesi",
            "Akademik bir paragrafta gereksiz sözcüklerin ayıklanması",
            "Teknik bir terimin Türkçe karşılığının tartışılması (ör. 'deployment')",
            "Noktalama işaretlerinin anlam değiştirdiği örneklerin açıklanması",
            "Bir metnin resmî üsluba çevrilmesi (günlük dilden kurumsal dile)",
            "Yazım kılavuzuna göre düzeltme gerektiren bir paragraf",
            "Devrik cümlelerin kurallı yapıya çevrilmesi",
            "Bir duyurunun farklı hedef kitleler için yeniden yazılması",
            "Eş anlamlı sözcükler arasındaki anlam/nüans farkları",
            "Bir metinde tutarsız terim kullanımının düzeltilmesi",
            "Türkçede edilgen/genel özne yapılarının kullanım farkları",
            "Bir rapor özetinin yönetici özetine dönüştürülmesi",
            "Yabancı sözcük yerine öneri geliştirme ve gerekçelendirme",
        ),
        levels=("orta", "ileri"),
        audience="editör, hukukçu veya kurumsal iletişim uzmanı",
        validators=("no_leak", "turkish", "length_window"),
        min_response_chars=300,
        preferred_roles=("general", "reasoning"),
        answer_suffix=TURKISH_SUFFIX,
        weight=1.0,
    )
)

_register(
    TaskFamily(
        id="system_architecture",
        label="Sistem mimarisi",
        description=(
            "Ölçeklenebilirlik, veri modelleme, güvenlik, maliyet, olay müdahale "
            "ve mimari karar kaydı (ADR) senaryoları."
        ),
        topics=(
            "Dakikada 50 bin isteği karşılayacak kısa bağlantı servisi tasarımı",
            "Çok kiracılı (multi-tenant) bir SaaS için veri izolasyonu stratejisi",
            "Olay (event) tabanlı sipariş akışında idempotency ve tekrar işleme",
            "Okuma ağırlıklı bir katalog servisi için önbellekleme katmanı",
            "Üç bölgeli (multi-region) dağıtımda veri tutarlılığı ve gecikme ödünleşimi",
            "Kimlik doğrulama ve yetkilendirme (RBAC/ABAC) mimarisi",
            "Toplu veri içe aktarma (ETL) hattında yeniden çalıştırılabilirlik",
            "Üretim ortamında 5xx artışı için kök neden analizi ve müdahale planı",
            "Servisler arası iletişimde senkron REST yerine kuyruk kullanma kararı (ADR)",
            "Maliyet optimizasyonu: gözlemlenebilirlik verisinin saklama politikası",
            "Ağır ML çıkarım servisinin GPU maliyetini düşürme stratejisi",
            "Sıfır kesinti hedefli şema göçü (migration) planı",
            "Mikroservis ayrıştırmasının sınırlarını belirleme (bounded context)",
            "Yedekleme ve felaket kurtarma (RTO/RPO) tasarımı",
            "Kritik akışlarda yük testi ve kapasite planlaması",
            "Gizli anahtar yönetimi ve rotasyon stratejisi",
            "Journaling/audit log mimarisi ve değiştirilemezlik",
            "Sürüm çıkarma (deploy) stratejisi: canary vs blue-green",
        ),
        audience="kıdemli yazılım mimarı veya platform ekibi",
        validators=("no_leak", "turkish", "length_window", "min_sections"),
        validator_kwargs={"min_sections": {"minimum": 3}},
        min_response_chars=500,
        max_response_chars=14000,
        preferred_roles=("reasoning", "general"),
        answer_suffix=ARCH_SUFFIX,
        drafter_extra=(
            "İstek, ölçek (kullanıcı/RPS/veri boyutu), kısıt (bütçe, ekip, mevzuat) "
            "ve başarı ölçütü içermeli."
        ),
        weight=1.2,
    )
)

_register(
    TaskFamily(
        id="tool_calling",
        label="Araç çağırma",
        description="Fonksiyon çağrısı biçiminde tek JSON nesnesi üreten görevler.",
        topics=(
            "hava_durumu(sehir, gun) aracıyla belirli şehir için tahmin isteme",
            "takvim_ekle(baslik, tarih, saat) aracıyla toplantı oluşturma",
            "arama(sorgu) aracıyla güncel bir bilgiyi bulma",
            "hesap_makinesi(ifade) aracıyla çok adımlı hesap yapma",
            "dosya_oku(yol) aracıyla rapor içeriğini okuma",
            "bilet_ara(kalkis, varis, tarih) aracıyla uygun sefer bulma",
            "stok_kontrol(urun_kodu) aracıyla depo durumunu sorgulama",
            "ceviri_yap(metin, hedef_dil) aracıyla çeviri isteme",
            "hatirlatici_ekle(baslik, zaman) aracıyla hatırlatıcı kurma",
            "kullanici_bilgisi(kullanici_id) aracıyla hesap ayrıntısı çekme",
        ),
        levels=("temel", "orta"),
        audience="asistan/ajan geliştiricisi",
        validators=("tool_json", "no_leak", "length_window"),
        validator_kwargs={"length_window": {"minimum": 10, "maximum": 800}},
        min_response_chars=10,
        max_response_chars=800,
        preferred_roles=("code", "general"),
        answer_suffix=TOOL_SUFFIX,
        drafter_extra=(
            "İstek, araç imzasını (ad + argümanlar + tipleri) ve kullanıcının "
            "istediği işlemi birlikte içermeli."
        ),
        weight=0.8,
    )
)

_register(
    TaskFamily(
        id="instruction_following",
        label="Talimat takibi",
        description="Kısıtlı biçim, uzunluk ve içerik kurallarına uyan yanıtlar.",
        topics=(
            "Tam olarak üç madde halinde çalışma planı yazma",
            "30 kelimeyi geçmeyen bir iş e-postası taslağı",
            "Yasaklı kelime kullanmadan bir süreci özetleme",
            "Belirli bir sıralamaya uyan adım listesi",
            "Yalnızca büyük harf ve noktalama olmadan bir uyarı metni",
            "İki cümleyi geçmeyen bir yönetici özeti",
            "Madde işareti kullanmadan üç öneri",
            "Belirli bir kelimeyle başlayan bir açıklama",
            "Tabloda üç satırlık karşılaştırma",
            "Her maddesi fiille başlayan beş kontrol listesi",
        ),
        levels=("temel", "orta"),
        validators=("no_leak", "turkish", "length_window"),
        min_response_chars=60,
        max_response_chars=2000,
        preferred_roles=("general",),
        answer_suffix=INSTRUCTION_SUFFIX,
        drafter_extra="İstek, en az iki ölçülebilir kısıt içermeli (madde sayısı, kelime sınırı, yasak kelime, biçim).",
        weight=0.9,
    )
)

_register(
    TaskFamily(
        id="summarization",
        label="Özetleme",
        description="Verilen metnin kapsamına bağlı kalarak özet çıkarma.",
        topics=(
            "Uzun bir teknik raporun yönetici özeti",
            "Bir toplantı notunun karar ve aksiyon özeti",
            "Bir haber metninin üç maddelik özeti",
            "Bir hata raporunun (postmortem) kısa özeti",
            "Bir sözleşme taslağının maddeler halinde özeti",
            "Bir araştırma özetinin 120 kelimelik Türkçe özeti",
            "Destek talepleri listesinin ortak sorunlar özeti",
            "Bir ürün duyurusunun hedef kitleye uygun özeti",
        ),
        validators=("no_leak", "turkish", "length_window"),
        min_response_chars=150,
        max_response_chars=3000,
        preferred_roles=("general",),
        answer_suffix=SUMMARY_SUFFIX,
        drafter_extra="İstek, özetlenecek gerçek kaynak metnin tamamını <kaynak>...</kaynak> etiketleri arasında içermeli; yalnızca metin türü belirtmek yeterli değil. İstenen uzunluğu da belirtmeli.",
        weight=0.7,
    )
)

_register(
    TaskFamily(
        id="rewrite",
        label="Yeniden yazma",
        description="Ton, uzunluk ve hedef kitleye göre metni yeniden üretme.",
        topics=(
            "Teknik bir metni teknik olmayan bir kitleye uyarlama",
            "Aşırı uzun bir cümleyi bölerek sadeleştirme",
            "Günlük dildeki bir metni resmî üsluba çevirme",
            "Bir hata mesajını kullanıcı dostu hale getirme",
            "Bir metni daha kısa ve vurucu hale getirme",
            "Bir paragrafı madde işaretlerine dönüştürme",
            "Bir metni ekip içi paylaşıma uygun tona çevirme",
            "Karmaşık bir paragrafı yeni başlayanlar için sadeleştirme",
        ),
        validators=("no_leak", "turkish", "length_window"),
        min_response_chars=150,
        max_response_chars=4000,
        preferred_roles=("general",),
        answer_suffix=REWRITE_SUFFIX,
        drafter_extra="İstek, kaynak metni (kısa) ve hedef tonu/uzunluğu belirtmeli.",
        weight=0.6,
    )
)

_register(
    TaskFamily(
        id="translation",
        label="Çeviri",
        description="İngilizce teknik metinlerin akıcı Türkçeye çevrilmesi.",
        topics=(
            "Dağıtık sistemlerde tutarlılık üzerine teknik bir paragraf",
            "Makine öğrenmesi model kartından bir bölüm",
            "Bir API dokümantasyonundan kimlik doğrulama bölümü",
            "Bir güvenlik açığı duyurusunun özeti",
            "Bir yazılım lisansı sorusu ve yanıtı",
            "Veri gizliliği (KVKK/GDPR) üzerine kısa bir metin",
        ),
        levels=("orta", "ileri"),
        validators=("no_leak", "turkish", "length_window"),
        min_response_chars=120,
        max_response_chars=4000,
        preferred_roles=("general",),
        answer_suffix=TRANSLATION_SUFFIX,
        drafter_extra="İstek, çevrilecek İngilizce metni (3-6 cümle) içermeli.",
        weight=0.5,
    )
)


# ---------------------------------------------------------------------------
# public helpers
# ---------------------------------------------------------------------------
def family_ids() -> tuple[str, ...]:
    return tuple(FAMILIES)


def get_family(family_id: str) -> TaskFamily:
    try:
        return FAMILIES[family_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown task family {family_id!r}; known: {sorted(FAMILIES)}"
        ) from exc


def resolve_families(selected: Iterable[str] | None = None) -> list[TaskFamily]:
    """Return families in a stable order, optionally restricted to ``selected``."""
    if not selected:
        return [FAMILIES[key] for key in FAMILIES]
    out = []
    for key in selected:
        out.append(get_family(key))
    return out


def family_summary() -> list[dict[str, Any]]:
    return [
        {
            "id": family.id,
            "label": family.label,
            "topics": len(family.topics),
            "levels": list(family.levels),
            "validators": list(family.validators),
            "weight": family.weight,
        }
        for family in FAMILIES.values()
    ]


def validate_generated(family_id: str, prompt: str, response: str) -> dict[str, Any]:
    """Run hard validators + soft checks for one generated pair."""
    family = get_family(family_id)
    problems = family.problems(prompt, response)
    notes = family.soft_notes(prompt, response)
    return {"ok": not problems, "problems": problems, "soft_notes": notes}
