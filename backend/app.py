import os
import sys
import ssl
import re
import nltk
import json # Added for better error handling/NLTK path

# ── Python 3.13 / Torch compatibility ────────────────────────────────────────
os.environ["TORCH_DYNAMO_DISABLE"]  = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["CURL_CA_BUNDLE"]        = ""
os.environ["PYTHONHTTPSVERIFY"]     = "0"

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

try:
    import torch
    # Ensure torch is available for CPU only to keep size down
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.suppress_errors = True
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU
except ImportError:
    pass

# ── NLTK Data Downloads for Vercel ───────────────────────────────────────────
# Vercel needs NLTK data to be downloaded into a writable directory during build.
# We'll put it in a temporary directory relative to the function.
NLTK_DATA_DIR = os.path.join("/tmp", "nltk_data") # Vercel /tmp is writable
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# Attempt downloads; these will run during Vercel build if not already present
try:
    nltk.download("punkt", quiet=True, download_dir=NLTK_DATA_DIR)
    nltk.download("punkt_tab", quiet=True, download_dir=NLTK_DATA_DIR)
    nltk.download("stopwords", quiet=True, download_dir=NLTK_DATA_DIR)
except Exception as e:
    print(f"NLTK download failed (might be cached): {e}")

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models…")

# --- REPLACED LARGE MODELS WITH SMALLER ALTERNATIVES ---
# Smaller model for sentiment analysis (~150MB)
sentiment_pipe = pipeline(
    "text-classification",
    model="finiteautomata/bertweet-base-sentiment-pipeline",
    device=-1, # -1 for CPU, 0 for GPU
    truncation=True,
    max_length=128, # Reduced max_length for smaller model efficiency
)
# Smaller model for zero-shot classification (~100MB)
zero_shot_pipe = pipeline(
    "zero-shot-classification",
    model="typeform/mobilebert-uncased-mnli",
    device=-1, # -1 for CPU, 0 for GPU
    truncation=True,
    max_length=128, # Reduced max_length for smaller model efficiency
)

print("Models loaded ✅")

# =============================================================================
# KNOWLEDGE BASES
# =============================================================================

# Phrases that are definitively false — always trigger max hallucination
HARD_FALSEHOODS = [
    "moon is made of cheese", "moon is made of green cheese",
    "made of green cheese", "made of cheese",
    "flat earth", "earth is flat", "the earth is flat",
    "moon landing was fake", "moon landing is fake", "never landed on the moon",
    "vaccines cause autism", "vaccine causes autism",
    "5g causes cancer", "5g spreads covid",
    "covid is a hoax", "coronavirus is a hoax", "pandemic is a hoax",
    "climate change is a hoax", "global warming is a hoax",
    "earth is 6000 years old", "dinosaurs never existed",
    "chemtrails are poison", "bill gates microchip",
    "sun revolves around earth", "earth is the center",
    "evolution is false", "evolution is a lie",
    "the holocaust never happened", "holocaust is a hoax",
    "obama was born in kenya",
]

# Phrases that deny scientific consensus — strong hallucination signal
ANTI_SCIENCE_PHRASES = [
    "scientists are liars", "scientists are evil", "scientists lie",
    "scientists are corrupt", "science is fake", "science is a lie",
    "doctors are liars", "doctors are evil", "doctors lie",
    "experts are liars", "experts are lying", "experts are corrupt",
    "research is fake", "studies are fake", "data is fake",
    "mainstream science", "fake science", "pseudoscience is real",
    "nasa lies", "nasa is lying", "nasa faked",
    "big pharma conspiracy", "medical conspiracy",
]

# Phrases that make confident factual claims with no basis
UNSOURCED_CLAIM_PHRASES = [
    "studies show", "research shows", "scientists say", "experts say",
    "experts agree", "scientists confirm", "it has been proven",
    "it is well known", "everyone knows", "it is common knowledge",
    "according to experts", "data shows", "statistics show",
    "evidence shows", "science confirms", "research confirms",
    "medical experts say", "doctors confirm",
]

# Absolute / overconfident language
ABSOLUTE_PHRASES = [
    "100%", "100 percent", "without any doubt", "without doubt",
    "definitely", "absolutely proven", "proven fact", "it is a fact",
    "scientifically proven", "research proves", "guaranteed",
    "without exception", "in every case", "no question about it",
    "undeniably", "undoubtedly", "certainly true",
    "nobody can deny", "everyone knows this",
]

# Hedging language (reduces hallucination score)
HEDGE_PHRASES = [
    "perhaps", "maybe", "possibly", "might", "could", "may", "seems",
    "appears", "likely", "probably", "approximately", "roughly",
    "suggests", "indicates", "allegedly", "reportedly",
    "according to", "research suggests", "studies indicate",
    "it is possible", "there is evidence", "some research",
]

# Emotional / inflammatory words
EMOTIONAL_WORDS = [
    "idiot", "stupid", "moron", "fool", "imbecile", "lunatic", "insane",
    "evil", "corrupt", "criminal", "disgusting", "horrifying", "terrible",
    "shocking", "outrageous", "appalling", "catastrophic", "devastating",
    "dangerous", "deadly", "fatal", "toxic", "poisonous", "destroying",
    "destroy", "brainwash", "manipulate", "control", "enslave",
    "hate", "despise", "loathe", "rage", "furious", "disgusted",
    "sick", "pathetic", "coward", "traitor", "enemy",
]

# Ad hominem / personal attacks
AD_HOMINEM = [
    "only an idiot", "only a fool", "only a moron", "only stupid people",
    "you have to be stupid", "you must be dumb", "are you blind",
    "wake up sheeple", "wake up sheep", "sheeple", "brainwashed",
    "mindless", "sheep who believe", "gullible fools",
    "total idiot", "complete idiot", "utter fool",
]

# Conspiracy / manipulation language
CONSPIRACY_PHRASES = [
    "they want to destroy", "want to control", "want to enslave",
    "destroy our minds", "control our minds", "control the population",
    "new world order", "deep state", "they are hiding", "cover up",
    "they don't want you to know", "hidden truth", "real truth",
    "mainstream media lies", "government lies", "government is hiding",
    "wake up", "open your eyes", "they are lying to us",
    "agenda", "propaganda", "illuminati", "elites control",
]

# Persuasion / manipulation tactics
PERSUASION_TACTICS = [
    "you must believe", "you need to wake up", "open your eyes",
    "don't be fooled", "don't let them", "they want you to",
    "resist the", "fight the", "stand against",
]

# Citation patterns
CITATION_PATTERNS = [
    r"\[[\d,\s]+\]",
    r"\(\w[\w\s\-]+,\s*\d{4}[a-z]?\)",
    r"https?://[^\s]+",
    r"doi:\s*10\.\d{4}",
    r"et al\.",
    r"source\s*:",
    r"ref\.",
]

# Formal academic connectors (increases trust)
FORMAL_CONNECTORS = [
    "furthermore", "moreover", "however", "nevertheless", "consequently",
    "therefore", "thus", "hence", "in contrast", "on the other hand",
    "in addition", "subsequently", "as a result", "in conclusion",
    "notwithstanding", "in summary", "to conclude",
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def tokenize_sentences(text):
    return nltk.sent_tokenize(text)

def word_count(text):
    return len(re.findall(r"\b\w+\b", text))

def avg_sentence_length(text):
    sents = tokenize_sentences(text)
    if not sents: return 0
    return sum(len(s.split()) for s in sents) / len(sents)

def lexical_diversity(text):
    words = re.findall(r"\b\w+\b", text.lower())
    return round(len(set(words)) / len(words), 3) if words else 0

def count_citations(text):
    return sum(1 for p in CITATION_PATTERNS if re.search(p, text, re.IGNORECASE))

def find_phrases(text_lower, phrase_list):
    return [p for p in phrase_list if p in text_lower]

def count_phrases(text_lower, phrase_list):
    return len(find_phrases(text_lower, phrase_list))

def chunk_text(text, max_chars=450):
    sentences = tokenize_sentences(text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            if current: chunks.append(current.strip())
            current = s
    if current: chunks.append(current.strip())
    return chunks or [text[:max_chars]]

def formality_score(text):
    tl = text.lower()
    fc = count_phrases(tl, FORMAL_CONNECTORS)
    al = min(avg_sentence_length(text), 30) / 30
    ld = lexical_diversity(text)
    return round(min(1.0, fc * 0.15 + al * 0.4 + ld * 0.45), 3)

def get_sentiment_breakdown(text, chunks):
    results = []
    # Limit chunks to process for faster response on smaller models
    for chunk in chunks[:3]: # Processing fewer chunks
        try:
            r = sentiment_pipe(chunk)[0]
            results.append(r)
        except Exception:
            pass
    if not results:
        return "neutral", 0.5, 0, 0, 0
    # Note: 'bertweet-base-sentiment-pipeline' returns 'NEG', 'NEU', 'POS'
    neg = sum(1 for r in results if r["label"] == "NEG")
    pos = sum(1 for r in results if r["label"] == "POS")
    neu = sum(1 for r in results if r["label"] == "NEU")
    avg_conf = sum(r["score"] for r in results) / len(results)
    dominant = "negative" if neg >= pos and neg >= neu else (
               "positive" if pos >= neu else "neutral")
    return dominant, round(avg_conf, 3), neg, pos, neu

# =============================================================================
# HALLUCINATION SCORING
# =============================================================================

def score_hallucination(text, chunks):
    tl     = text.lower()
    signal = 0.0
    flags  = []
    det    = {}

    # ── RULE 1: Hard falsehood keyword match (highest priority) ──────────────
    matched_falsehoods = find_phrases(tl, HARD_FALSEHOODS)
    det["matched_falsehoods"] = matched_falsehoods
    if matched_falsehoods:
        signal += 75  # guaranteed high score for known falsehoods
        flags.append(f"Known scientific falsehood detected")

    # ── RULE 2: Anti-science / denial language ────────────────────────────────
    anti_sci = find_phrases(tl, ANTI_SCIENCE_PHRASES)
    det["anti_science"] = anti_sci
    if anti_sci:
        signal += 30
        flags.append("Denies scientific community")

    # ── RULE 3: Zero-shot model (used as supporting signal, not primary) ──────
    try:
        # Use first chunk or text if short enough
        zs_input = chunks[0][:300] if len(chunks[0]) > 300 else chunks[0]
        zs = zero_shot_pipe(
            zs_input,
            candidate_labels=[
                "factual and accurate",
                "false or misleading",
                "speculative",
            ],
            hypothesis_template="This text is {}.",
        )
        ls = dict(zip(zs["labels"], zs["scores"]))
        false_p      = ls.get("false or misleading", 0)
        speculative_p= ls.get("speculative", 0)
        factual_p    = ls.get("factual and accurate", 0)
        det.update({"false_p": round(false_p,3), "speculative_p": round(speculative_p,3), "factual_p": round(factual_p,3)})

        # Only use model score if no hard rules already fired
        if not matched_falsehoods and not anti_sci:
            signal += false_p * 45 + speculative_p * 25

        if false_p > 0.4 and not matched_falsehoods:
            flags.append("Model: high false/misleading probability")
        if speculative_p > 0.45 and not matched_falsehoods:
            flags.append("Speculative language")
    except Exception as e:
        print(f"Zero-shot pipe error: {e}")
        det["false_p"] = det["speculative_p"] = det["factual_p"] = 0

    # ── RULE 4: Unsourced factual claim phrases ───────────────────────────────
    unsourced = find_phrases(tl, UNSOURCED_CLAIM_PHRASES)
    cit_count = count_citations(text)
    det["unsourced"] = unsourced
    det["cit_count"] = cit_count
    if unsourced and cit_count == 0:
        signal += 8 + min(len(unsourced) * 3, 15)
        flags.append(f"Unsourced factual claims ({len(unsourced)})")
    elif cit_count >= 3:
        signal -= 12  # reward well-cited text

    # ── RULE 5: Absolute / overconfident language ─────────────────────────────
    abs_found = find_phrases(tl, ABSOLUTE_PHRASES)
    det["abs_found"] = abs_found
    if len(abs_found) >= 3:
        signal += 15
        flags.append(f"Heavy absolute language ({len(abs_found)} phrases)")
    elif len(abs_found) >= 1:
        signal += 7
        flags.append("Absolute language present")

    # ── RULE 6: Hedge phrases reduce score ────────────────────────────────────
    hedge_found = find_phrases(tl, HEDGE_PHRASES)
    det["hedge_found"] = hedge_found
    if len(hedge_found) >= 3 and not matched_falsehoods:
        signal -= 10
    elif len(hedge_found) >= 1 and not matched_falsehoods:
        signal -= 4

    score = min(100, max(0, int(signal)))
    return score, flags, det

# =============================================================================
# BIAS SCORING
# =============================================================================

def score_bias(text, chunks):
    tl     = text.lower()
    signal = 0.0
    flags  = []
    det    = {}

    # ── RULE 1: Sentiment analysis ────────────────────────────────────────────
    dominant, avg_conf, neg, pos, neu = get_sentiment_breakdown(text, chunks)
    total = max(neg + pos + neu, 1)
    neg_ratio = neg / total
    pos_ratio = pos / total
    det.update({"dominant": dominant, "avg_conf": avg_conf,
                "neg": neg, "pos": pos, "neu": neu})

    if neg_ratio >= 0.8:
        signal += 35
        flags.append("Overwhelmingly negative tone")
    elif neg_ratio >= 0.6:
        signal += 22
        flags.append("Predominantly negative tone")
    elif neg_ratio >= 0.4:
        signal += 10

    if pos_ratio >= 0.8:
        signal += 25
        flags.append("Excessively promotional tone")
    elif pos_ratio >= 0.6:
        signal += 14

    if avg_conf > 0.85 and dominant != "neutral":
        signal += 10
        flags.append("High-confidence sentiment skew")

    # ── RULE 2: Ad hominem / personal attacks ─────────────────────────────────
    ad_hom = find_phrases(tl, AD_HOMINEM)
    det["ad_hominem"] = ad_hom
    if ad_hom:
        signal += 25
        flags.append(f"Personal attacks / ad hominem")

    # ── RULE 3: Conspiracy language ───────────────────────────────────────────
    conspiracy = find_phrases(tl, CONSPIRACY_PHRASES)
    det["conspiracy"] = conspiracy
    if len(conspiracy) >= 2:
        signal += 25
        flags.append("Conspiracy framing detected")
    elif len(conspiracy) == 1:
        signal += 12
        flags.append(f"Conspiracy language present")

    # ── RULE 4: Emotional / inflammatory words ────────────────────────────────
    emo_found = find_phrases(tl, EMOTIONAL_WORDS)
    det["emo_found"] = emo_found
    if len(emo_found) >= 4:
        signal += 20
        flags.append(f"Heavy inflammatory language ({len(emo_found)} words)")
    elif len(emo_found) >= 2:
        signal += 12
        flags.append("Emotionally charged language")
    elif len(emo_found) == 1:
        signal += 6

    # ── RULE 5: Persuasion / manipulation tactics ─────────────────────────────
    persuasion = find_phrases(tl, PERSUASION_TACTICS)
    det["persuasion"] = persuasion
    if persuasion:
        signal += 15
        flags.append("Manipulation tactics present")

    # ── RULE 6: Zero-shot bias classification ─────────────────────────────────
    try:
        # Use first chunk or text if short enough
        zs_input = chunks[0][:300] if len(chunks[0]) > 300 else chunks[0]
        zs = zero_shot_pipe(
            zs_input,
            candidate_labels=[
                "balanced and objective",
                "biased and one-sided",
                "emotionally manipulative",
            ],
            hypothesis_template="This text is {}.",
        )
        ls = dict(zip(zs["labels"], zs["scores"]))
        objective_p    = ls.get("balanced and objective", 0)
        biased_p       = ls.get("biased and one-sided", 0)
        manipulative_p = ls.get("emotionally manipulative", 0)
        det.update({"objective_p": round(objective_p,3), "biased_p": round(biased_p,3),
                    "manipulative_p": round(manipulative_p,3)})

        signal += biased_p * 20 + manipulative_p * 25

        if biased_p > 0.45 and not conspiracy and not ad_hom:
            flags.append("Model: one-sided framing")
        if manipulative_p > 0.4:
            flags.append("Model: emotionally manipulative")
        if objective_p > 0.55:
            signal = max(0, signal - 12)
    except Exception as e:
        print(f"Zero-shot pipe error: {e}")
        det["objective_p"] = det["biased_p"] = det["manipulative_p"] = 0

    # ── RULE 7: Opinion markers ───────────────────────────────────────────────
    opinion = find_phrases(tl, ["i think", "i believe", "in my opinion",
                                 "personally", "i feel", "from my perspective"])
    det["opinion"] = opinion
    if len(opinion) >= 2:
        signal += 8
        flags.append("Personal opinion markers")

    score = min(100, max(0, int(signal)))
    return score, flags, det

# =============================================================================
# TRUST SCORING
# =============================================================================

def score_trust(hallucination, bias, text):
    tl    = text.lower()
    flags = []
    base  = 100 - (hallucination * 0.6 + bias * 0.4)

    cit = count_citations(text)
    if cit >= 4:   base += 12; flags.append("Well-cited")
    elif cit >= 2: base += 7;  flags.append("Has citations")
    elif cit == 1: base += 3

    hedge = count_phrases(tl, HEDGE_PHRASES)
    if hedge >= 3 and not "Known scientific falsehood detected" in flags: base += 6; flags.append("Well-hedged")
    elif hedge >= 1 and not "Known scientific falsehood detected" in flags: base += 3

    formal = formality_score(text)
    if formal >= 0.65: base += 5; flags.append("Academic tone")
    elif formal >= 0.4: base += 2

    wc = word_count(text)
    if wc < 15:   base -= 10
    elif wc < 30: base -= 5
    elif wc > 80: base += 3

    score = min(100, max(0, int(base)))
    if score >= 75:   flags.append("High reliability")
    elif score >= 50: flags.append("Moderate reliability")
    elif score >= 30: flags.append("Low reliability")
    else:             flags.append("Very low reliability")

    return score, flags

# =============================================================================
# EXPLANATION BUILDER — fully text-aware
# =============================================================================

def build_explanation(text, hallucination, bias, trust, hal_det, bias_det):
    tl    = text.lower()
    wc    = word_count(text)
    sents = tokenize_sentences(text)
    sc    = len(sents)
    parts = []

    # ── 1. Opening verdict ────────────────────────────────────────────────────
    if hallucination >= 80 and bias >= 70:
        parts.append(
            f"This text scores {hallucination}/100 on hallucination and {bias}/100 on bias — "
            "it contains demonstrably false claims combined with aggressive, manipulative language, "
            "making it highly unreliable and potentially harmful misinformation."
        )
    elif hallucination >= 80:
        parts.append(
            f"This text has a very high hallucination score ({hallucination}/100), "
            "meaning its core claims are factually incorrect or contradict established science."
        )
    elif hallucination >= 50 and bias >= 50:
        parts.append(
            f"With a hallucination score of {hallucination}/100 and bias score of {bias}/100, "
            "this text contains significant misinformation combined with strong emotional bias."
        )
    elif hallucination >= 50:
        parts.append(
            f"This text scores {hallucination}/100 on hallucination — "
            "its claims are largely unverified, speculative, or factually inaccurate."
        )
    elif bias >= 70:
        parts.append(
            f"While factual risk is lower ({hallucination}/100), this text is heavily biased "
            f"({bias}/100) — it uses emotionally charged or one-sided language to influence the reader."
        )
    elif hallucination <= 20 and bias <= 25:
        parts.append(
            f"This text appears largely accurate and balanced "
            f"(hallucination: {hallucination}/100, bias: {bias}/100)."
        )
    else:
        parts.append(
            f"This text scores {hallucination}/100 on hallucination and {bias}/100 on bias — "
            "some claims and language choices warrant scrutiny."
        )

    # ── 2. Specific falsehood ─────────────────────────────────────────────────
    kf = hal_det.get("matched_falsehoods", [])
    if kf:
        # Quote the exact phrase from the original text context
        parts.append(
            f"Specifically, the claim that the '{kf[0]}' is a well-documented scientific falsehood. "
            "The Moon is a rocky body composed of silicate rock and metal oxides — "
            "this has been confirmed by multiple Apollo missions, lunar samples, and independent research."
        )

    # ── 3. Anti-science language ──────────────────────────────────────────────
    anti_sci = hal_det.get("anti_science", [])
    if anti_sci:
        example = anti_sci[0]
        parts.append(
            f"The text uses anti-science rhetoric (e.g., '{example}'), "
            "dismissing the entire scientific community without evidence — "
            "this is a classic pattern in misinformation designed to pre-emptively discredit fact-checkers."
        )

    # ── 4. Citations / unsourced claims ──────────────────────────────────────
    unsourced = hal_det.get("unsourced", [])
    cit_count = hal_det.get("cit_count", 0)
    abs_found = hal_det.get("abs_found", [])

    if abs_found and cit_count == 0:
        examples = ", ".join([f"'{p}'" for p in abs_found[:3]])
        parts.append(
            f"The text uses absolute language ({examples}) without providing "
            "any citations, studies, or verifiable sources — "
            "legitimate factual claims are supported by evidence, not just confident assertions."
        )
    elif unsourced and cit_count == 0:
        examples = " and ".join([f"'{p}'" for p in unsourced[:2]])
        parts.append(
            f"Phrases like {examples} assert facts without any cited sources — "
            "no references, studies, or links are provided to support these claims."
        )
    elif cit_count >= 2:
        parts.append(
            f"The text includes {cit_count} citation(s), "
            "which supports the credibility of its factual claims."
        )
    elif cit_count == 0 and wc >= 25:
        parts.append(
            "No citations or references were found — "
            "factual claims in this text cannot be independently verified."
        )

    # ── 5. Ad hominem / conspiracy / emotional language ───────────────────────
    ad_hom     = bias_det.get("ad_hominem", [])
    conspiracy = bias_det.get("conspiracy", [])
    emo_found  = bias_det.get("emo_found", [])
    persuasion = bias_det.get("persuasion", [])

    if ad_hom:
        example = ad_hom[0]
        parts.append(
            f"The text resorts to personal attacks (e.g., '{example}'), "
            "which is a logical fallacy (ad hominem) — "
            "attacking people who hold a view rather than addressing the evidence is a hallmark of unreliable content."
        )

    if conspiracy:
        examples = ", ".join([f"'{c}'" for c in conspiracy[:2]])
        parts.append(
            f"Conspiracy-style language is used ({examples}), "
            "framing scientists or authorities as malicious agents — "
            "this is a rhetorical device to discourage critical thinking and fact-checking."
        )
    elif emo_found and not ad_hom:
        examples = ", ".join([f"'{w}'" for w in emo_found[:3]])
        parts.append(
            f"Inflammatory words ({examples}) are used to provoke an emotional response "
            "rather than present a reasoned argument, which is a strong bias indicator."
        )

    if persuasion:
        parts.append(
            f"The text uses persuasion tactics ('{persuasion[0]}') "
            "designed to pressure readers into accepting claims without critical evaluation."
        )

    # ── 6. Hedge / balance check ──────────────────────────────────────────────
    hedge_found = hal_det.get("hedge_found", [])
    if hedge_found and hallucination < 40:
        examples = ", ".join([f"'{p}'" for p in hedge_found[:2]])
        parts.append(
            f"The author uses appropriate hedging language ({examples}), "
            "which reflects epistemic caution consistent with accurate, careful writing."
        )

    # ── 7. Trust verdict ──────────────────────────────────────────────────────
    if trust <= 15:
        parts.append(
            f"Trust score: {trust}/100 — this text should not be used or shared as factual information. "
            "It contains false claims and manipulative language."
        )
    elif trust <= 35:
        parts.append(
            f"Trust score: {trust}/100 — this text is unreliable and should not be cited "
            "without substantial independent verification from credible sources."
        )
    elif trust <= 55:
        parts.append(
            f"Trust score: {trust}/100 — treat this as a starting point only; "
            "cross-reference key claims with peer-reviewed or authoritative sources."
        )
    else:
        parts.append(
            f"Trust score: {trust}/100 — this text is relatively reliable, "
            "though independent verification of specific claims is always recommended."
        )

    return " ".join(parts[:6])

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "No text provided."}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "No text provided."}), 400
        if len(text) > 12000:
            return jsonify({"error": "Text too long (max 12,000 chars)."}), 400

        chunks    = chunk_text(text)
        # NLTK tokenization requires data to be loaded.
        # Ensure it's available or handle errors gracefully.
        try:
            sentences = tokenize_sentences(text)
        except LookupError:
            # If NLTK data isn't found even after download attempt,
            # this indicates a Vercel-specific issue or a new environment.
            # In a deployed setting, this should ideally not happen if downloads work.
            print("NLTK data not found. Attempting re-download.")
            nltk.download("punkt", quiet=True, download_dir=NLTK_DATA_DIR)
            nltk.download("punkt_tab", quiet=True, download_dir=NLTK_DATA_DIR)
            nltk.download("stopwords", quiet=True, download_dir=NLTK_DATA_DIR)
            sentences = tokenize_sentences(text) # Try again

        hal_score,  hal_flags,  hal_det  = score_hallucination(text, chunks)
        bias_score, bias_flags, bias_det = score_bias(text, chunks)
        trust_score, trust_flags         = score_trust(hal_score, bias_score, text)

        all_flags = list(dict.fromkeys(hal_flags + bias_flags + trust_flags))[:7]

        explanation = build_explanation(
            text, hal_score, bias_score, trust_score,
            hal_det, bias_det
        )

        return jsonify({
            "hallucination_score": hal_score,
            "bias_score":          bias_score,
            "trust_score":         trust_score,
            "explanation":         explanation,
            "flags":               all_flags,
        })

    except Exception as err:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(err)}), 500
