/* ── Config ──────────────────────────────────────────────────────────────── */
const API_URL  = ""; // Vercel will handle the base URL. The API routes start with /api
const MAX_CHARS = 12000;

/* ── Tag colour palette ───────────────────────────────────────────────────── */
const TAG_COLORS = [
  { color: "#00e5ff", border: "rgba(0,229,255,0.35)",  bg: "rgba(0,229,255,0.07)" },
  { color: "#ff6b35", border: "rgba(255,107,53,0.35)", bg: "rgba(255,107,53,0.07)" },
  { color: "#ff4d4d", border: "rgba(255,77,77,0.35)",  bg: "rgba(255,77,77,0.07)" },
  { color: "#39ff7a", border: "rgba(57,255,122,0.35)", bg: "rgba(57,255,122,0.07)" },
  { color: "#ffd93d", border: "rgba(255,217,61,0.35)", bg: "rgba(255,217,61,0.07)" },
];

/* ── Score colour / caption helpers ──────────────────────────────────────── */
function scoreColor(val, invert) {
  if (invert) {
    if (val >= 70) return "#39ff7a";
    if (val >= 45) return "#ffd93d";
    if (val >= 25) return "#ff6b35";
    return "#ff4d4d";
  }
  if (val <= 20) return "#39ff7a";
  if (val <= 45) return "#ffd93d";
  if (val <= 70) return "#ff6b35";
  return "#ff4d4d";
}

function scoreCaption(val, invert) {
  if (invert) {
    if (val >= 70) return "High trust";
    if (val >= 45) return "Moderate";
    if (val >= 25) return "Low trust";
    return "Very low";
  }
  if (val <= 20) return "Very low";
  if (val <= 45) return "Low";
  if (val <= 70) return "Moderate";
  return "High";
}

/* ── DOM refs ─────────────────────────────────────────────────────────────── */
const inputText  = document.getElementById("inputText");
const charCount  = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const btnText    = analyzeBtn.querySelector(".btn-text");
const spinner    = document.getElementById("spinner");
const errorBox   = document.getElementById("errorBox");
const results    = document.getElementById("results");

/* ── Char counter ─────────────────────────────────────────────────────────── */
inputText.addEventListener("input", () => {
  const len = inputText.value.length;
  charCount.textContent = `${len.toLocaleString()} / ${MAX_CHARS.toLocaleString()}`;
  charCount.style.color = len > MAX_CHARS ? "#ff4d4d" : "";
});

/* ── Keyboard shortcut ────────────────────────────────────────────────────── */
inputText.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
});

/* ── UI helpers ───────────────────────────────────────────────────────────── */
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.add("visible");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.remove("visible");
}

function setLoading(on) {
  analyzeBtn.disabled = on;
  spinner.classList.toggle("active", on);
  btnText.textContent = on ? "Analyzing…" : "Analyze Text";
}

function applyScore(valId, barId, capId, value, invert) {
  const color = scoreColor(value, invert);
  document.getElementById(valId).textContent  = value;
  document.getElementById(valId).style.color  = color;
  document.getElementById(capId).textContent  = scoreCaption(value, invert);
  document.getElementById(barId).style.background = color;
  // Trigger CSS width transition on next frame
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.getElementById(barId).style.width = `${value}%`;
    });
  });
}

/* ── Main analyze ─────────────────────────────────────────────────────────── */
async function analyze() {
  const text = inputText.value.trim();
  if (!text)             { showError("Please paste some text to analyze."); return; }
  if (text.length > MAX_CHARS) { showError(`Text exceeds ${MAX_CHARS.toLocaleString()} characters.`); return; }

  clearError();
  results.classList.remove("visible");
  setLoading(true);

  // Reset bars
  ["barHallucination", "barBias", "barTrust"].forEach(id => {
    document.getElementById(id).style.width = "0%";
  });

  try {
    const res = await fetch(`/api/analyze`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || `Server error ${res.status}`);
    }

    applyScore("valHallucination", "barHallucination", "capHallucination", data.hallucination_score, false);
    applyScore("valBias",          "barBias",           "capBias",          data.bias_score,          false);
    applyScore("valTrust",         "barTrust",          "capTrust",         data.trust_score,         true);

    document.getElementById("explanationText").textContent = data.explanation;

    const flagsRow = document.getElementById("flagsRow");
    flagsRow.innerHTML = "";
    (data.flags || []).forEach((flag, i) => {
      const c    = TAG_COLORS[i % TAG_COLORS.length];
      const span = document.createElement("span");
      span.className        = "flag";
      span.textContent      = flag;
      span.style.color      = c.color;
      span.style.border     = `1px solid ${c.border}`;
      span.style.background = c.bg;
      flagsRow.appendChild(span);
    });

    results.classList.add("visible");

  } catch (err) {
    showError("Analysis failed: " + err.message);
  } finally {
    setLoading(false);
  }
}

analyzeBtn.addEventListener("click", analyze);