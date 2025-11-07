// static/app.js
const micBtn = document.getElementById("mic");
const searchBtn = document.getElementById("search");
const queryInput = document.getElementById("query");
const langSel = document.getElementById("lang");
const resultsDiv = document.getElementById("results");

let recognition = null;
if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.onresult = (e) => {
    const text = e.results[0][0].transcript;
    queryInput.value = text;
  };
  recognition.onerror = (e) => {
    console.warn("Speech error", e);
  };
} else {
  micBtn.disabled = true;
  micBtn.title = "Speech recognition not supported by this browser";
}

micBtn.onclick = () => {
  if (!recognition) return;
  recognition.lang = langSel.value || "en-US";
  recognition.start();
};

async function search() {
  const q = queryInput.value.trim();
  if (!q) return alert("Type or speak a query first.");
  resultsDiv.innerHTML = "<p>Searching…</p>";
  try {
    const resp = await fetch("/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, top_k: 8 })
    });
    const data = await resp.json();
    renderResults(data);
  } catch (err) {
    resultsDiv.innerHTML = "<p>Error: " + err.message + "</p>";
  }
}

function renderResults(data) {
  const res = data.results || [];
  if (!res.length) {
    resultsDiv.innerHTML = "<p>No results.</p>";
    return;
  }
  let html = "";
  for (const r of res) {
    html += `
      <div class="card">
        <div class="row">
          <div class="left"><strong>Section ${r.section}</strong> — ${r.title}</div>
          <div class="right">Confidence: ${r.confidence.toFixed(1)}%</div>
        </div>
        <div class="desc">${r.description}</div>
      </div>
    `;
  }
  resultsDiv.innerHTML = html;
}

searchBtn.onclick = search;
queryInput.onkeydown = (e) => {
  if (e.key === "Enter") search();
};
