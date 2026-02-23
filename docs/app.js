/* ─────────────────────────────────────────────────────────
   Vietnamese Food Classification – App Logic
   ───────────────────────────────────────────────────────── */

const API_BASE = "https://vietnamese-food-classification.onrender.com";

// ── Food metadata (emoji + display name) ─────────────────
const FOOD_META = {
  "banh chung": { emoji: "🍱", display: "Bánh Chưng" },
  "banh mi": { emoji: "🥖", display: "Bánh Mì" },
  "banh xeo": { emoji: "🥞", display: "Bánh Xèo" },
  "bun bo hue": { emoji: "🍜", display: "Bún Bò Huế" },
  "bun dau mam tom": { emoji: "🍲", display: "Bún Đậu Mắm Tôm" },
  "cha gio": { emoji: "🥟", display: "Chả Giò" },
  "chao long": { emoji: "🥣", display: "Cháo Lòng" },
};

// Fallback: try to match normalized class_name against known keys
function normalizeKey(name) {
  return name
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
    .replace(/\s+/g, " ")
    .trim();
}

function getMeta(className) {
  const key = normalizeKey(className);
  // Direct match
  if (FOOD_META[key]) return FOOD_META[key];
  // Partial match
  for (const [k, v] of Object.entries(FOOD_META)) {
    if (key.includes(k) || k.includes(key)) return v;
  }
  return { emoji: "🍽️", display: className };
}

// ── Element refs ─────────────────────────────────────────
const dropZone = document.getElementById("dropZone");
const dropInner = document.getElementById("dropInner");
const previewOverlay = document.getElementById("previewOverlay");
const previewImg = document.getElementById("previewImg");
const fileInput = document.getElementById("fileInput");
const btnRemove = document.getElementById("btnRemove");
const btnPredict = document.getElementById("btnPredict");
const btnLabel = document.getElementById("btnLabel");
const btnSpinner = document.getElementById("btnSpinner");
const alertError = document.getElementById("alertError");
const resultIdle = document.getElementById("resultIdle");
const resultContent = document.getElementById("resultContent");
const resultClass = document.getElementById("resultClass");
const resultConf = document.getElementById("resultConfidence");
const probList = document.getElementById("probList");
const classesGrid = document.getElementById("classesGrid");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

// ── State ─────────────────────────────────────────────────
let selectedFile = null;
let serverClasses = [];

// ── Health check ─────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      const data = await res.json();
      setStatus(data.model_loaded ? "online" : "loading");
      if (data.classes && data.classes.length > 0) {
        serverClasses = data.classes;
        renderClasses(serverClasses);
      }
    } else {
      setStatus("offline");
    }
  } catch {
    setStatus("offline");
  }
}

function setStatus(state) {
  statusDot.className = "badge-dot " + state;
  const labels = {
    online: "Model sẵn sàng",
    loading: "Model đang tải...",
    offline: "Server offline",
  };
  statusText.textContent = labels[state] || "Không rõ";
}

// ── Render class chips ────────────────────────────────────
function renderClasses(classes) {
  classesGrid.innerHTML = "";
  classes.forEach(cls => {
    const meta = getMeta(cls);
    const chip = document.createElement("div");
    chip.className = "class-chip";
    chip.dataset.class = cls;
    chip.innerHTML = `<div class="chip-emoji">${meta.emoji}</div>${meta.display}`;
    classesGrid.appendChild(chip);
  });
}

// Highlight active class chip
function highlightClass(className) {
  document.querySelectorAll(".class-chip").forEach(chip => {
    chip.classList.toggle("active", chip.dataset.class === className);
  });
}

// ── Drag & Drop ───────────────────────────────────────────
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith("image/")) setFile(file);
});

// Click zone → trigger file input (prevent double-trigger from label)
dropZone.addEventListener("click", e => {
  if (e.target === dropZone) fileInput.click();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) setFile(file);
});

btnRemove.addEventListener("click", e => {
  e.stopPropagation();
  clearFile();
});

function setFile(file) {
  if (file.size > 10 * 1024 * 1024) {
    showError("Kích thước file vượt quá 10 MB. Vui lòng chọn ảnh nhỏ hơn.");
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewOverlay.hidden = false;
  dropInner.hidden = true;
  btnPredict.disabled = false;
  clearError();
  clearResult();
}

function clearFile() {
  selectedFile = null;
  previewImg.src = "";
  previewOverlay.hidden = true;
  dropInner.hidden = false;
  btnPredict.disabled = true;
  fileInput.value = "";
  clearError();
  clearResult();
}

// ── Predict ───────────────────────────────────────────────
btnPredict.addEventListener("click", async () => {
  if (!selectedFile) return;
  setLoading(true);
  clearError();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Lỗi HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError(`Không thể kết nối đến server hoặc có lỗi xảy ra: ${err.message}`);
    clearResult();
  } finally {
    setLoading(false);
  }
});

// ── Render result ─────────────────────────────────────────
function renderResult(data) {
  const meta = getMeta(data.predicted_class);

  resultClass.textContent = meta.display;
  resultConf.innerHTML = `Độ tin cậy: <span>${data.confidence.toFixed(1)}%</span>`;

  // Probability bars
  probList.innerHTML = "";
  data.top_classes.forEach((item, idx) => {
    const isTop = idx === 0;
    const m = getMeta(item.class_name);
    const div = document.createElement("div");
    div.className = "prob-item";
    div.innerHTML = `
      <div class="prob-row">
        <span class="prob-name">${m.emoji} ${m.display}</span>
        <span class="prob-value">${item.probability.toFixed(1)}%</span>
      </div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill${isTop ? " top" : ""}" data-width="${item.probability}" style="width:0"></div>
      </div>
    `;
    probList.appendChild(div);
  });

  // Animate bars after render
  requestAnimationFrame(() => {
    document.querySelectorAll(".prob-bar-fill").forEach(bar => {
      const target = parseFloat(bar.dataset.width);
      bar.style.width = `${target}%`;
    });
  });

  resultIdle.hidden = true;
  resultContent.hidden = false;
  resultContent.classList.add("slide-in");

  highlightClass(data.predicted_class);
}

// ── Helpers ───────────────────────────────────────────────
function setLoading(isLoading) {
  btnPredict.disabled = isLoading;
  btnLabel.hidden = isLoading;
  btnSpinner.hidden = !isLoading;
}

function showError(msg) {
  alertError.textContent = msg;
  alertError.hidden = false;
}
function clearError() {
  alertError.textContent = "";
  alertError.hidden = true;
}

function clearResult() {
  resultIdle.hidden = false;
  resultContent.hidden = true;
  resultContent.classList.remove("slide-in");
  document.querySelectorAll(".class-chip").forEach(c => c.classList.remove("active"));
}

// ── Init ──────────────────────────────────────────────────
(async function init() {
  // Render default classes while server loads
  const defaultClasses = Object.keys(FOOD_META);
  renderClasses(defaultClasses.map(k => k));

  await checkHealth();
  // Recheck every 15 seconds
  setInterval(checkHealth, 15000);
})();
