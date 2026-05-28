"use strict";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const uploadSection  = document.getElementById("uploadSection");
const uploadZone     = document.getElementById("uploadZone");
const fileInput      = document.getElementById("fileInput");
const browseBtn      = document.getElementById("browseBtn");
const analysisSection = document.getElementById("analysisSection");
const previewWrap    = document.getElementById("previewWrap");
const fileMeta       = document.getElementById("fileMeta");
const fallbackNotice = document.getElementById("fallbackNotice");
const stateAnalyzing = document.getElementById("stateAnalyzing");
const stateResult    = document.getElementById("stateResult");
const resultCard     = document.getElementById("resultCard");
const verdict        = document.getElementById("verdict");
const verdictIcon    = document.getElementById("verdictIcon");
const verdictLabel   = document.getElementById("verdictLabel");
const verdictSub     = document.getElementById("verdictSub");
const probFill       = document.getElementById("probFill");
const realPct        = document.getElementById("realPct");
const fakePct        = document.getElementById("fakePct");
const statsGrid      = document.getElementById("statsGrid");
const framesSection  = document.getElementById("framesSection");
const framesTrack    = document.getElementById("framesTrack");
const resetBtn       = document.getElementById("resetBtn");

// ── Upload interactions ───────────────────────────────────────────────────────
browseBtn.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("click", (e) => {
  if (e.target !== browseBtn) fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

resetBtn.addEventListener("click", reset);

// ── Core flow ─────────────────────────────────────────────────────────────────
function handleFile(file) {
  const isImage = file.type.startsWith("image/");
  const isVideo = file.type.startsWith("video/");
  if (!isImage && !isVideo) {
    showError("Unsupported file type. Please upload JPG, PNG, or MP4.");
    return;
  }

  showAnalysisPanel();
  renderPreview(file, isVideo);
  fileMeta.textContent = `${file.name} · ${formatSize(file.size)}`;
  showAnalyzing();
  sendToAPI(file);
}

function sendToAPI(file) {
  const form = new FormData();
  form.append("file", file);

  fetch("/api/detect", { method: "POST", body: form })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) { showError(data.error); return; }
      showResult(data);
    })
    .catch(() => showError("Network error — is the server running?"));
}

// ── Preview rendering ─────────────────────────────────────────────────────────
function renderPreview(file, isVideo) {
  previewWrap.innerHTML = "";

  const url = URL.createObjectURL(file);

  if (isVideo) {
    const vid = document.createElement("video");
    vid.src = url;
    vid.controls = true;
    vid.muted = true;
    vid.style.maxWidth = "100%";
    previewWrap.appendChild(vid);
  } else {
    const img = document.createElement("img");
    img.src = url;
    img.alt = "Uploaded image";
    img.id = "previewImg";
    previewWrap.appendChild(img);

    const canvas = document.createElement("canvas");
    canvas.id = "faceCanvas";
    previewWrap.appendChild(canvas);
  }
}

function drawFaceBoxes(faces, imgNaturalW, imgNaturalH) {
  const img = document.getElementById("previewImg");
  const canvas = document.getElementById("faceCanvas");
  if (!img || !canvas) return;

  const draw = () => {
    const dispW = img.offsetWidth;
    const dispH = img.offsetHeight;
    const scaleX = dispW / imgNaturalW;
    const scaleY = dispH / imgNaturalH;

    const rect = img.getBoundingClientRect();
    const wrapRect = previewWrap.getBoundingClientRect();
    canvas.style.left = (rect.left - wrapRect.left) + "px";
    canvas.style.top  = (rect.top  - wrapRect.top)  + "px";
    canvas.style.width  = dispW + "px";
    canvas.style.height = dispH + "px";
    canvas.width  = dispW;
    canvas.height = dispH;

    const ctx = canvas.getContext("2d");
    ctx.lineWidth = 2.5;
    ctx.font = "bold 12px system-ui";

    faces.forEach((face) => {
      const { x, y, w, h } = face.bbox;
      const bx = x * scaleX;
      const by = y * scaleY;
      const bw = w * scaleX;
      const bh = h * scaleY;

      const color = face.is_fake ? "#dc2626" : "#16a34a";
      ctx.strokeStyle = color;
      ctx.fillStyle   = color;

      ctx.strokeRect(bx, by, bw, bh);

      const label = face.is_fake
        ? `Fake ${pct(face.probabilities.fake)}`
        : `Real ${pct(face.probabilities.real)}`;
      const textW = ctx.measureText(label).width;
      ctx.globalAlpha = 0.8;
      ctx.fillRect(bx, by - 20, textW + 10, 20);
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#fff";
      ctx.fillText(label, bx + 5, by - 5);
      ctx.fillStyle = color;
    });
  };

  if (img.complete) draw();
  else img.addEventListener("load", draw, { once: true });
}

// ── Result rendering ──────────────────────────────────────────────────────────
function showResult(data) {
  const isFake = data.is_fake;
  const fakeProbVal = data.media_type === "video"
    ? data.avg_fake_probability
    : (data.faces?.[0]?.probabilities?.fake ?? (isFake ? data.confidence : 1 - data.confidence));
  const realProbVal = 1 - fakeProbVal;

  verdict.className = "verdict " + (isFake ? "fake" : "real");
  verdictIcon.textContent = isFake ? "🔴" : "✅";
  verdictLabel.textContent = isFake ? "FAKE" : "REAL";
  verdictSub.textContent = `Confidence: ${pct(data.confidence)}`;

  probFill.style.width = pct(fakeProbVal);
  realPct.textContent = pct(realProbVal);
  fakePct.textContent = pct(fakeProbVal);

  const stats = [
    { label: "Processing time", value: `${(data.elapsed_seconds * 1000).toFixed(0)} ms` },
  ];
  if (data.media_type === "image") {
    stats.push({ label: "Faces detected", value: data.num_faces ?? 0 });
    if (data.original_width) {
      stats.push({ label: "Resolution", value: `${data.original_width}×${data.original_height}` });
    }
  } else {
    stats.push({ label: "Frames analysed", value: data.num_frames_analyzed });
    stats.push({ label: "Fake frames", value: `${data.fake_frame_count} / ${data.num_frames_analyzed}` });
  }
  statsGrid.innerHTML = stats
    .map((s) => `<div class="stat-item"><div class="stat-label">${s.label}</div><div class="stat-value">${s.value}</div></div>`)
    .join("");

  if (data.media_type === "image" && data.faces?.length) {
    fallbackNotice.hidden = !data.used_full_frame_fallback;
    drawFaceBoxes(data.faces, data.original_width, data.original_height);
    renderHeatmaps(data.faces);
  }

  if (data.media_type === "video" && data.frame_results?.length) {
    framesSection.hidden = false;
    renderFrameTimeline(data.frame_results);
  }

  stateAnalyzing.hidden = true;
  stateResult.hidden    = false;
  resultCard.style.alignItems = "flex-start";
}

function renderHeatmaps(faces) {
  const existing = document.getElementById("heatmapSection");
  if (existing) existing.remove();

  const fakeFaces = faces.filter((f) => f.is_fake && f.gradcam_heatmap);
  if (!fakeFaces.length) return;

  const section = document.createElement("div");
  section.id = "heatmapSection";
  section.style.cssText = "margin-top:20px;";

  const title = document.createElement("p");
  title.textContent = "Model attention map";
  title.style.cssText = "font-weight:600;font-size:13px;margin-bottom:8px;color:#374151;";
  section.appendChild(title);

  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-wrap:wrap;gap:10px;";

  fakeFaces.forEach((face, idx) => {
    const item = document.createElement("div");
    const img = document.createElement("img");
    img.src = `data:image/jpeg;base64,${face.gradcam_heatmap}`;
    img.style.cssText = "width:160px;height:160px;object-fit:cover;border:2px solid #dc2626;border-radius:6px;display:block;";
    img.alt = `Face ${idx + 1} attention map`;
    const lbl = document.createElement("p");
    lbl.textContent = `Face ${idx + 1} — regions that influenced FAKE verdict`;
    lbl.style.cssText = "font-size:11px;color:#6b7280;margin-top:4px;max-width:160px;";
    item.appendChild(img);
    item.appendChild(lbl);
    wrap.appendChild(item);
  });

  section.appendChild(wrap);
  stateResult.appendChild(section);
}

function renderFrameTimeline(frames) {
  const maxProb = Math.max(...frames.map((f) => f.best_fake_prob));
  framesTrack.innerHTML = frames.map((f) => {
    const heightPct = maxProb > 0 ? Math.round((f.best_fake_prob / maxProb) * 100) : 50;
    const cls = f.is_fake ? "fake" : "real";
    return `<div class="frame-bar ${cls}" style="height:${heightPct}%"
              title="Frame ${f.frame_index} · Fake ${pct(f.best_fake_prob)}"></div>`;
  }).join("");
}

// ── UI state helpers ──────────────────────────────────────────────────────────
function showAnalysisPanel() {
  uploadSection.hidden  = true;
  analysisSection.hidden = false;
}

function showAnalyzing() {
  stateAnalyzing.hidden = false;
  stateResult.hidden    = true;
  fallbackNotice.hidden = true;
  framesSection.hidden  = true;
  resultCard.style.alignItems = "center";
}

function showError(msg) {
  stateAnalyzing.hidden = true;
  stateResult.hidden    = false;
  resultCard.style.alignItems = "center";
  verdict.className = "verdict fake";
  verdictIcon.textContent = "⚠️";
  verdictLabel.textContent = "Error";
  verdictSub.textContent = msg;
  probFill.style.width = "0%";
  statsGrid.innerHTML = "";
  framesSection.hidden = true;
}

function reset() {
  fileInput.value = "";
  previewWrap.innerHTML = "";
  statsGrid.innerHTML   = "";
  framesTrack.innerHTML = "";
  uploadSection.hidden  = false;
  analysisSection.hidden = true;
  stateAnalyzing.hidden  = false;
  stateResult.hidden     = true;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function pct(v) { return (v * 100).toFixed(1) + "%"; }

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 ** 2) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 ** 2).toFixed(1) + " MB";
}
