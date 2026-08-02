/* SpectroGuard frontend logic.
 * Flow: upload -> analyze -> poll status -> render result.
 */

const ALLOWED = ["mp4", "mov", "avi", "mkv", "webm"];

let videoId = null;
let pollTimer = null;

const $ = (id) => document.getElementById(id);

const dropzone = $("dropzone");
const fileInput = $("file-input");
const analyzeBtn = $("analyze-btn");
const resetBtn = $("reset-btn");
const errorBox = $("error-box");
const uploadCard = $("upload-card");
const statusCard = $("status-card");
const resultCard = $("result-card");

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }
function setError(msg) {
  errorBox.textContent = msg;
  show(errorBox);
}

/* ---------- File selection ---------- */
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  hide(errorBox);
  const ext = file.name.split(".").pop().toLowerCase();
  if (!ALLOWED.includes(ext)) {
    setError(`Unsupported file type ".${ext}". Please upload MP4, MOV, AVI, MKV or WEBM.`);
    analyzeBtn.disabled = true;
    return;
  }
  if (file.size > 200 * 1024 * 1024) {
    setError("File is larger than the 200 MB limit.");
    analyzeBtn.disabled = true;
    return;
  }
  $("dz-file").textContent = file.name;
  resetBtn.disabled = false;
  uploadFile(file);
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    videoId = data.video_id;
    $("video-preview").src = data.video_url;
    show($("preview-block"));
    analyzeBtn.disabled = false;
  } catch (err) {
    setError(err.message);
  }
}

/* ---------- Analyze ---------- */
analyzeBtn.addEventListener("click", async () => {
  if (!videoId) return;
  hide(errorBox);
  hide(uploadCard);
  show(statusCard);
  hide(resultCard);
  $("status-message").textContent = "Starting analysis...";
  $("progress-fill").style.width = "5%";

  try {
    const res = await fetch(`/api/analyze/${videoId}`, { method: "POST" });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || "Could not start analysis");
    }
    pollTimer = setInterval(pollStatus, 1000);
  } catch (err) {
    show(uploadCard);
    hide(statusCard);
    setError(err.message);
  }
});

async function pollStatus() {
  try {
    const res = await fetch(`/api/status/${videoId}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Status check failed");

    $("status-message").textContent = data.message || data.status;
    $("progress-fill").style.width = `${data.progress || 0}%`;

    if (data.status === "done") {
      clearInterval(pollTimer);
      fetchResult();
    } else if (data.status === "error") {
      clearInterval(pollTimer);
      throw new Error(data.error || "Analysis failed");
    }
  } catch (err) {
    clearInterval(pollTimer);
    show(uploadCard);
    hide(statusCard);
    setError(err.message);
  }
}

/* ---------- Result ---------- */
async function fetchResult() {
  const res = await fetch(`/api/result/${videoId}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Could not fetch result");
  renderResult(data.result);
}

function renderResult(r) {
  const prediction = r.prediction; // "REAL" | "FAKE"
  const confidence = Math.round((r.confidence || 0) * 100);
  const isFake = prediction === "FAKE";

  const badge = $("prediction-badge");
  badge.textContent = prediction;
  badge.className = `badge ${prediction}`;

  $("confidence-value").textContent = `${confidence}%`;
  const bar = $("confidence-bar");
  bar.className = `bar-fill ${prediction}`;
  // animate shortly after paint
  requestAnimationFrame(() => setTimeout(() => { bar.style.width = `${confidence}%`; }, 50));

  $("frames").textContent = r.frames_analyzed ?? "-";
  $("faces").textContent = r.faces_detected ?? "-";
  $("time").textContent = r.processing_time != null ? `${r.processing_time}s` : "-";
  $("votes").textContent =
    r.votes ? `${r.votes.REAL} / ${r.votes.FAKE}` : "-";

  hide(statusCard);
  show(resultCard);
  show(uploadCard);

  document.querySelector(".health").classList.add("ok");
}

/* ---------- Reset ---------- */
resetBtn.addEventListener("click", () => location.reload());

/* ---------- Backend health pill ---------- */
(async () => {
  try {
    const res = await fetch("/health");
    if (res.ok) document.querySelector(".health").classList.add("ok");
    else document.querySelector(".health").classList.add("down");
  } catch {
    document.querySelector(".health").classList.add("down");
  }
})();
