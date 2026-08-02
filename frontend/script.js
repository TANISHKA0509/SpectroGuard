/* SpectroGuard frontend logic.
 * Views: landing (animated hero + CTA) and upload/analyze (hash '#/upload').
 * Auth: 3 free anonymous checks, then sign-in required (JWT in localStorage).
 */

const ALLOWED = ["mp4", "mov", "avi", "mkv", "webm"];

let videoId = null;
let pollTimer = null;
let pendingFile = null;

const $ = (id) => document.getElementById(id);

/* ================= Auth state ================= */
let token = localStorage.getItem("sg_token") || null;
let clientId = localStorage.getItem("sg_client_id");
if (!clientId) {
  clientId = (crypto.randomUUID && crypto.randomUUID()) || `cid-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  localStorage.setItem("sg_client_id", clientId);
}

function apiHeaders(extra) {
  const h = Object.assign({ "X-Client-Id": clientId }, extra || {});
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

/* ================= View routing ================= */
const landingView = $("landing-view");
const appView = $("app-view");
let counted = false;

function showView(name) {
  const isLanding = name !== "app";
  landingView.classList.toggle("hidden", !isLanding);
  appView.classList.toggle("hidden", isLanding);
  window.scrollTo({ top: 0 });
  if (isLanding) startCountUps();
}

function onHash() {
  showView(location.hash === "#/upload" ? "app" : "landing");
}

$("go-upload").addEventListener("click", () => (location.hash = "#/upload"));
$("back-home").addEventListener("click", () => (location.hash = "#/"));
$("brand").addEventListener("click", () => (location.hash = "#/"));
window.addEventListener("hashchange", onHash);

/* ================= Animated count-up stats ================= */
function startCountUps() {
  if (counted) return;
  counted = true;
  document.querySelectorAll("[data-count]").forEach((el) => {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || "";
    const decimals = target % 1 !== 0 ? 1 : 0;
    const duration = 1500;
    const t0 = performance.now();
    const fmt = (v) => (decimals ? v.toFixed(decimals) : Math.round(v)) + suffix;
    (function step(now) {
      const p = Math.min((now - t0) / duration, 1);
      el.textContent = fmt(target * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(step);
    })(t0);
  });
}

/* ================= Seamless news ticker ================= */
const tickerTrack = $("ticker-track");
tickerTrack.innerHTML += tickerTrack.innerHTML;

/* ================= Auth UI ================= */
function updateAuthUI() {
  const logged = Boolean(token);
  $("auth-btns").classList.toggle("hidden", logged);
  $("user-chip").classList.toggle("hidden", !logged);
  if (!logged) return;
  fetch("/api/auth/me", { headers: apiHeaders() })
    .then((r) => (r.ok ? r.json() : Promise.reject()))
    .then((data) => { $("user-email").textContent = data.user.email; })
    .catch(() => logout());
}

function logout() {
  token = null;
  localStorage.removeItem("sg_token");
  updateAuthUI();
}

/* ================= Auth modal ================= */
const modal = $("auth-modal");
let authTab = "signin";

function openAuthModal(note) {
  if (note) $("modal-note").textContent = note;
  setAuthTab(authTab);
  modal.classList.remove("hidden");
  setTimeout(() => $("auth-email").focus(), 50);
}

function closeAuthModal() {
  modal.classList.add("hidden");
  hide($("auth-error"));
}

function setAuthTab(tab) {
  authTab = tab;
  const signup = tab === "signup";
  $("tab-signin").classList.toggle("active", !signup);
  $("tab-signup").classList.toggle("active", signup);
  $("modal-title").textContent = signup ? "Create account" : "Sign in";
  $("modal-note").textContent = signup
    ? "Create an account to continue."
    : "Sign in to continue.";
  $("auth-submit").textContent = signup ? "Create account" : "Sign in";
  $("auth-password").autocomplete = signup ? "new-password" : "current-password";
}

$("signin-btn").addEventListener("click", () => { setAuthTab("signin"); openAuthModal(); });
$("signup-btn").addEventListener("click", () => { setAuthTab("signup"); openAuthModal(); });
$("modal-close").addEventListener("click", closeAuthModal);
modal.addEventListener("click", (e) => { if (e.target === modal) closeAuthModal(); });
$("tab-signin").addEventListener("click", () => setAuthTab("signin"));
$("tab-signup").addEventListener("click", () => setAuthTab("signup"));
$("logout-btn").addEventListener("click", logout);

$("auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  hide($("auth-error"));
  const email = $("auth-email").value.trim();
  const password = $("auth-password").value;
  const endpoint = authTab === "signup" ? "register" : "login";
  const btn = $("auth-submit");
  btn.disabled = true;
  btn.textContent = "Please wait...";
  try {
    const res = await fetch(`/api/auth/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = typeof data.detail === "string" ? data.detail : "Authentication failed.";
      $("auth-error").textContent = msg;
      show($("auth-error"));
      return;
    }
    token = data.token;
    localStorage.setItem("sg_token", token);
    closeAuthModal();
    updateAuthUI();
    if (pendingFile) {
      const f = pendingFile;
      pendingFile = null;
      uploadFile(f);
    }
  } catch {
    $("auth-error").textContent = "Network error. Please try again.";
    show($("auth-error"));
  } finally {
    btn.disabled = false;
    btn.textContent = authTab === "signup" ? "Create account" : "Sign in";
  }
});

/* ================= Upload / Analyze elements ================= */
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
    const res = await fetch("/api/upload", { method: "POST", body: form, headers: apiHeaders() });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail || {};
      if (typeof detail === "object" && detail.code === "login_required") {
        pendingFile = file;
        openAuthModal(detail.message || "Sign in to continue.");
        return;
      }
      throw new Error(typeof detail === "string" ? detail : "Upload failed");
    }
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
  $("votes").textContent = r.votes ? `${r.votes.REAL} / ${r.votes.FAKE}` : "-";

  hide(statusCard);
  show(resultCard);
  show(uploadCard);
}

/* ---------- Reset ---------- */
resetBtn.addEventListener("click", () => location.reload());

/* ---------- Backend health pill ---------- */
(async () => {
  try {
    const res = await fetch("/health");
    document.querySelector(".health").classList.toggle("ok", res.ok);
    document.querySelector(".health").classList.toggle("down", !res.ok);
  } catch {
    document.querySelector(".health").classList.add("down");
  }
})();

/* ---------- Init ---------- */
onHash();
updateAuthUI();
