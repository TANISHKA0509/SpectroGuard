# SpectroGuard — Complete Architecture Guide & Interview Q&A

A full, from-scratch explanation of how the project works, written so that a
**non-technical person can follow it** but with **enough depth for a technical
interviewer to be satisfied** when they cross-question you.

Every fact below is taken directly from the code in this repository. If you are
asked something not covered here, point at the exact file and function — that
answers it better than a memorized sentence.

---

## 0. The one-paragraph non-technical explanation

> SpectroGuard is a website where you upload a short video clip and it tells you
> whether the person in that video is **real** or **computer-generated (a deepfake)**.
>
> It works like this: instead of looking at every single picture (frame) in the
> video, it grabs a handful of frames spread evenly through the clip, zooms in on
> the face in each one, and sends those face images through a **neural network**
> (a program "taught" to recognize deepfakes by looking at tens of thousands of
> real and fake faces). Each frame votes "real" or "fake". The votes are counted
> and the majority wins — that becomes the final verdict, shown on screen with a
> confidence percentage.
>
> Behind the scenes there is a server (backend) that receives the upload, does
> the math, and gives the answer back; a page (frontend) that you actually see
> and click; and a small database that stores accounts so you can sign in. The
> whole thing is packaged as a "Docker" container so it can run on free cloud
> hosting, and it's live at a public URL.

---

## 1. System map — who talks to whom

```
                     ┌───────────────────────────────────────────────┐
                     │                 YOUR BROWSER                   │
                     │   index.html  (structure)                     │
                     │   style.css   (looks)                         │
                     │   config.js   (where the backend lives)       │
                     │   script.js   (all the logic)                 │
                     └───────────────┬───────────────────────────────┘
                                     │  HTTPS + JSON (REST API)
                     ┌───────────────▼───────────────────────────────┐
                     │               FASTAPI  (backend/app.py)       │
                     │                                               │
                     │   /api/upload        → save file + quota check │
                     │   /api/analyze/{id}  → start background thread │
                     │   /api/status/{id}   → progress polling        │
                     │   /api/result/{id}   → final verdict           │
                     │   /api/auth/*        → register / login / me   │
                     │   /health            → is it alive?            │
                     └───────────────┬───────────────────────────────┘
                                     │ calls
                     ┌───────────────▼───────────────────────────────┐
                     │          PREDICTION PIPELINE (CPU)             │
                     │                                               │
                     │  video_processor.py   OpenCV: open video,     │
                     │                       sample every 10th frame │
                     │                       (max 30), face-crop      │
                     │  inference.py         build batch → model →    │
                     │                       softmax → votes          │
                     │  model.py             pre-trained Xception     │
                     │                       (21.8M params, 250 MB)   │
                     └───────────────┬───────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  SQLite  (backend/database.py)  │
                    │   users table  (email + hash)   │
                    │   anon_usage   (free checks)    │
                    └─────────────────────────────────┘
```

Three kinds of storage, all different on purpose:

| Storage | What | Why it's like that |
|---|---|---|
| `uploads/` | the raw video files | big files; kept on disk, not in memory |
| `JOBS` (in-memory dict) | analysis state per video | fast, but **lost on restart** (documented trade-off) |
| `data/spectroguard.db` (SQLite) | accounts + free-check counters | real persistence using Python's built-in DB |

---

## 2. The full journey of ONE video (follow these 12 steps)

1. User drags a `.mp4` onto the page.
2. `script.js` checks the extension (`.mp4/.mov/.avi/.mkv/.webm`) and size (≤ 200 MB) *in the browser first* — instant feedback, no server round-trip.
3. The file is uploaded with `POST /api/upload` (multipart/form-data). The browser adds a header `X-Client-Id` (a random ID stored in the browser's localStorage) so the server can count free checks per browser.
4. FastAPI reads the file **in 1 MB chunks**, writing to `uploads/<video_id>.<ext>`, aborting with **413** if it exceeds 200 MB. This streaming matters: it never loads a 200 MB file fully into RAM.
5. If this is an anonymous visitor and they've already used **3 free checks**, the server rejects with **403 `{"code":"login_required"}`** — the frontend reacts by opening the sign-in modal, remembering the file, and retrying after login.
6. The server registers a **job** in the in-memory `JOBS` dict (`status: "uploaded"`, `progress: 0`) and returns a 12-character `video_id`.
7. The browser shows a preview (the server streams the file back via `/api/videos/{id}/video`) and enables **Analyze Video**.
8. User clicks Analyze → `POST /api/analyze/{video_id}`. The server flips the job to `"processing"` and spawns a **background thread** (`_process_video`) so the request returns instantly and the server isn't blocked.
9. The background worker:
   - asks `get_model()` to load the model **if not already loaded** (lazy loading, thread-safe with a lock),
   - `extract_frames()` opens the file with OpenCV, reads frames, keeps **every 10th**, stops at **30**,
   - for each frame, `crop_face()` finds the biggest face with a Haar cascade and crops it (falls back to the whole frame if no face),
   - `preprocess_frame()` converts BGR→RGB, resizes to **224×224**, divides by 255, normalizes with mean/std **0.5**, transposes to CHW, adds the batch dimension,
   - all frames are concatenated into **one tensor** and run through the model in a single `forward()` pass under `torch.no_grad()` (no gradient tracking → faster, less memory),
   - `softmax` turns logits into probabilities; each frame's class is the argmax (`0`=REAL, `1`=FAKE); its confidence is the winning probability.
10. `aggregate_video_prediction()` counts how many frames voted REAL vs FAKE. **Majority wins** (ties go to REAL). Confidence = the *average probability the model assigned to the winning class* across all frames.
11. Meanwhile the browser **polls** `GET /api/status/{id}` **every 1000 ms**, updating a progress bar (5% → 80% → 100%). When status is `done`, it calls `GET /api/result/{id}` and `renderResult()` animates the verdict, confidence bar, frames analyzed, faces found, processing time, and the REAL/FAKE vote counts.
12. The verdict screen also shows a disclaimer that this is an educational prototype, not a forensic tool.

---

## 3. Deep dive — every file

### 3.1 `frontend/index.html`
Pure HTML, no framework. Contains two "views" (landing page + upload app) toggled with the `hidden` class via hash routing (`#/upload`). Also the auth modal and the backend-connect banner.

### 3.2 `frontend/style.css`
Single stylesheet, ~430 lines. CSS variables for the dark "cyber" theme (`--cyan`, `--green`, `--red`), animated gradient orbs, a scan-line, count-up stats, news cards, a marquee ticker, and the modal. `.hidden { display:none !important }` is the master visibility switch.

### 3.3 `frontend/config.js`
Two lines of real logic: `window.SG_API_BASE = ""`. Because the frontend can be hosted on Vercel while the backend lives on Render, `script.js` reads `SG_API_BASE` and prefixes every API call with it. `""` means "same server that served this page".

### 3.4 `frontend/script.js`
All interactivity. Key functions:
- `apiHeaders()` — adds `X-Client-Id` and, if logged in, `Authorization: Bearer <token>`.
- `onHash()`/`showView()` — hash routing between landing and app.
- `uploadFile()` — uploads, handles the `login_required` retry flow.
- `analyzeBtn` handler + `pollStatus()` — polls every second.
- `fetchResult()`/`renderResult()` — renders the verdict.
- `checkHealth()` — pings `/health` with a 4-second timeout; if it fails, shows the **"Please wait while we connect the backend to the frontend…"** banner and retries every 5 s (every 30 s once healthy).
- Auth functions — `updateAuthUI`, `openAuthModal`, `setAuthTab`, login/logout.

### 3.5 `backend/app.py` — the FastAPI application (423 lines)
The entry point. Defines every route, the job store, quota logic, and static serving. Details:

- **Lazy model loading** (`get_model()`): the model is *not* loaded at startup. `lifespan` only initializes the DB. On the first analysis, `get_model()` loads it inside a `threading.Lock` so two concurrent analyses can't load it twice. This keeps the container's RAM usage low enough for Render's free 512 MB tier to boot.
- **`_process_video`**: the background worker. Runs `run_inference` then `aggregate_video_prediction`, attaches metadata, and flips the job to `done`/`error`. Any exception is caught and stored in the job (`status:"error"`, `error:<message>`).
- **`JOBS` / `JOBS_LOCK`**: an in-memory dict guarded by a lock, because the dict is shared between FastAPI's threadpool threads and background threads.
- **Static serving**: `app.mount("/", StaticFiles(directory=frontend, html=True))` is **registered last**, so all `/api/*` routes win and anything else (like `/`) serves the frontend. This is how the same server runs both the API and the website locally.

### 3.6 `backend/video_processor.py` — OpenCV pipeline
- `open_video()` — `cv2.VideoCapture`; raises `UnsupportedVideoError` if unreadable.
- `extract_frames()` — iterates the whole file, keeps frames where `frame_index % 10 == 0`, stops at 30. Returns frames + metadata (`total_frames`, `fps`, `frames_sampled`, `sampling_ratio`).
- `crop_face()` — Haar cascade `haarcascade_frontalface_default.xml` (ships inside OpenCV), `scaleFactor=1.1`, `minNeighbors=5`, `minSize=64px`. Picks the **largest** face box (biggest area), pads it by 20% on each side, crops. No face → returns the full frame with `face_found=False`.
- `preprocess_frame()` — the exact training-time transform the model expects (see §4).

### 3.7 `backend/inference.py`
- `build_batch()` — preprocesses each frame (optionally face-cropping) and `torch.cat`s them into one batched tensor; counts how many had faces.
- `run_inference()` — end-to-end: extract → batch → `model(batch)` under `no_grad` → `softmax(dim=1)` → per-frame argmax + confidence. Times itself.
- `aggregate_video_prediction()` — the decision math (§4).

### 3.8 `backend/model.py`
- Downloads the weights (`huzaifanasirrr/faceforge-detector` → `models/detector_best.pth`, ~250 MB) using `huggingface_hub` with a plain-`urllib` fallback, auto-runs on first use.
- `DeepfakeDetector` = timm `xception` backbone (`pretrained=False`, `num_classes=0` → raw 2048-d features) + custom head: `Dropout(0.5) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→2)`.
- `load_model()` — downloads if needed, loads the checkpoint with `weights_only=True` (safe-loading), `strict=True`, `eval()`, caps PyTorch threads at `min(2, cpu_count)` to save memory.

### 3.9 `backend/auth.py`
- PBKDF2-SHA256 password hashing (100,000 iterations, 16-byte random salt per user) — stdlib only.
- JWT tokens (PyJWT, HS256), 7-day expiry. Secret from env `SPECTROGUARD_SECRET`, dev fallback otherwise.
- `verify_password` uses `hmac.compare_digest` — constant-time comparison (timing-attack safe).

### 3.10 `backend/database.py`
- SQLite via stdlib `sqlite3`. Tables auto-created on connect (self-healing): `users(id, email UNIQUE, password_hash, salt, created_at)` and `anon_usage(client_id PK, used)`.
- Every helper opens a fresh connection, does its work, and closes it (simple + safe for threads). A module-level lock serializes writes.

### 3.11 `backend/utils.py`
Extension whitelist, `ensure_dir`, 12-char `uuid4` video IDs, file-extension helpers, a `timing()` context manager, `seconds_to_hms()`.

### 3.12 Deployment files
- `Dockerfile`: `python:3.12-slim`; installs **CPU-only** torch from the PyTorch CPU index, with a **fallback to PyPI for ARM64** (Oracle free VMs are ARM); downloads the model at **build time**; exposes 7860; runs `uvicorn --port ${PORT:-7860}`.
- `render.yaml`: Render blueprint — Docker runtime, free plan, health check `/health`, port 7860.
- `scripts/download_model.py` — pre-fetch weights.
- `scripts/create_test_video.py` — generates a synthetic MP4 (moving shapes, **no face**) to smoke-test the pipeline.
- `scripts/deploy_space.py` / `scripts/deploy_oracle.sh` — HF Spaces upload / Oracle VM one-shot deploy.

---

## 4. The math of a verdict (be precise here)

Per frame `i`, the model outputs 2 logits. `softmax([a,b]) = [e^a/(e^a+e^b), e^b/(e^a+e^b)]` gives `P(REAL)` and `P(FAKE)`.

- **Frame class** = `argmax`: `0 → REAL`, `1 → FAKE`.
- **Frame confidence** = the probability of the winning class.
- **Video verdict** = **majority vote**: `FAKE` if `votes_fake > votes_real`, else `REAL` (ties → REAL).
- **Video confidence** = mean over all frames of `P(winning class)` = `probabilities[:, final_idx].mean()`.

Example (from README): 4 frames, all FAKE → `{REAL:0, FAKE:4}`, confidence ≈ 0.87.

Why this is defensible in an interview:
- Voting is **robust** to a few misclassified frames.
- Average-probability confidence is more granular than "4/4".
- It is **interpretable** and explainable on the UI (votes shown as `Real / Fake`).
- Honest limitation: it ignores **temporal** structure (order of frames) — each frame judged independently. Future work: LSTM over frame features.

---

## 5. Auth, security & the free-check policy

- **Anonymous quota**: each browser generates a random `X-Client-Id` (persisted in localStorage). The server stores a counter per ID. **3 free uploads**, then 403. Only *successful* uploads count (the increment happens after the file is written).
- **Register**: email validated by regex, password ≥ 6 chars, duplicate → 409. Password stored as `salt + PBKDF2(100k, SHA-256)`; the raw password is never stored.
- **Login**: verifies against the stored hash with a constant-time compare → JWT (HS256, 7 days, claims `sub`/`iat`/`exp`).
- **Authenticated requests**: `Authorization: Bearer <token>`. `current_user_or_none()` decodes it and looks up the user. Signed-in users have no quota limit.
- **CORS**: `allow_origins=["*"]` so the Vercel-hosted frontend can call the Render-hosted backend from a different origin.
- **Honest security caveats** (say these out loud — they make you look senior): the JWT secret has a dev default (env var should be set in production); uploads are stored unencrypted; there is no rate limiting; SQLite uses parameterized queries (SQL-injection safe) but there is no email verification; CSRF is not a concern here because auth uses a bearer header, not cookies.

---

## 6. Deployment topology (as of today)

- **Frontend**: Vercel (`https://spectro-guard-murex.vercel.app`) — static files, always-on, free. It talks to the backend through `config.js`'s `SG_API_BASE`.
- **Backend**: Render (`https://spectroguard-r5rh.onrender.com`) — Docker container, free tier. **Sleeps after 15 min idle** → first request after sleep takes ~1 min (the frontend's connect banner explains this to users). The `/health` endpoint is the platform health check.
- **Database**: SQLite inside the container's disk → resets on redeploy (documented, acceptable for a prototype).
- **GitHub**: `TANISHKA0509/SpectroGuard` is the source of truth; pushing to `main` auto-deploys both hosts.

---

## 7. Known limitations (own them — interviewers respect this)

1. **In-memory job store** — jobs/videos vanish on restart; no multi-worker scaling.
2. **Single-frame analysis** — no temporal features; frames voted independently.
3. **Domain-limited model** — trained on FaceForensics++ c40; accuracy drops on heavy compression, other manipulation types, unusual angles, no clear face.
4. **Haar cascade** — fast but imprecise face detection (good enough for a demo).
5. **CPU latency** — seconds per video on CPU; free cloud CPU is shared.
6. **Model param count discrepancy** — README table says 20.8M; the running model reports **21.86M** params and the frontend shows 21.8M. The 20.8M figure in README is the stale/copied value. The code-computed value is 21.86M. (Fix: update README table.)
7. **Video preview on split deployment** — the upload response's `video_url` is relative; `script.js` now prefixes it with `SG_API_BASE` so the preview works cross-origin.
8. **No tests committed** — smoke-tested manually + the generated test video; no unit/integration test suite in the repo (worth adding).

---

## 8. The 60-second summary you can say out loud

> "SpectroGuard is an end-to-end deepfake detection prototype. The frontend is a
> vanilla JS single-page app with hash routing; the backend is FastAPI with an
> upload endpoint that streams files to disk, a quota + JWT auth layer over
> SQLite, and an async job system using a background thread. Analysis works by
> sampling every tenth frame (max 30), face-cropping with OpenCV's Haar cascade,
> preprocessing to 224×224 normalized tensors, batch-inferring through a
> pre-trained 21.8M-parameter Xception model under torch.no_grad(), and
> aggregating per-frame predictions by majority vote into a Real/Fake verdict
> with average confidence. The model is lazy-loaded and thread-capped to fit
> free-tier memory, the app is containerized with Docker, and it's deployed with
> the frontend on Vercel and the backend on Render, with a health poll and a
> connect banner to handle Render's cold starts. The known trade-offs are an
> in-memory job store, frame-independent voting with no temporal modeling, and a
> dataset-limited model — all documented as prototype limitations."
