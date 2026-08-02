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

## 8. The interview Q&A (start here when preparing)

### A. Non-technical / elevator

**Q1. What does the project do in one sentence?**
Upload a video clip and it tells you whether the person is real or a deepfake, with a confidence score, in a few seconds.

**Q2. How does it "know" it's a deepfake?**
It doesn't "know". It was trained on thousands of real and fake faces, learned the subtle visual differences (compression artifacts, blending edges, lighting), and applies that learning to the faces in your video. It's a statistical guess with a confidence number.

**Q3. What is a "frame"?**
A video is just a rapid sequence of still pictures — typically 24–30 per second. Each still picture is a frame.

**Q4. Why don't you check every frame?**
A 1-minute video at 30 fps has 1,800 frames. Running a neural network on all of them on a free CPU server would take minutes. Sampling every 10th (max 30) gives a representative spread in ~3–6 seconds.

**Q5. What is a neural network here?**
A function with ~21.8 million adjustable numbers ("weights") that maps a 224×224 face image to two scores: how real-like and how fake-like. The weights were learned during training, not written by a person.

**Q6. Is this production/forensic grade?**
No. It's an educational prototype. Good enough to demonstrate the engineering; not admissible evidence.

### B. Architecture

**Q7. What are the main components?**
Browser frontend (vanilla HTML/CSS/JS), FastAPI backend, an OpenCV+PyTorch inference pipeline, SQLite, and Docker for deployment.

**Q8. Why FastAPI?**
Async-ready, fast, auto-generates OpenAPI docs (`/docs`), typed with Pydantic, and trivially serves static files too — one server for API + UI.

**Q9. How do frontend and backend communicate?**
HTTP REST + JSON. The frontend calls `/api/*` endpoints with `fetch()`. On a split deployment, every URL is prefixed with `SG_API_BASE` from `config.js`.

**Q10. Why does the frontend poll instead of getting a WebSocket push?**
Simpler and robust: `setInterval` + `GET /api/status`. For a single-user demo, polling every 1 s is more than enough. WebSockets are listed as future work for real live-call detection.

**Q11. How does a request "start analysis and not block"?**
`POST /api/analyze` spawns a Python `threading.Thread` (daemon) that does the heavy work, then returns immediately. The browser polls for progress.

**Q12. Why is the job store in memory?**
Prototype trade-off for zero setup. It means jobs die with the process. A production version would use Redis/Postgres + a worker queue (Celery/RQ).

**Q13. What locks exist and why?**
`JOBS_LOCK` guards the shared `JOBS` dict (FastAPI threadpool + worker threads). `MODEL_LOCK` guards lazy model loading (two analyses must not load the 250 MB model simultaneously). `database._lock` serializes SQLite writes. All are `threading.Lock`.

**Q14. Why lazy-load the model instead of at startup?**
Render's free tier has 512 MB RAM. Loading torch + the model at boot can exceed it and crash the container permanently. Loading on first analysis keeps boot light and `/health` green.

### C. Video processing (OpenCV)

**Q15. How many frames are sampled and why 10?**
Every 10th frame, capped at 30. 10 gives good temporal spread; 30 keeps latency and memory bounded.

**Q16. What if a video has fewer than 10 frames?**
It takes whatever frames exist. If zero readable frames, `UnsupportedVideoError` → a friendly error on screen.

**Q17. How is the face found?**
OpenCV's bundled Haar cascade (`frontalface_default.xml`). It's a pre-trained cascade of simple rectangular features — fast, no neural net needed.

**Q18. What if no face is detected?**
The whole frame is used as-is (`face_found=False`). The system still runs; `faces_detected` shows 0. The model may still classify, just less reliably.

**Q19. Why crop the face at all?**
The model was trained on faces. Feeding it just the face region focuses it on the manipulated area and avoids distraction from the background.

**Q20. Which face is picked when several are present?**
The largest box by area (`max(faces, key=area)`), assuming it's the call's subject. It's padded by 20% so the forehead/chin aren't cut.

**Q21. What exact preprocessing does each frame get?**
BGR→RGB, resize to 224×224 (INTER_AREA), `/255` → [0,1], normalize `(x-0.5)/0.5` → [-1,1], transpose HWC→CHW, unsqueeze to a batch of 1. These constants **must match training** — the model card says 224×224, mean=std=0.5.

### D. Machine learning

**Q22. Which model?**
A timm Xception (XceptionNet, Chollet 2017) backbone with a custom two-layer MLP head, weights from `huzaifanasirrr/faceforge-detector` (trained on FaceForensics++ c40, Real vs Fake).

**Q23. How many parameters?**
The running code computes **21.86M** (the README table's 20.8M is stale — be ready for this exact question).

**Q24. Why Xception specifically?**
It was the backbone the checkpoint author used; we don't retrain, we reuse the weights. That keeps the project "inference-only".

**Q25. Why is `num_classes=0` used?**
timm's `create_model("xception", num_classes=0)` drops the default classifier and returns raw 2048-d features, so we can attach our own head that exactly matches the checkpoint's keys (`xception.*`, `classifier.*`).

**Q26. What is `torch.no_grad()`?**
It disables gradient computation — we're only doing inference, not training. It saves memory and speeds up forward passes.

**Q27. How is a batch formed?**
All preprocessed frames are `torch.cat`ed into one tensor `[N, 3, 224, 224]` and passed through the model once — one forward pass for all frames, far faster than N separate calls.

**Q28. Why softmax?**
Logits are unbounded; softmax squashes them into probabilities in [0,1] that sum to 1, so we can report a meaningful confidence.

**Q29. Why majority vote rather than averaging the scores?**
Voting is robust and interpretable ("4 of 4 frames voted FAKE"). Averaging probabilities is the alternative; we show both (votes *and* average confidence).

**Q30. What does "c40" mean in FaceForensics++?**
The compression level used to create the dataset (c40 = medium). Real-world videos with different compression behave differently — a documented limitation.

**Q31. Does the model see motion across frames?**
No. Each frame is independent. Aggregating by vote is a post-processing step. Temporal modeling is explicitly listed as future work.

### E. The API & data flow

**Q32. What does `POST /api/upload` do, in order?**
(1) validate extension → (2) check free quota for anonymous → (3) stream file to disk in 1 MB chunks, enforcing 200 MB → (4) increment quota if anonymous → (5) register job → (6) return `{video_id, video_url, used_free, free_limit}`.

**Q33. Why stream in 1 MB chunks?**
So a 200 MB file never fully loads into server RAM. `await file.read(1024*1024)` reads chunk by chunk and aborts with 413 the moment the total exceeds the limit.

**Q34. Why is the size check both client- and server-side?**
Client-side for instant UX; server-side because the client can't be trusted (someone could POST via curl). Never trust the client.

**Q35. What is `video_id`?**
`uuid.uuid4().hex[:12]` — a 12-char random hex string, the key of the job. It's in every subsequent URL.

**Q36. What statuses does a job go through?**
`uploaded → processing → done` (or `error`).

**Q37. What happens if you analyze a video that's already `processing` or `done`?**
409 Conflict — analysis can only be started once per job.

**Q38. What does `/api/status` return?**
`status`, `message`, `progress` (0–100), `error`, and elapsed seconds. The frontend uses `message` for the spinner text and `progress` for the bar.

**Q39. Why does `/api/result` return 202 sometimes?**
202 Accepted = "not ready yet"; the frontend only calls it when status is `done`, so in practice it's always ready.

**Q40. How is the video preview streamed?**
`FileResponse` with the correct `Content-Type` from `mimetypes`, at `/api/videos/{id}/video`.

**Q41. What does DELETE do?**
Removes the file from disk and the job from the dict; 409 if currently processing.

### F. Auth & security

**Q42. How are passwords stored?**
Random 16-byte salt + PBKDF2-HMAC-SHA256 with 100,000 iterations. Salt and hash stored as hex; raw password never persisted.

**Q43. Why PBKDF2 and not plain hashing?**
Password hashing must be slow, salted, and one-way. PBKDF2 is a battle-tested KDF that does this with just the Python stdlib (no extra dependency).

**Q44. What is `hmac.compare_digest` for?**
Constant-time comparison — prevents timing attacks that compare string equality character-by-character.

**Q45. What is a JWT?**
A signed JSON object. Ours contains `sub` (user id), `iat` (issued at), `exp` (7 days). Signed with HS256 using the secret; the server verifies the signature before trusting it. Stateless — no session storage needed.

**Q46. How is the token sent on later requests?**
`Authorization: Bearer <token>` header, added by `apiHeaders()` from localStorage.

**Q47. What happens when a token expires?**
`decode_token` returns None → treated as anonymous (or 401 on `/me`). The frontend catches it and effectively logs the user out.

**Q48. How is the 3-free-checks quota enforced?**
Per anonymous browser via `X-Client-Id`. Server stores `anon_usage.used` per ID. When `used >= 3` and no valid token → 403. Clearing localStorage "resets" it (a client-controlled limit — acceptable for a demo, trivially bypassable; say this honestly).

**Q49. Is the app SQL-injection safe?**
Yes — every query uses SQLite parameter binding (`?` placeholders), never string concatenation.

**Q50. What about XSS/CSRF?**
No user content is rendered as HTML (the verdict is text from the model). Auth uses a header, not cookies, so classic CSRF doesn't apply. Uploaded files are served from the same origin only as `<video>` sources.

**Q51. Is the JWT secret safe?**
Only if `SPECTROGUARD_SECRET` env var is set. The code has a dev default. For a demo it's fine; for production it's the first thing to configure.

### G. Database

**Q52. Why SQLite?**
Zero setup, zero extra dependency (stdlib), a single file, works fine for this scale. The DB is real (survives restarts locally) even if job state is not.

**Q53. What tables exist and what are the columns?**
`users(id TEXT PK, email TEXT UNIQUE, password_hash TEXT, salt TEXT, created_at REAL)` and `anon_usage(client_id TEXT PK, used INTEGER)`.

**Q54. What does "self-healing schema" mean?**
`_ensure_schema` runs `CREATE TABLE IF NOT EXISTS` on every connection, so even if the DB file is deleted while running, the next query recreates the tables — the app never 500s.

**Q55. Why open a new connection per call instead of one shared?**
SQLite connections aren't great across threads; opening/closing per call with a module lock is simple and correct for this scale.

**Q56. Where does the DB file live and when does it reset?**
`data/spectroguard.db`. On Render/Docker, the container's disk is ephemeral — a redeploy wipes accounts/quota. Documented as acceptable for a prototype.

### H. Concurrency & reliability

**Q57. FastAPI is async; the worker is a thread — why is that OK?**
Sync route handlers run in a threadpool; the background analysis runs in its own daemon thread. The shared `JOBS` dict is guarded by `JOBS_LOCK`. GIL means Python threads share CPU, but the heavy work is torch/OpenCV which release the GIL during native calls.

**Q58. What if two videos are analyzed at once?**
Both threads run. `MODEL_LOCK` ensures the model is loaded once and shared (it's read-only in `eval()` mode — safe for concurrent forward passes).

**Q59. What happens if the analysis crashes mid-way?**
The `try/except` in `_process_video` sets `status:"error"` and stores the message; the frontend polling shows the error and lets the user retry with a new file.

**Q60. What happens on a server restart mid-analysis?**
The job (and the in-memory store) is lost; the browser's poll gets 404 and shows "Video not found". Accepted prototype behavior.

### I. Frontend

**Q61. How does the page decide which view to show?**
Hash routing: `location.hash === "#/upload"` shows the app; anything else shows the landing page. No page reload needed (`hashchange` listener).

**Q62. What is the connect banner and how does it work?**
`checkHealth()` fetches `/health` with a 4 s `AbortController` timeout. If it fails (backend sleeping/down), it shows "Please wait while we connect the backend to the frontend…" and retries every 5 s; once healthy it hides and re-checks every 30 s.

**Q63. Why `config.js` separate from `script.js`?**
So the backend URL can change without editing logic — needed because the frontend (Vercel) and backend (Render) are different origins. Same-origin deployments just leave it `""`.

**Q64. What does `apiHeaders()` do?**
Sends `X-Client-Id` always, and `Authorization: Bearer` when logged in. It's the single place auth headers are injected.

**Q65. How is the free-check UI handled now?**
The quota chip UI was removed from the frontend; the backend still enforces the limit. When a 403 `login_required` arrives, the file is stashed in `pendingFile`, the sign-in modal opens, and the upload auto-retries after a successful login.

**Q66. Why `crypto.randomUUID()` for the client id?**
A per-browser unique ID that survives refreshes (stored in localStorage) so the server can count free checks per browser.

### J. Model & performance numbers (know these cold)

**Q67. Model size / parameters / input / classes.**
~250 MB file; **21.86M** params; 224×224 RGB normalized to [-1,1]; classes `0=REAL`, `1=FAKE`.

**Q68. How long does inference take?**
~3–6 s for a 30-frame sample on a normal CPU (README). On Render's free 0.1 CPU it's slower. The API times it (`inference_time`, `processing_time`) and the UI shows it.

**Q69. Why cap torch threads at 2?**
Fewer threads = less RAM and no oversubscription on a 1-vCPU free container; the model is small enough that this barely hurts speed.

**Q70. How is `model_loaded` reported in `/health`?**
`hasattr(app.state, "model")` — `false` until the first analysis triggers lazy loading. The health check still returns 200 (`status:"ok"`), which is what Render checks.

### K. Deployment

**Q71. Where is the app currently deployed?**
Frontend on Vercel (free), backend on Render (free), both deployed from the GitHub repo. GitHub is the source of truth; pushes auto-deploy.

**Q72. What does the Dockerfile do?**
Base `python:3.12-slim` → install CPU-only torch (PyPI fallback for ARM64) → copy code → download the model at build → expose 7860 → run uvicorn on `$PORT` (7860 default).

**Q73. Why download the model at build time?**
So the container starts instantly at runtime (no slow first-request model download).

**Q74. Why is the model git-ignored?**
250 MB binaries shouldn't live in git (GitHub limit, slow clones). `.gitignore` blocks `models/*.pth`; the Dockerfile/startup downloads it instead.

**Q75. Why port 7860?**
Render passes `$PORT`; HF Spaces expects 7860. `CMD` uses `${PORT:-7860}` so the same image works on both.

**Q76. Why does Render free sleep, and what's the fix?**
Free tier spins down after 15 min idle; first request wakes it (~1 min). Fixes: cron ping every ~14 min, or pay for Starter ($7/mo). The frontend's connect banner was built exactly to handle this.

**Q77. What does `render.yaml` configure?**
Docker runtime, free plan, repo, health check path `/health`, env `PORT=7860`. Picked up automatically when you deploy a "Blueprint".

**Q78. Why the ARM fallback in the Dockerfile?**
Oracle's free VMs are ARM64; the PyTorch CPU index historically lacked aarch64 wheels, so the build tries the CPU index then falls back to PyPI (which ships aarch64 wheels).

**Q79. How does CORS fit deployment?**
Vercel (origin A) calling Render (origin B) is cross-origin. `allow_origins=["*"]` lets the browser accept the response. Without it, the API calls would be silently blocked by the browser.

### L. "Gotcha" design-decision questions

**Q80. Why not just upload and auto-analyze in one request?**
Analysis takes seconds. Tying it to the HTTP request would make the client wait or time out. Separating upload → analyze → poll is the classic async job pattern.

**Q81. Why a thread and not a process/queue?**
For a single-node demo a daemon thread is the simplest correct choice. A message queue (Celery/RQ/Redis) is the scaling path, listed as future work.

**Q82. Why store the job in a dict and not the DB?**
Jobs are transient runtime state (video + progress), while users/quota are persistent. Mixing them would complicate the schema. Documented trade-off.

**Q83. Why is `FRAME_SKIP` configurable via parameters even though it defaults to 10?**
So tests and future callers can adjust sampling without code changes — good API hygiene.

**Q84. Why does `safe_extension` fall back to `.mp4`?**
Because the extension was already validated before upload; the fallback is defensive so a weird-but-allowed name never produces an unsafe path.

**Q85. Why `weights_only=True` in `torch.load`?**
Loading arbitrary pickles is a security risk (RCE). `weights_only=True` restricts to tensors — the safe-loading option for inference.

**Q86. Why does the backend serve the frontend too, if Vercel exists?**
Locally (and on a single-server deploy) it's one process — simplest. The split deployment (Vercel + Render) is the production-like arrangement, enabled by `config.js`.

**Q87. What was the "connect banner" problem you actually solved?**
Render free cold starts (~1 min). Without the banner, users see a frozen page and think the site is broken. The banner explains the wait, and the `/health` poll hides it the moment the backend answers.

**Q88. What would you do differently in production?**
Persistent DB for jobs (Postgres/Redis), a real queue + workers, object storage for videos (S3), HTTPS-only + env secrets, rate limiting, email verification, a robust face detector (MTCNN/YuNet), temporal modeling, and a proper test suite.

### M. Test-your-understanding micro-questions

**Q89. If you upload a 3-second video at 12 fps (36 frames), how many frames are analyzed?**
Every 10th frame: indices 0,10,20,30 → **4 frames** (the README example shows exactly this: `frames_analyzed: 4`, `total_frames: 36`).

**Q90. If votes are REAL=2, FAKE=2, what is the verdict?**
REAL (ties go to REAL by `final_idx = 1 if votes_fake > votes_real else 0`).

**Q91. A user uploads their 4th video anonymously. What HTTP status and what happens in the UI?**
403 with `{"detail":{"code":"login_required",...}}` → the sign-in modal opens, the file is remembered, and after sign-in the upload continues automatically.

**Q92. Why is the 4th check blocked but the count only increments on success?**
`increment_anon_used` runs *after* the file is successfully written, so failed/canceled uploads don't burn quota.

**Q93. What's in the `/health` response before the first analysis?**
`{"status":"ok","model_loaded":false,"device":"n/a","model":"huzaifanasirrr/faceforge-detector"}`. `model_loaded` flips to `true` after the first run.

**Q94. What are the allowed upload extensions?**
`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` (case-insensitive; primary target MP4).

**Q95. What is the upload size limit and where is it enforced?**
200 MB — in the browser (before upload) *and* server-side (413 during chunked write).

**Q96. What is `seconds_to_hms` used for?**
Formatting elapsed time as "m ss.s"; present in utils but not wired into the UI (processing time is shown raw in seconds) — a small leftover.

**Q97. If the model is on CPU and 30 frames are batched, what is the tensor shape?**
`[30, 3, 224, 224]`, fp32.

**Q98. Why does the frontend show "Faces found" as a number?**
The worker counts frames where a face was actually cropped (`face_found`), surfaced as `faces_detected` — tells the user whether the detector found faces at all.

**Q99. What happens if you hit `/api/analyze/abc123` for a job that doesn't exist?**
404 "Video not found" from `get_job()`.

**Q100. What is the difference between `inference_time` and `processing_time`?**
`inference_time` = time inside `run_inference` (extract+batch+forward). `processing_time` = total from the start of `_process_video` to the completed result (includes aggregation). They're close; the difference is overhead.

---

## 9. The 60-second summary you can say out loud

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
