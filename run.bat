@echo off
REM SpectroGuard quick-start for Windows.
REM Activates the venv and launches the API on port 8001 (8000 is commonly taken).

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found. Run:
  echo     python -m venv .venv
  echo     .venv\Scripts\python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
  echo     .venv\Scripts\python -m pip install -r requirements.txt
  exit /b 1
)

echo [INFO] Starting SpectroGuard on http://localhost:8001
".venv\Scripts\python.exe" -m uvicorn backend.app:app --host 0.0.0.0 --port 8001
