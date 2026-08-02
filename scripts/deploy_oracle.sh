#!/usr/bin/env bash
#
# One-shot deploy for SpectroGuard on an Oracle Cloud free ARM VM.
#
# Runs as root (ssh ubuntu@IP then: sudo bash scripts/deploy_oracle.sh).
# Installs Docker, builds the image (downloading the model at build time),
# and runs the container as an always-on service that restarts on boot.
#
# Prereqs (done in the Oracle Cloud console before running this):
#   * Ubuntu 22.04+ ARM VM with at least 4 OCPU / 16 GB RAM
#   * Ingress rules open for ports 80, 443 (and 22 for SSH)
#
# After it finishes, SpectroGuard is live at:
#   http://<VM_PUBLIC_IP>/            (health check at /health)
#
set -euo pipefail

set -x

echo "[1/5] Installing Docker..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y docker.io
systemctl enable --now docker

echo "[2/5] Cloning the project..."
cd /opt
rm -rf spectroguard
git clone --depth 1 https://github.com/TANISHKA0509/SpectroGuard.git
cd spectroguard

echo "[3/5] Building the image (downloads ~250 MB model at build time)..."
docker build -t spectroguard:latest .

echo "[4/5] Starting the container (always-on, restart on boot)..."
docker rm -f spectroguard >/dev/null 2>&1 || true
docker run -d \
  --name spectroguard \
  --restart unless-stopped \
  -p 80:7860 \
  spectroguard:latest

echo "[5/5] Waiting for the app to come up..."
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

PUBLIC_IP=$(curl -fsS --max-time 5 http://checkip.amazonaws.com || echo "YOUR_VM_IP")
echo ""
echo "Done! SpectroGuard is live at:"
echo "    http://${PUBLIC_IP}/"
echo "Health check: http://${PUBLIC_IP}/health"
echo ""
echo "View logs:   docker logs -f spectroguard"
