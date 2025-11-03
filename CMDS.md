# Sesame CSM Deployment

This guide documents how to configure an Ubuntu EC2 instance for GPU
inference, expose multiple services via Cloudflare tunnels, manage
processes with PM2, and run HuggingFace-based LLM services. Steps
include both commands and reasoning.

---

## 1) SSH into EC2

Proper key permissions are required; otherwise SSH will refuse to use
the key.

```bash
chmod 600 sesame.pem
ssh -i "sesame.pem" ubuntu@54.209.210.181
```

## 2) Install CUDA

Update repository list:

```bash
sudo apt update
```

Optional checks:

```bash
nvcc --version
nvidia-smi
```

Install drivers + CUDA:

```bash
sudo apt install -y nvidia-driver-535 nvidia-utils-535 nvidia-cuda-toolkit
```

Reboot:

```bash
sudo reboot
```

## 3) Install Python 3.10

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip
```

## 4) Clone & Create Virtual Environment

```bash
git clone https://github.com/hussain1441/csm-complete.git
cd csm-complete
python3.10 -m venv .env
source .env/bin/activate
pip install --upgrade pip
```

## 5) Cloudflare Setup

Download and install `cloudflared`.

```bash
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
rm cloudflared.deb
cloudflared --version
```

Authenticate:

```bash
cloudflared tunnel login
```

> Follow browser prompt.

Create tunnel:

```bash
cloudflared tunnel create <name>
```

Find credentials file:

```bash
ls /home/ubuntu/.cloudflared/
```

### Create config.yml

`~/.cloudflared/config.yml`

```yaml
tunnel: TUNNEL_ID
credentials-file: ~/.cloudflared/TUNNEL_ID.json

ingress:
    - hostname: voice.hussainkazarani.site
      service: http://localhost:8000
    - hostname: tts.hussainkazarani.site
      service: http://localhost:8100
    - hostname: ttsai.hussainkazarani.site
      service: http://localhost:8200
    - hostname: selection.hussainkazarani.site
      service: http://localhost:8300
    - service: http_status:404
```

### DNS

In Cloudflare dashboard → DNS → Create `CNAME`

    <subdomain> -> <TUNNEL_ID>.cfargotunnel.com

## 6) Install PM2 + Node

```bash
sudo apt install -y curl
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
sudo npm install -g pm2
node -v
npm -v
pm2 --version
```

## 7) csm-stream Setup

```bash
cd csm-stream
pip install -r requirements.txt
```

Install audio deps:

```bash
sudo apt install -y ffmpeg
sudo apt install -y libportaudio2 libportaudiocpp0 portaudio19-dev
sudo apt install -y lsof
```

## 8) HuggingFace Login

Get token from HF → login

```bash
huggingface-cli login
```

## 9) Configs

```bash
python setup.py
```

## 10) Download Model

```bash
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF --include "Llama-3.2-1B-Instruct-Q6_K_L.gguf" --local-dir ./models --local-dir-use-symlinks False \
&& mv ./models/Llama-3.2-1B-Instruct-Q6_K_L.gguf ./models/llama32-1b.gguf
```

## 11) Run tunnel with PM2

Check tunnel:

```bash
cloudflared tunnel list
pm2 ls
```

Start tunnel:

```bash
pm2 start cloudflared --name sesame-tunnel -- tunnel --config ~/.cloudflared/config.yml run <name>
```

## 12) Run Servers

```bash
pm2 start ../.env/bin/python --name tts -- tts-server.py
pm2 start ../.env/bin/python --name ttsai -- tts-ai-server.py
pm2 start ../.env/bin/python --name selection -- tts-selection-server.py
pm2 start ../.env/bin/python --name voice -- tts-voice-server.py
```

Voice forwarding requirement:

```bash
ssh -L 8000:localhost:8000 -i "sesame.pem" ubuntu@54.209.210.181
```

## 13) Useful Monitoring

Watch GPU:

```bash
watch -n 1 nvidia-smi
```

Watch disk:

```bash
watch -n 5 df -h
```

---
