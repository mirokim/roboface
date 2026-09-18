# PC LLM 서버 세팅 (LLM_BACKEND=remote)

Pi의 Qwen2.5-3B(느림) 대신 **PC에서 Ollama로 큰 모델**을 돌리고 로봇이 LAN으로 호출.
PC가 꺼져 있거나 안 닿으면 로봇은 자동으로 Pi 로컬 Qwen으로 fallback (60초 후 재시도).

검증: 2026-09-18 노트북(Core Ultra 7, GPU 없음) + qwen2.5:7b → Pi에서 호출 성공.

## 1. PC (Windows) — Ollama

```powershell
winget install --id Ollama.Ollama -e --silent
# LAN에서 접속 허용 (기본은 127.0.0.1만)
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:11434", "User")
# Ollama 재시작 (트레이 앱 종료 후 다시 실행, 또는)
Stop-Process -Name "ollama app","ollama" -Force -ErrorAction SilentlyContinue
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList serve -WindowStyle Hidden
# 모델
ollama pull qwen2.5:7b          # 4.7GB. GPU 있으면 qwen2.5:14b 도 OK
# vision까지 쓰려면: ollama pull qwen2.5vl:7b  → .env LLM_REMOTE_VISION=1
```

방화벽 (관리자 PowerShell, 노트북에선 없이도 됐지만 데스크탑은 보통 필요):

```powershell
New-NetFirewallRule -DisplayName "Ollama 11434" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow
```

확인: PC에서 `curl http://127.0.0.1:11434/api/version`, Pi에서 `curl http://<PC주소>:11434/api/version`.

## 2. 네트워크 — Pi가 PC에 닿아야 함

| 상황 | 방법 |
|---|---|
| 같은 Wi-Fi/LAN (클라이언트 격리 없음) | PC IP 또는 mDNS 이름(`<PC이름>.local`) 그대로 |
| PC는 사내 유선, Pi는 게스트 Wi-Fi (격리) | **USB 랜 어댑터로 Pi와 직결**. 어댑터 IPv4를 고정 `169.254.1.1/16`으로 (Pi eth0는 link-local 169.254.x.x라 같은 /16). Pi 쪽은 이미 `ipv4.link-local fallback` 설정돼 있어 DHCP 없이도 붙음 |
| 둘 다 인터넷만 있음 | Tailscale 양쪽 설치 → Tailscale IP 사용 |

USB 어댑터 고정 IP (관리자 PowerShell, 어댑터 이름은 `Get-NetAdapter`로 확인):

```powershell
New-NetIPAddress -InterfaceAlias "이더넷 2" -IPAddress 169.254.1.1 -PrefixLength 16
```

## 3. Pi — .env

```
LLM_BACKEND=remote
LLM_REMOTE_URL=http://169.254.1.1:11434/v1      # 직결이면 이 주소, 같은 LAN이면 PC IP/이름
LLM_REMOTE_MODEL=qwen2.5:7b
LLM_REMOTE_VISION=0                             # vision 모델이면 1
AGENT_DISABLED=0
```

```
sudo systemctl restart roboface   # 또는 원격 세션에서 systemctl restart roboface
```

LCD 좌하단 라벨이 `PC`면 원격, `Qwen`이면 로컬 fallback 중.
로그: `grep -E "원격 LLM|PC LLM" logs/roboface.log`.

## 4. 속도 감

| 머신 | 모델 | agent 1회 결정 |
|---|---|---|
| Pi 5 (llama-cpp) | qwen2.5:3b Q4 | ~20~40s |
| 노트북 Core Ultra 7 CPU | qwen2.5:7b | ~45s (첫 호출은 모델 로드 포함) |
| 데스크탑 + NVIDIA GPU | qwen2.5:7b~14b | 수 초 |

agent 프롬프트가 길어(상황 컨텍스트 + 도구 스키마) CPU에선 prefill이 지배적. GPU가 답.
