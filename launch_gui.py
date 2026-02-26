# coding=utf-8
"""
Qwen3-TTS-JP Launcher
- Auto port selection (7860~)
- Auto browser launch on server ready
- Model selection via command line argument
"""
import socket
import subprocess
import sys
import time
import webbrowser
import threading
import urllib.request
import urllib.error
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def find_free_port(start_port=7860, max_attempts=100):
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"All ports {start_port}-{start_port + max_attempts} are in use")


def wait_for_server_and_open_browser(url, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=2)
            print(f"\n[Launcher] Server ready -> opening browser: {url}")
            webbrowser.open(url)
            return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            time.sleep(2)
    print("[Launcher] Timeout: server did not start within the expected time.")
    return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_voice_local = os.path.join(script_dir, "models", "Qwen3-TTS-12Hz-1.7B-CustomVoice")

    if len(sys.argv) > 1 and sys.argv[1] == "--custom-voice":
        model = custom_voice_local if os.path.isdir(custom_voice_local) else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    else:
        model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    port = find_free_port(7860)
    url = f"http://127.0.0.1:{port}"

    print("=" * 60)
    print("  Qwen3-TTS-JP Launcher")
    print("=" * 60)
    print(f"  Model : {model}")
    print(f"  URL   : {url}")
    print("=" * 60)
    print()

    threading.Thread(
        target=wait_for_server_and_open_browser,
        args=(url, 300),
        daemon=True,
    ).start()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "qwen_tts.cli.demo",
            model,
            "--ip", "127.0.0.1",
            "--port", str(port),
            "--no-flash-attn",
        ]
    )


if __name__ == "__main__":
    main()
