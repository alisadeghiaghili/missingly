import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    import fcntl
except ImportError:  # pragma: no cover - only used by local Windows tooling
    fcntl = None


def _load_private_environment():
    path = os.path.join(os.path.dirname(__file__), ".smtp.env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_private_environment()


ROOT = os.path.dirname(__file__)
APP_PORT = int(os.environ.get("PASSENGER_UVICORN_PORT", "18103"))
APP_URL = f"http://127.0.0.1:{APP_PORT}"
RUNTIME_DIR = os.path.join(ROOT, "tmp")
PID_PATH = os.path.join(RUNTIME_DIR, "uvicorn.pid")
VERSION_PATH = os.path.join(RUNTIME_DIR, "uvicorn.version")
LOCK_PATH = os.path.join(RUNTIME_DIR, "uvicorn.lock")
PROCESS = None


def _request(path="/health", timeout=2):
    return urllib.request.urlopen(f"{APP_URL}{path}", timeout=timeout)


def _runtime_version():
    tracked_paths = [
        os.path.join(ROOT, "main.py"),
        os.path.join(ROOT, ".env"),
        __file__,
    ]
    parts = []
    for path in tracked_paths:
        try:
            stat = os.stat(path)
            parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            parts.append(f"{path}:missing")
    return "|".join(parts)


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _pid_is_running(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _stop_recorded_uvicorn():
    raw_pid = _read_text(PID_PATH)
    if not raw_pid.isdigit():
        return
    pid = int(raw_pid)
    if not _pid_is_running(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    for _ in range(30):
        if not _pid_is_running(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _healthy():
    try:
        with _request():
            return True
    except Exception:
        return False


def _ensure_uvicorn():
    global PROCESS

    if PROCESS and PROCESS.poll() is None:
        return

    os.makedirs(RUNTIME_DIR, exist_ok=True)
    with open(LOCK_PATH, "a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            current_version = _runtime_version()
            if _read_text(VERSION_PATH) == current_version and _healthy():
                return

            _stop_recorded_uvicorn()
            env = os.environ.copy()
            env.setdefault("APP_HOST", "127.0.0.1")
            env.setdefault("APP_PORT", str(APP_PORT))
            PROCESS = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(APP_PORT),
                    "--proxy-headers",
                ],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            with open(PID_PATH, "w", encoding="utf-8") as pid_file:
                pid_file.write(str(PROCESS.pid))
            with open(VERSION_PATH, "w", encoding="utf-8") as version_file:
                version_file.write(current_version)
            for _ in range(40):
                if _healthy():
                    return
                time.sleep(0.25)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def application(environ, start_response):
    _ensure_uvicorn()

    path = environ.get("PATH_INFO", "/")
    query = environ.get("QUERY_STRING", "")
    url = f"{APP_URL}{path}"
    if query:
        url = f"{url}?{query}"

    method = environ.get("REQUEST_METHOD", "GET")
    body_length = int(environ.get("CONTENT_LENGTH") or 0)
    body = environ["wsgi.input"].read(body_length) if body_length else None
    headers = {}
    for key, value in environ.items():
        if key.startswith("HTTP_"):
            header_name = key[5:].replace("_", "-").title()
            headers[header_name] = value
    if environ.get("CONTENT_TYPE"):
        headers["Content-Type"] = environ["CONTENT_TYPE"]

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(request, timeout=300)
        status = f"{response.status} {response.reason}"
        response_headers = [
            (key, value)
            for key, value in response.headers.items()
            if key.lower() not in {"connection", "content-length", "transfer-encoding"}
        ]
        start_response(status, response_headers)
        return _iter_response(response)
    except urllib.error.HTTPError as exc:
        status = f"{exc.code} {exc.reason}"
        response_headers = [
            (key, value)
            for key, value in exc.headers.items()
            if key.lower() not in {"connection", "content-length", "transfer-encoding"}
        ]
        start_response(status, response_headers)
        return _iter_response(exc)


def _iter_response(response, chunk_size=8192):
    try:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        response.close()
