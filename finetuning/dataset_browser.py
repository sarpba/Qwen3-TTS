# coding=utf-8
import argparse
import json
import mimetypes
import os
import tempfile
import threading
from collections import Counter, defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import soundfile as sf


mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/mpeg", ".mp3")


class DatasetIndex:
    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path.resolve()
        self.base_dir = self.jsonl_path.parent
        self.records: List[Dict] = []
        self.audio_id_to_path: Dict[str, Path] = {}
        self.path_to_audio_id: Dict[Path, str] = {}
        self.references: Dict[str, Dict] = {}
        self.language_counts: Counter = Counter()
        self.reference_ids_by_language: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
        self._load()

    def _resolve_audio_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = (self.base_dir / path).resolve()
        else:
            path = path.resolve()
        return path

    def _get_audio_id(self, path: Path) -> str:
        audio_id = self.path_to_audio_id.get(path)
        if audio_id is not None:
            return audio_id
        audio_id = f"a{len(self.path_to_audio_id):06d}"
        self.path_to_audio_id[path] = audio_id
        self.audio_id_to_path[audio_id] = path
        return audio_id

    def _load(self) -> None:
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                try:
                    audio_path = self._resolve_audio_path(payload["audio"])
                    ref_audio_path = self._resolve_audio_path(payload["ref_audio"])
                    text = str(payload["text"]).strip()
                    ref_text = str(payload["ref_text"]).strip()
                    language = str(payload["language"]).strip()
                except KeyError as exc:
                    raise ValueError(
                        f"Missing field {exc.args[0]!r} in {self.jsonl_path}:{line_number}"
                    ) from exc

                audio_id = self._get_audio_id(audio_path)
                ref_audio_id = self._get_audio_id(ref_audio_path)
                reference = self.references.get(ref_audio_id)
                if reference is None:
                    reference = {
                        "id": ref_audio_id,
                        "audio_id": ref_audio_id,
                        "audio_name": ref_audio_path.name,
                        "audio_path": str(ref_audio_path),
                        "text": ref_text,
                        "language": language,
                        "count": 0,
                    }
                    self.references[ref_audio_id] = reference
                    self.reference_ids_by_language[language].append(ref_audio_id)

                reference["count"] += 1
                self.language_counts[language] += 1
                self.records.append(
                    {
                        "id": len(self.records),
                        "language": language,
                        "text": text,
                        "audio_id": audio_id,
                        "audio_name": audio_path.name,
                        "audio_path": str(audio_path),
                        "ref_audio_id": ref_audio_id,
                        "ref_audio_name": ref_audio_path.name,
                        "ref_audio_path": str(ref_audio_path),
                        "ref_text": ref_text,
                    }
                )

        for language in self.reference_ids_by_language:
            self.reference_ids_by_language[language].sort(
                key=lambda ref_id: (
                    self.references[ref_id]["audio_name"].lower(),
                    self.references[ref_id]["id"],
                )
            )

    def get_summary(self) -> Dict:
        languages = [
            {"language": language, "count": count}
            for language, count in sorted(self.language_counts.items())
        ]
        references = []
        for ref_id, ref in sorted(
            self.references.items(),
            key=lambda item: (
                item[1]["language"].lower(),
                item[1]["audio_name"].lower(),
                item[0],
            ),
        ):
            references.append(
                {
                    "id": ref_id,
                    "language": ref["language"],
                    "audio_id": ref["audio_id"],
                    "audio_name": ref["audio_name"],
                    "audio_path": ref["audio_path"],
                    "text": ref["text"],
                    "count": ref["count"],
                }
            )
        return {
            "record_count": len(self.records),
            "reference_count": len(self.references),
            "languages": languages,
            "references": references,
        }

    def query_records(
        self,
        language: Optional[str],
        reference_id: Optional[str],
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> Dict:
        results = self.records
        if language:
            results = [record for record in results if record["language"] == language]
        if reference_id:
            results = [record for record in results if record["ref_audio_id"] == reference_id]
        total = len(results)
        if offset < 0:
            offset = 0
        if limit is None or limit < 0:
            page = results[offset:]
        else:
            page = results[offset : offset + limit]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "records": page,
        }

    def get_audio_path(self, audio_id: str) -> Optional[Path]:
        return self.audio_id_to_path.get(audio_id)

    def get_waveform(self, audio_id: str, points: int = 1200) -> Dict:
        audio_path = self.get_audio_path(audio_id)
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError(audio_id)
        data, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
        mono = data.mean(axis=1)
        total_samples = mono.shape[0]
        if total_samples == 0:
            return {"audio_id": audio_id, "sample_rate": sample_rate, "duration": 0.0, "peaks": []}
        bucket_size = max(1, total_samples // points)
        peaks = []
        for start in range(0, total_samples, bucket_size):
            chunk = mono[start : start + bucket_size]
            peaks.append(float(abs(chunk).max()) if chunk.size else 0.0)
        return {
            "audio_id": audio_id,
            "sample_rate": sample_rate,
            "duration": total_samples / float(sample_rate),
            "peaks": peaks,
        }

    def update_record_text(self, record_id: int, text: str, ref_text: Optional[str] = None) -> Dict:
        with self._lock:
            record = self.records[record_id]
            record["text"] = text.strip()
            if ref_text is not None:
                ref_text = ref_text.strip()
                ref_audio_id = record["ref_audio_id"]
                self.references[ref_audio_id]["text"] = ref_text
                for item in self.records:
                    if item["ref_audio_id"] == ref_audio_id:
                        item["ref_text"] = ref_text
            self._write_jsonl()
            return record

    def trim_audio(self, audio_id: str, start_sec: float, end_sec: float) -> Dict:
        with self._lock:
            audio_path = self.get_audio_path(audio_id)
            if audio_path is None or not audio_path.exists():
                raise FileNotFoundError(audio_id)

            info = sf.info(str(audio_path))
            total_frames = info.frames
            sample_rate = info.samplerate
            duration = total_frames / float(sample_rate)

            start_sec = max(0.0, min(start_sec, duration))
            end_sec = max(start_sec, min(end_sec, duration))
            if end_sec <= start_sec:
                raise ValueError("End must be greater than start.")

            start_frame = int(start_sec * sample_rate)
            end_frame = int(end_sec * sample_rate)
            if end_frame <= start_frame:
                raise ValueError("Selection is too short.")

            data, read_sample_rate = sf.read(str(audio_path), start=start_frame, stop=end_frame, always_2d=True)
            suffix = audio_path.suffix or ".wav"
            fd, temp_name = tempfile.mkstemp(prefix="trim_", suffix=suffix, dir=str(audio_path.parent))
            os.close(fd)
            temp_path = Path(temp_name)
            try:
                sf.write(str(temp_path), data, read_sample_rate)
                temp_path.replace(audio_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            return {
                "audio_id": audio_id,
                "path": str(audio_path),
                "duration": data.shape[0] / float(read_sample_rate),
            }

    def _write_jsonl(self) -> None:
        lines = []
        for record in self.records:
            lines.append(
                json.dumps(
                    {
                        "audio": record["audio_path"],
                        "text": record["text"],
                        "ref_audio": record["ref_audio_path"],
                        "ref_text": record["ref_text"],
                        "language": record["language"],
                    },
                    ensure_ascii=False,
                )
            )
        tmp_path = self.jsonl_path.with_suffix(self.jsonl_path.suffix + ".tmp")
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp_path.replace(self.jsonl_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Serve a local browser for prepared dataset JSONL files.")
    parser.add_argument("--jsonl", required=True, help="Prepared dataset JSONL path.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    return parser.parse_args()


def build_handler(index: DatasetIndex):
    static_html = (Path(__file__).resolve().parent / "web" / "dataset_browser.html").read_text(
        encoding="utf-8"
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(static_html)
                return
            if parsed.path == "/api/summary":
                self._send_json(index.get_summary())
                return
            if parsed.path == "/api/records":
                params = parse_qs(parsed.query)
                language = params.get("language", [""])[0] or None
                reference_id = params.get("reference", [""])[0] or None
                offset = _parse_int(params.get("offset", ["0"])[0], default=0, minimum=0)
                limit = _parse_int(params.get("limit", ["100"])[0], default=100, minimum=1)
                self._send_json(index.query_records(language, reference_id, offset=offset, limit=limit))
                return
            if parsed.path == "/api/waveform":
                params = parse_qs(parsed.query)
                audio_id = params.get("audio_id", [""])[0]
                if not audio_id:
                    self.send_error(HTTPStatus.BAD_REQUEST, "audio_id is required")
                    return
                points = _parse_int(params.get("points", ["1200"])[0], default=1200, minimum=100)
                try:
                    self._send_json(index.get_waveform(audio_id, points=points))
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Audio not found")
                return
            if parsed.path.startswith("/audio/"):
                audio_id = parsed.path.split("/", 2)[-1]
                audio_path = index.get_audio_path(audio_id)
                if audio_path is None or not audio_path.exists() or not audio_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "Audio not found")
                    return
                self._send_file(audio_path)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self):
            parsed = urlparse(self.path)
            try:
                payload = self._read_json_body()
            except ValueError as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return

            if parsed.path == "/api/update-text":
                try:
                    record_id = int(payload["record_id"])
                    text = str(payload["text"])
                    ref_text = payload.get("ref_text")
                    if ref_text is not None:
                        ref_text = str(ref_text)
                    record = index.update_record_text(record_id, text, ref_text=ref_text)
                except (KeyError, ValueError) as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self._send_json({"ok": True, "record": record})
                return

            if parsed.path == "/api/trim-audio":
                try:
                    audio_id = str(payload["audio_id"])
                    start_sec = float(payload["start_sec"])
                    end_sec = float(payload["end_sec"])
                    result = index.trim_audio(audio_id, start_sec, end_sec)
                except (KeyError, ValueError) as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Audio not found")
                    return
                self._send_json({"ok": True, "result": result})
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format, *args):
            return

        def _send_html(self, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._send_no_cache_headers()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: Dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_no_cache_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self._send_no_cache_headers()
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 128)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        def _read_json_body(self) -> Dict:
            content_length = _parse_int(self.headers.get("Content-Length", "0"), default=0, minimum=0)
            raw = self.rfile.read(content_length)
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON body") from exc

        def _send_no_cache_headers(self) -> None:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")

    return Handler


def _parse_int(value: str, default: int, minimum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def main():
    args = parse_args()
    index = DatasetIndex(Path(args.jsonl))
    server = ThreadingHTTPServer((args.host, args.port), build_handler(index))
    print(f"Serving dataset browser on http://{args.host}:{args.port}")
    print(f"Loaded {len(index.records)} rows and {len(index.references)} reference groups from {index.jsonl_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
