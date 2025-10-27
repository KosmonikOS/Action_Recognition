from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .inference import (
    MULTI_PERSON_TOLERANCE,
    WINDOW_SIZE,
    get_labeler,
    get_sode_bundle,
    prediction_ndjson_stream,
)


app = FastAPI(
    title="SODE Streaming Inference",
    version="0.1.0",
    description=(
        "Stream action recognition predictions from a YOLO pose backbone "
        f"and SODE classifier (window={WINDOW_SIZE}, tolerance={MULTI_PERSON_TOLERANCE})."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/stream/predict")
def stream_predictions(file: UploadFile = File(...)) -> StreamingResponse:
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="Unsupported media type. Expected a video file.")

    suffix = Path(file.filename).suffix if file.filename else ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

    try:
        # Force lazy singletons to initialize so we surface errors before streaming starts.
        get_sode_bundle()
        get_labeler()
    except Exception as exc:  # noqa: BLE001
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialise inference pipeline: {exc}") from exc

    def _stream() -> bytes:
        try:
            yield from prediction_ndjson_stream(tmp_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SODE streaming inference API.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload. Use only in development environments.",
    )
    args = parser.parse_args()

    target = "act_rec.api.app:app"
    uvicorn.run(target, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
