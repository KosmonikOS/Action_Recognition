from __future__ import annotations

import base64
import io
import json
import os
from typing import Iterator

import requests
import streamlit as st
import streamlit.components.v1 as components


DEFAULT_API_URL = os.getenv("ACT_REC_API_URL", "http://localhost:8000")
PREDICTION_ENDPOINT = "/stream/predict"
SUPPORTED_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]


def _build_endpoint_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base:
        return f"http://localhost:8000{PREDICTION_ENDPOINT}"
    return f"{base}{PREDICTION_ENDPOINT}"


def render_video_player(
    video_bytes: bytes, mime_type: str, autoplay: bool, width: int = 720, height: int = 405
) -> None:
    """Render the uploaded video inside a fixed-size player."""
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    autoplay_attr = "autoplay muted" if autoplay else ""
    html = f"""
    <div style="width:{width}px;height:{height}px;display:flex;align-items:center;justify-content:center;background:#000;border-radius:8px;overflow:hidden;">
      <video id="action-rec-video" width="{width}" height="{height}" controls playsinline style="background:#000;" {autoplay_attr}>
        <source src="data:{mime_type};base64,{video_b64}" type="{mime_type}">
        Your browser does not support the video tag.
      </video>
    </div>
    """
    components.html(html, height=height + 30, scrolling=False)


def stream_predictions(
    endpoint_url: str,
    video_bytes: bytes,
    filename: str,
    content_type: str | None,
) -> Iterator[dict[str, object]]:
    request_files = {
        "file": (
            filename or "uploaded_video.mp4",
            io.BytesIO(video_bytes),
            content_type or "video/mp4",
        )
    }

    with requests.post(endpoint_url, files=request_files, stream=True, timeout=None) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            try:
                yield json.loads(raw_line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError("Received malformed JSON from inference service.") from exc


def main() -> None:
    st.set_page_config(page_title="Action Recognition Streaming", layout="wide")
    st.title("Action Recognition Streaming Demo")
    st.caption("Upload a video, send it to the SODE streaming API, and watch predictions arrive in real time.")

    predictions = st.session_state.setdefault("predictions", [])
    st.session_state.setdefault("streaming_requested", False)
    st.session_state.setdefault("streaming_active", False)

    api_base = st.text_input(
        "Inference API base URL",
        value=DEFAULT_API_URL,
        help="Point this to the host running act_rec.api.app (default: http://localhost:8000).",
    )
    endpoint_url = _build_endpoint_url(api_base)

    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        help="Supported formats: mp4, mov, avi, mkv, webm",
    )

    if uploaded_file is None:
        st.info("Choose a video to begin.")
        return

    video_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    if not video_bytes:
        st.warning("The selected file appears to be empty.")
        return

    video_signature = (uploaded_file.name, len(video_bytes))
    if st.session_state.get("current_video_signature") != video_signature:
        st.session_state["current_video_signature"] = video_signature
        st.session_state["predictions"] = []
        st.session_state["streaming_requested"] = False
        st.session_state["streaming_active"] = False

    predictions = st.session_state["predictions"]

    mime_type = uploaded_file.type or "video/mp4"

    col_video, col_table = st.columns([3, 2], gap="large")
    with col_video:
        video_container = st.container()
        status_placeholder = st.empty()
        start_clicked = st.button(
            "Start streaming predictions",
            type="primary",
            disabled=st.session_state.get("streaming_active", False),
            use_container_width=True,
        )

    with col_table:
        table_placeholder = st.empty()

    if start_clicked and not st.session_state.get("streaming_active", False):
        st.session_state["streaming_requested"] = True

    autoplay_flag = bool(st.session_state.get("streaming_requested") or st.session_state.get("streaming_active"))

    with video_container:
        render_video_player(video_bytes, mime_type, autoplay=autoplay_flag)

    should_stream = bool(st.session_state.get("streaming_requested"))

    if predictions:
        last = predictions[-1]
        label = last.get("prediction", "unknown")
        confidence = last.get("confidence", 0.0)
        window_index = last.get("window_index", 0)
        status_placeholder.markdown(f"**Label:** {label} _(confidence {confidence:.2f})_")
        table_placeholder.dataframe(predictions)
    else:
        status_placeholder.markdown("_No predictions yet._")
        table_placeholder.info("Prediction windows will appear here once streaming starts.")

    if not should_stream:
        return

    st.session_state["predictions"] = []
    st.session_state["streaming_active"] = True
    st.session_state["streaming_requested"] = False
    status_placeholder.markdown("_Waiting for predictions..._")
    table_placeholder.empty()

    try:
        with st.spinner("Contacting inference service..."):
            for record in stream_predictions(
                endpoint_url=endpoint_url,
                video_bytes=video_bytes,
                filename=uploaded_file.name,
                content_type=mime_type,
            ):
                st.session_state["predictions"].append(record)
                label = record.get("prediction", "unknown")
                confidence = record.get("confidence", 0.0)
                window_index = record.get("window_index", 0)
                status_placeholder.markdown(f"**Window {window_index}:** {label} _(confidence {confidence:.2f})_")
                table_placeholder.dataframe(st.session_state["predictions"])
    except requests.HTTPError as exc:
        st.error(f"Inference service returned an error: {exc.response.status_code} {exc.response.reason}")
    except requests.RequestException as exc:
        st.error(f"Failed to reach inference service: {exc}")
    except RuntimeError as exc:
        st.error(str(exc))
    finally:
        st.session_state["streaming_active"] = False

    if not st.session_state["predictions"]:
        table_placeholder.empty()
        st.warning("No predictions were received from the inference service.")


if __name__ == "__main__":
    main()
