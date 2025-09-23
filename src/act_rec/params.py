import pydantic


class YoloPoseVideoInferenceParams(pydantic.BaseModel):
    """Parameters for inference of a video with YOLOv11-pose model.

    Attributes:
        device: Device to run the inference on.
        rect: Whether to use rect mode.
        batch: Batch size.
        vid_stride: Stride for the video (How many frames to skip).
        verbose: Whether to print verbose output.
    """

    device: str = "mps"
    rect: bool = True
    batch: int = 10
    vid_stride: int = 2
    verbose: bool = False
    imgsz: tuple[int, int] = (320, 512)
