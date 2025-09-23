import numpy as np
import ultralytics

from act_rec.params import YoloPoseVideoInferenceParams


class YoloPoseVideoLabeler:
    """Wrapper to label videos with YOLO-pose model.

    This class is intended to be used during main dataset labeling process.

    Args
        model_path: Path to the YOLO-pose model.
        params: Parameters for the YOLO-pose model.
    """

    def __init__(self, model_path: str, params: YoloPoseVideoInferenceParams):
        self.model = ultralytics.YOLO(model_path)
        self.params = params

    def label_video(self, video_path: str) -> tuple[bool, np.ndarray | None]:
        """Label a video.

        Args
            video_path: Path to the video.

        Returns
            The tuple containing:
                - Whether the video contains only one person.
                - The keypoints of the person in the video.
        """
        frame_results = self.model(
            video_path,
            stream=True,
            device=self.params.device,
            imgsz=self.params.imgsz,
            rect=self.params.rect,
            batch=self.params.batch,
            vid_stride=self.params.vid_stride,
            verbose=self.params.verbose,
        )
        frame_keypoints = []
        for frame_result in frame_results:
            # Frame contains more then one person
            # For the sake of this project, we skip such videos
            if frame_result.keypoints.shape[0] != 1:
                return False, None

            frame_keypoints.append(frame_result.keypoints.data[0].cpu())
        return True, np.stack(frame_keypoints)
