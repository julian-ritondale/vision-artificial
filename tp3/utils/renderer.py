import math
import cv2
import numpy as np
import config


class Renderer:

    def draw_marker_contour(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        pts = corners[0].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
        for pt in corners[0].astype(np.int32):
            cv2.circle(frame, tuple(pt), config.CORNER_CIRCLE_RADIUS, color, -1)

    def draw_marker_label(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        marker_id: int,
        centroid_mm: np.ndarray | None = None,
        angle_deg: float | None = None,
    ) -> None:
        top_left = corners[0][0].astype(np.int32)
        lines: list[str] = [f"ID: {marker_id}"]
        if centroid_mm is not None and angle_deg is not None:
            lines.append(f"x={centroid_mm[0]:.1f}mm  y={centroid_mm[1]:.1f}mm")
            lines.append(f"\u03b8={angle_deg:.1f}\u00b0")

        line_h = 20
        text_x = int(top_left[0])
        text_y = int(top_left[1]) - len(lines) * line_h - 4

        for i, line in enumerate(lines):
            y = text_y + i * line_h
            (tw, th), _ = cv2.getTextSize(line, config.FONT, config.FONT_SCALE_SMALL, config.FONT_THICKNESS)
            cv2.rectangle(frame, (text_x - 2, y - th - 2), (text_x + tw + 2, y + 2), config.COLORS["text_bg"], -1)
            cv2.putText(frame, line, (text_x, y), config.FONT, config.FONT_SCALE_SMALL,
                        config.COLORS["text"], config.FONT_THICKNESS, cv2.LINE_AA)

    def draw_pose_arrow(
        self,
        image: np.ndarray,
        centroid_px: tuple[int, int],
        angle_deg: float,
        length_px: int = config.ARROW_LENGTH_PX,
        color: tuple[int, int, int] = config.COLORS["arrow"],
        thickness: int = 2,
    ) -> None:
        rad = math.radians(angle_deg)
        tip = (int(centroid_px[0] + length_px * math.cos(rad)),
               int(centroid_px[1] - length_px * math.sin(rad)))
        cv2.arrowedLine(image, centroid_px, tip, color, thickness, tipLength=0.3, line_type=cv2.LINE_AA)

    def draw_axes(
        self,
        image: np.ndarray,
        origin_px: tuple[int, int],
        scale: float,
        length_mm: float = config.AXIS_LENGTH_MM,
    ) -> None:
        length_px = int(length_mm * scale)
        x_tip = (origin_px[0] + length_px, origin_px[1])
        cv2.arrowedLine(image, origin_px, x_tip, config.COLORS["axis_x"], 2, tipLength=0.08, line_type=cv2.LINE_AA)
        cv2.putText(image, "X", (x_tip[0] + 4, x_tip[1] + 4), config.FONT,
                    config.FONT_SCALE_SMALL, config.COLORS["axis_x"], 1, cv2.LINE_AA)
        y_tip = (origin_px[0], origin_px[1] - length_px)
        cv2.arrowedLine(image, origin_px, y_tip, config.COLORS["axis_y"], 2, tipLength=0.08, line_type=cv2.LINE_AA)
        cv2.putText(image, "Y", (y_tip[0] + 4, y_tip[1] - 4), config.FONT,
                    config.FONT_SCALE_SMALL, config.COLORS["axis_y"], 1, cv2.LINE_AA)

    def overlay_pip(
        self,
        frame: np.ndarray,
        pip: np.ndarray,
        position: str = config.PIP_POSITION,
        border: bool = True,
    ) -> np.ndarray:
        out = frame.copy()
        fh, fw = out.shape[:2]
        ph, pw = pip.shape[:2]
        m = config.PIP_MARGIN
        if position == "top-right":
            y0, x0 = m, fw - pw - m
        elif position == "top-left":
            y0, x0 = m, m
        elif position == "bottom-right":
            y0, x0 = fh - ph - m, fw - pw - m
        else:
            y0, x0 = fh - ph - m, m
        y1, x1 = y0 + ph, x0 + pw
        if y0 < 0 or x0 < 0 or y1 > fh or x1 > fw:
            return out
        out[y0:y1, x0:x1] = pip
        if border:
            cv2.rectangle(out,
                          (x0 - config.PIP_BORDER_THICKNESS, y0 - config.PIP_BORDER_THICKNESS),
                          (x1 + config.PIP_BORDER_THICKNESS, y1 + config.PIP_BORDER_THICKNESS),
                          config.COLORS["pip_border"], config.PIP_BORDER_THICKNESS)
        return out

    def draw_registration_status(
        self,
        frame: np.ndarray,
        detected: int,
        required: int,
        is_registered: bool,
    ) -> None:
        fh, fw = frame.shape[:2]
        if is_registered:
            msg = "  [R] REGISTERED  |  [Z] Reset  |  [Q] Quit"
            color = config.COLORS["status_ok"]
        else:
            msg = (f"  Registration: {detected}/{required} ref markers detected"
                   f"  |  [R] Register  |  [Q] Quit")
            color = config.COLORS["status_ok"] if detected >= required else config.COLORS["status_warn"]
        bar_h = 32
        cv2.rectangle(frame, (0, fh - bar_h), (fw, fh), (20, 20, 20), -1)
        cv2.putText(frame, msg, (8, fh - 10), config.FONT, config.FONT_SCALE_SMALL,
                    color, 1, cv2.LINE_AA)

    def draw_mode_banner(self, frame: np.ndarray, mode: str) -> None:
        label = f"  MODE: {mode.upper()}  "
        color = config.COLORS["status_warn"] if mode == "registration" else config.COLORS["status_ok"]
        (tw, th), _ = cv2.getTextSize(label, config.FONT, config.FONT_SCALE_LARGE, config.FONT_THICKNESS)
        cv2.rectangle(frame, (0, 0), (tw + 8, th + 12), (20, 20, 20), -1)
        cv2.putText(frame, label, (4, th + 4), config.FONT, config.FONT_SCALE_LARGE,
                    color, config.FONT_THICKNESS, cv2.LINE_AA)
