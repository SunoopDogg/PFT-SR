import cv2


class ROISelector:
    def __init__(self):
        self.roi_start = None
        self.roi_end = None
        self.roi_selecting = False
        self.roi_selected = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
            self.roi_selecting = True
            self.roi_selected = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.roi_selecting:
                self.roi_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_end = (x, y)
            self.roi_selecting = False
            self.roi_selected = True


def select_roi(image_path):
    selector = ROISelector()

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    original_h, original_w = img.shape[:2]

    # Resize for display if image is too large
    max_display_size = 1200
    scale_factor = 1.0
    if max(original_h, original_w) > max_display_size:
        scale_factor = max_display_size / max(original_h, original_w)
        display_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    else:
        display_img = img.copy()

    window_name = "Select ROI (drag mouse, press ENTER to confirm, ESC to cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, selector.mouse_callback)

    print("=" * 60)
    print("ROI Selection")
    print("=" * 60)
    print(f"Image size: {original_w} x {original_h}")
    print(f"Display scale: {scale_factor:.2f}")
    print("-" * 60)
    print("Instructions:")
    print("  - Drag mouse to select ROI")
    print("  - Press ENTER to confirm selection")
    print("  - Press ESC to cancel")
    print("=" * 60)

    while True:
        temp_img = display_img.copy()

        # Draw rectangle while selecting
        if selector.roi_start is not None and selector.roi_end is not None:
            cv2.rectangle(temp_img, selector.roi_start, selector.roi_end, (0, 255, 0), 2)

            # Show ROI dimensions
            x1, y1 = selector.roi_start
            x2, y2 = selector.roi_end
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            # Calculate actual dimensions
            actual_w = int(w / scale_factor)
            actual_h = int(h / scale_factor)

            info_text = f"ROI: {actual_w}x{actual_h}"
            cv2.putText(temp_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        cv2.imshow(window_name, temp_img)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None

        elif key == 13 or key == 10:  # ENTER
            if selector.roi_selected and selector.roi_start is not None and selector.roi_end is not None:
                # Convert display coordinates to original image coordinates
                x1 = int(min(selector.roi_start[0], selector.roi_end[0]) / scale_factor)
                y1 = int(min(selector.roi_start[1], selector.roi_end[1]) / scale_factor)
                x2 = int(max(selector.roi_start[0], selector.roi_end[0]) / scale_factor)
                y2 = int(max(selector.roi_start[1], selector.roi_end[1]) / scale_factor)

                # Clamp to image bounds
                x1 = max(0, min(x1, original_w))
                y1 = max(0, min(y1, original_h))
                x2 = max(0, min(x2, original_w))
                y2 = max(0, min(y2, original_h))

                cv2.destroyAllWindows()
                return (x1, y1, x2, y2)
