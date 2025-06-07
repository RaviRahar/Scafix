import os
import cv2
import numpy as np

# Uncomment if using as standalone file
# import fitz  # PyMuPDF
# from PIL import Image

# === Config ===
PDF_PATH = "./MHD_02_Book_1.pdf"
OUTPUT_DIR = "./output"
THRESHOLD_VALUE = 20
MORPH_KERNEL_SIZE = (15, 15)
DILATION_ITERATIONS = 1
MIN_BORDER_AREA = 10000
# EXCLUSION_ZONE = 0.96
# SUPER_SENSITIVE_ZONE = 0.01
EXCLUSION_ZONE = 0.96
SUPER_SENSITIVE_ZONE = 0.01
DPI = 300  # keep full clarity
PNG_COMPRESSION = 0  # no PNG compression for intermediary files

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_scanned(image):
    """Simple heuristic to check if an image is likely from a scanned PDF."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_ratio = np.mean(gray < 40)
    return black_ratio > 0.01


def rect_overlap(r1, r2):
    """Check if two rectangles overlap"""
    x1, y1, x2, y2 = r1
    a1, b1, a2, b2 = r2
    return not (x2 <= a1 or x1 >= a2 or y2 <= b1 or y1 >= b2)


def is_inside(rect_inner, rect_outer):
    """Check if one rectangle is fully inside another"""
    x1, y1, x2, y2 = rect_inner
    a1, b1, a2, b2 = rect_outer
    return x1 >= a1 and y1 >= b1 and x2 <= a2 and y2 <= b2


def remove_black_bars(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # === Step 1: Binary mask for dark regions ===
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph = cv2.dilate(morph, kernel, iterations=DILATION_ITERATIONS)

    overlay = img.copy()
    cleaned = img.copy()

    # === Step 2: Define Zones ===
    # Exclusion (center)
    cx, cy = width // 2, height // 2
    center_w = int(EXCLUSION_ZONE / 2 * width)
    center_h = int(EXCLUSION_ZONE / 2 * height)
    exclusion_rect = (cx - center_w, cy - center_h, cx + center_w, cy + center_h)

    # Super Sensitive (edges)
    ss_margin_x = int(width * SUPER_SENSITIVE_ZONE)
    ss_margin_y = int(height * SUPER_SENSITIVE_ZONE)
    super_sensitive_rects = [
        (0, 0, width, ss_margin_y),  # Top
        (0, height - ss_margin_y, width, height),  # Bottom
        (0, 0, ss_margin_x, height),  # Left
        (width - ss_margin_x, 0, width, height),  # Right
    ]

    # Sensitive (between exclusion and super sensitive)
    sensitive_rects = [
        (0, ss_margin_y, width, exclusion_rect[1]),  # Top
        (0, exclusion_rect[3], width, height - ss_margin_y),  # Bottom
        (ss_margin_x, 0, exclusion_rect[0], height),  # Left
        (exclusion_rect[2], 0, width - ss_margin_x, height),  # Right
    ]

    # === Step 3: Contour detection and zone-based removal ===
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        contour_rect = (x, y, x + w, y + h)

        # Skip if entirely inside exclusion zone
        if is_inside(contour_rect, exclusion_rect):
            continue

        removed = False

        # Remove anything overlapping with super sensitive zone
        for ss_rect in super_sensitive_rects:
            if rect_overlap(contour_rect, ss_rect):
                cv2.drawContours(cleaned, [cnt], -1, (255, 255, 255), -1)
                cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 3)
                removed = True
                break
        if removed:
            continue

        # Remove if in sensitive zone, area > MIN_BORDER_AREA, and not in exclusion
        for sens_rect in sensitive_rects:
            if rect_overlap(contour_rect, sens_rect) and area > MIN_BORDER_AREA:
                cv2.drawContours(cleaned, [cnt], -1, (255, 255, 255), -1)
                cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 3)
                break

    return cleaned, overlay


def extract_number(filename):
    # match = re.search(r'(\d+)', filename)
    # return int(match.group(1)) if match else -1

    # return int(filename.split('-')[1].split('.')[0])
    return int(filename.removeprefix("page-").removesuffix(".png"))


class ImageNavigator:
    def __init__(self, files):
        self.files = files
        self.index = 0

    def current(self):
        return self.files[self.index]

    def next(self):
        if self.index < len(self.files) - 1:
            self.index += 1

    def prev(self):
        if self.index > 0:
            self.index -= 1

    def has_next(self):
        return self.index < len(self.files) - 1

    def has_prev(self):
        return self.index > 0


def wait_key_loop():
    """Wait for user key input: 'j' = next (cleaned), 'i'=original, 'o'=cleaned, 'q'=quit."""
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in [ord("q"), ord("j"), ord("k"), ord("i"), ord("o")]:
            return key


def process_images(input_dir, output_dir):
    files = [file for file in os.listdir(input_dir) if file.endswith(".png")]
    files.sort(key=extract_number)
    # files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    navigator = ImageNavigator(files)

    while True:
        file = navigator.current()
        img_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        img = cv2.imread(img_path)

        if not is_scanned(img):
            print(f"Image {img_path}: Skipped (not scanned).")
            if navigator.has_next():
                navigator.next()
                continue
            else:
                break

        cleaned, overlay = remove_black_bars(img.copy())
        # prepare grid: original | overlay | cleaned
        orig = img.copy()
        h, w = orig.shape[:2]
        overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_CUBIC)
        cleaned_resized = cv2.resize(cleaned, (w, h), interpolation=cv2.INTER_CUBIC)
        # grid = np.hstack([orig, overlay_resized, cleaned_resized])

        # cv2.imshow("Preview", grid)
        cv2.imshow("Preview", overlay_resized)
        key = wait_key_loop()
        if key == ord("q"):
            break
        elif key == ord("j"):
            navigator.prev()
        elif key == ord("k"):
            navigator.next()
        elif key == ord("i"):
            out_path = os.path.join(output_dir, file)
            cv2.imwrite(out_path, orig, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
            print(f"Saved original: {out_path}")
            navigator.next()
        elif key == ord("o"):
            out_path = os.path.join(output_dir, file)
            cv2.imwrite(
                out_path,
                cleaned_resized,
                [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
            )
            print(f"Saved cleaned: {out_path}")
            navigator.next()
        else:
            print(
                "Unrecognized key. Use 'j'/'k' to navigate, 'i'/'o' to save, 'q' to quit."
            )

    cv2.destroyAllWindows()

    # for file in files:
    #     if file.endswith(".png"):
    #         img_path = os.path.join(input_dir, file)
    #         output_path = os.path.join(output_dir, file)

    #         img = cv2.imread(img_path)

    #         if not is_scanned(img):
    #             print(f"Image {img_path}: Skipped (not scanned).")
    #             pass

    #         cleaned, overlay = remove_black_bars(img.copy())
    #         # prepare grid: original | overlay | cleaned
    #         orig = img.copy()
    #         h, w = orig.shape[:2]
    #         overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_CUBIC)
    #         cleaned_resized = cv2.resize(cleaned, (w, h), interpolation=cv2.INTER_CUBIC)
    #         # grid = np.hstack([orig, overlay_resized, cleaned_resized])

    #         # cv2.imshow("Preview", grid)
    #         cv2.imshow("Preview", overlay_resized)
    #         key = wait_key_loop()

    #         if key == ord("q"):
    #             break

    #         # Save choice: 'i'=orig, 'o'=cleaned, 'j'=cleaned
    #         chosen = orig if (key == ord("i")) else cleaned_resized
    #         out_path = os.path.join(output_dir, os.path.basename(img_path))
    #         cv2.imwrite(
    #             out_path, chosen, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
    #         )

    # cv2.destroyAllWindows()


def process_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    selected_images = []

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=DPI)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        img = arr.reshape((pix.height, pix.width, pix.n))
        img = (
            cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            if pix.n == 4
            else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )

        if not is_scanned(img):
            print(f"Page {page_num+1}: Skipped (not scanned).")
            continue

        cleaned, overlay = remove_black_bars(img.copy())

        # prepare grid: original | overlay | cleaned
        orig = img.copy()
        h, w = orig.shape[:2]
        overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_CUBIC)
        cleaned_resized = cv2.resize(cleaned, (w, h), interpolation=cv2.INTER_CUBIC)
        grid = np.hstack([orig, overlay_resized, cleaned_resized])

        cv2.imshow("Preview", grid)
        key = wait_key_loop()
        if key == ord("q"):
            break

        # Save choice: 'i'=orig, 'o'=cleaned, 'j'=cleaned
        chosen = orig if (key == ord("i")) else cleaned_resized
        out_path = os.path.join(output_dir, f"page_{page_num+1:03}.png")
        cv2.imwrite(out_path, chosen, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
        selected_images.append(out_path)

    cv2.destroyAllWindows()
    return selected_images


def images_to_pdf(image_paths, output_path):
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    if not pil_images:
        print("No pages were selected.")
        return

    # Increase JPEG quality inside PDF
    save_kwargs = {
        "format": "PDF",
        "save_all": True,
        "append_images": pil_images[1:],
        "dpi": (DPI, DPI),
        "quality": 100,
        "optimize": True,
        "subsampling": 0,
    }
    pil_images[0].save(output_path, **save_kwargs)
    print(f"Saved PDF to: {output_path}")


def main():
    cleaned_images = process_pdf(PDF_PATH, OUTPUT_DIR)
    final_pdf = os.path.join(OUTPUT_DIR, "selected_output.pdf")
    images_to_pdf(cleaned_images, final_pdf)


if __name__ == "__main__":
    main()
