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
EDGE_PROXIMITY_RATIO = 0.2
CENTER_PROXIMITY_RATIO = 0.4
EXTREMELY_SENSITIVE_ZONE = 0.02
DPI = 300  # keep full clarity
PNG_COMPRESSION = 0  # no PNG compression for intermediary files

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_scanned(image):
    """Simple heuristic to check if an image is likely from a scanned PDF."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_ratio = np.mean(gray < 40)
    return black_ratio > 0.01


def remove_black_bars(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # === Step 1: Binary mask for dark regions ===
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph = cv2.dilate(morph, kernel, iterations=DILATION_ITERATIONS)

    # === Step 2: Infer text zones ===
    text_binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    text_mask = cv2.dilate(text_binary, text_kernel, iterations=2)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, (5, 5))

    overlay = img.copy()
    cleaned = img.copy()

    # === Step 3: Contour detection & removal ===
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = width // 2, height // 2
    center_w = int(CENTER_PROXIMITY_RATIO * width)
    center_h = int(CENTER_PROXIMITY_RATIO * height)
    x1, y1 = cx - center_w, cy - center_h
    x2, y2 = cx + center_w, cy + center_h
    sx1, sy1 = int(width * EXTREMELY_SENSITIVE_ZONE), int(
        height * EXTREMELY_SENSITIVE_ZONE
    )
    sx2, sy2 = width - sx1, height - sy1

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        spans_full_w = w >= width * 0.98
        spans_full_h = h >= height * 0.98
        touches_zone = x <= sx1 or x + w >= sx2 or y <= sy1 or y + h >= sy2
        near_edge = (
            x < EDGE_PROXIMITY_RATIO * width
            or x + w > (1 - EDGE_PROXIMITY_RATIO) * width
            or y < EDGE_PROXIMITY_RATIO * height
            or y + h > (1 - EDGE_PROXIMITY_RATIO) * height
            or spans_full_w
            or spans_full_h
        )
        is_center = x < x2 and x + w > x1 and y < y2 and y + h > y1

        bar_mask = np.zeros_like(text_mask)
        cv2.drawContours(bar_mask, [cnt], -1, 255, -1)
        overlaps = cv2.bitwise_and(bar_mask, text_mask)
        overlaps_text = np.count_nonzero(overlaps) > 0

        force = touches_zone
        regular = (
            area > MIN_BORDER_AREA and near_edge and not is_center and not overlaps_text
        )

        if force or regular:
            color = (0, 255, 0) if not force else (0, 0, 255)
            bg_color = (255, 255, 255)
            cv2.drawContours(cleaned, [cnt], -1, bg_color, -1)
            cv2.drawContours(overlay, [cnt], -1, color, 5)

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
