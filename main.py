import argparse
import os
import glob
import subprocess
import json
import tempfile
from jbig2topdf import create_pdf
from remove_blacks import process_images as process_pngs

PDF_PY = "./jbig2topdf.py"  # Path to pdf.py inside jbig2enc
JBIG2_OUTPUT_PREFIX = "output"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run(
    cmd,
    cwd=None,
    stdout=None,
    stderr=None,
    check=True,
    capture_output=False,
    text=False,
):
    print(f"        Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        capture_output=capture_output,
        text=text,
    )


def pdf_to_jbig2s(pdf_path, jb2s_dir):
    run(["pdfimages", "-all", pdf_path, os.path.join(jb2s_dir, "page")])


def pdf_to_dpi_map(pdf_path, dpi_map_path):
    result = run(["pdfimages", "-list", pdf_path], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    dpi_map = {}
    for line in lines:
        if line.strip().startswith("page"):
            continue  # Skip header
        parts = line.split()
        if len(parts) >= 14:
            try:
                page_num = int(parts[0])
                x_ppi = int(parts[12])
                y_ppi = int(parts[13])
                dpi_map[page_num - 1] = {"x": x_ppi, "y": y_ppi}
            except ValueError:
                continue

    with open(dpi_map_path, "w") as f:
        json.dump(dpi_map, f)


def jb2_to_png_single_image(jb2_path, png_path):
    run(["jbig2dec", "-t", "png", "-o", png_path, jb2_path])


def jb2s_to_pngs(jb2s_dir, pngs_dir):
    for file in os.listdir(jb2s_dir):
        if file.endswith(".jb2"):
            jb2_path = os.path.join(jb2s_dir, file)
            png_path = os.path.join(pngs_dir, file.replace(".jb2", ".png"))
            jb2_to_png_single_image(jb2_path, png_path)


def pdf_to_pngs(pdf_path, pngs_dir):
    run(["pdfimages", "-png", pdf_path, os.path.join(pngs_dir, "page")])


def pngs_to_jbig2s(pngs_dir, jb2s_dir):
    pngs = sorted(f for f in os.listdir(pngs_dir) if f.endswith(".png"))

    before_files = set(os.listdir(pngs_dir))

    run(
        ["jbig2", "-s", "-p", "-O", JBIG2_OUTPUT_PREFIX] + pngs,
        cwd=pngs_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    after_files = set(os.listdir(pngs_dir))

    new_files = after_files - before_files

    for file in new_files:
        # if file.startswith("output."):
        src = os.path.join(pngs_dir, file)
        dst = os.path.join(jb2s_dir, file)
        os.replace(src, dst)


# def create_pdf_from_jbig2(input_dir, output_pdf_path):
#     sym_path = os.path.join(input_dir, "output.sym")
#     page_files = sorted(
#         [
#             os.path.join(input_dir, f)
#             for f in os.listdir(input_dir)
#             if f.startswith("output.") and f[7:].isdigit()
#         ]
#     )

#     if not page_files or not os.path.exists(sym_path):
#         raise FileNotFoundError("Missing JBIG2 symbol table or page files")

#     # Run the PDF generator with proper arguments
#     with open(output_pdf_path, "wb") as out:
#         run(
#             ["python3", PDF_PY, os.path.join(input_dir, JBIG2_OUTPUT_PREFIX)],
#             stdout=out,
#         )


def jbig2s_to_pdf(jb2s_dir, dpi_map_path, pdf_path):
    sym_path = os.path.join(jb2s_dir, "output.sym")
    page_files = sorted(
        [
            os.path.join(jb2s_dir, f)
            for f in os.listdir(jb2s_dir)
            if f.startswith("output.") and f[7:].isdigit()
        ]
    )

    if not page_files or not os.path.exists(sym_path):
        raise FileNotFoundError("Missing JBIG2 symbol table or page files")

    with open(pdf_path, "wb") as out:
        run(
            [
                "python3",
                PDF_PY,
                os.path.join(jb2s_dir, JBIG2_OUTPUT_PREFIX),
                "--dpi-map",
                dpi_map_path,
            ],
            stdout=out,
        )

    # file = os.path.join(jb2s_dir, JBIG2_OUTPUT_PREFIX)
    # sym = f"{file}.sym"
    # pages = glob.glob(f"{file}.[0-9]*")
    # create_pdf(sym, pages, dpi_map_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert and clean PDF with JBIG2 and OpenCV."
    )
    parser.add_argument(
        "-i",
        "--input_pdf",
        default="input.pdf",
        help="Path to input PDF file (default: input.pdf)",
    )
    parser.add_argument(
        "-w",
        "--work_dir",
        default="./output",
        help="Working directory (default: ./output)",
    )
    parser.add_argument(
        "-o",
        "--output_pdf",
        default=os.path.join("./output", "final_output.pdf"),
        help="Output PDF file path (default: ./output/final_output.pdf)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    INPUT_PDF = args.input_pdf
    WORK_DIR = args.work_dir
    OUTPUT_PDF = args.output_pdf

    # Validate input PDF file
    if not os.path.isfile(INPUT_PDF):
        print(f"Error: input PDF file does not exist: {INPUT_PDF}")
        exit(1)

    INPUT_PDF_PATH = os.path.abspath(INPUT_PDF)

    WORK_DIR_JBIG2 = os.path.join(WORK_DIR, "jbig2")
    WORK_DIR_PNG = os.path.join(WORK_DIR, "png")
    WORK_DIR_PNG_CLEANED = os.path.join(WORK_DIR, "png_cleaned")
    WORK_DIR_JBIG2_CLEANED = os.path.join(WORK_DIR, "jbig2_cleaned")
    WORK_DIR_DPI_MAP = os.path.join(WORK_DIR_JBIG2_CLEANED, "dpi_map.json")

    # Ensure all necessary directories exist
    ensure_dir(WORK_DIR)
    ensure_dir(WORK_DIR_JBIG2)
    ensure_dir(WORK_DIR_PNG)
    ensure_dir(WORK_DIR_PNG_CLEANED)
    ensure_dir(WORK_DIR_JBIG2_CLEANED)

    # print("Step 1: Extracting JBIG2 images...")
    # extract_jbig2_images(INPUT_PDF_PATH, WORK_DIR_JBIG2)

    # print("Step 2: Converting JBIG2 to PNG...")
    # convert_jb2_to_png(WORK_DIR_JBIG2, WORK_DIR_PNG)

    print("Step 1: Extracting PNG...")
    pdf_to_pngs(INPUT_PDF_PATH, WORK_DIR_PNG)

    print("Step 2: Extracting DPIs...")
    pdf_to_dpi_map(INPUT_PDF_PATH, WORK_DIR_DPI_MAP)

    print("Step 3: Processing images with OpenCV...")
    process_pngs(WORK_DIR_PNG, WORK_DIR_PNG_CLEANED)

    print("Step 4: Encoding back to JBIG2...")
    pngs_to_jbig2s(WORK_DIR_PNG_CLEANED, WORK_DIR_JBIG2_CLEANED)

    print("Step 5: Creating PDF from JBIG2 images...")
    jbig2s_to_pdf(WORK_DIR_JBIG2_CLEANED, WORK_DIR_DPI_MAP, OUTPUT_PDF)

    print(f"Done! Output PDF: {os.path.abspath(OUTPUT_PDF)}")


if __name__ == "__main__":
    main()
