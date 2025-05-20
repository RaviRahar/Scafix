let liveMode = true;
let curVisPage = 0;
let curProcPage = 1;
let totalPages = 0;
let inPdf = null;
// let diffPdf = null;
let outPdfBlob = null;
let outPdf = null;

// Each element will contain 3 images: in, diff, out
let allImgArr = [];
let isMobile = false;
const maxWidthMobile = "768px";

const RoughCanvas = document.createElement("canvas");

// Event binding
document.addEventListener("DOMContentLoaded", () => {
  document
    .getElementById("pdfInput")
    .addEventListener("change", handlePDFUpload);

  // Other initialization code
  document.getElementById("downloadButton").disabled = true;
  isMobile = window.matchMedia("(max-width: " + maxWidthMobile + ")").matches;
});

// Handle uploaded PDF file
function handlePDFUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  showUIAfterLoad(file.name);

  processPdf(file);
}

// Show UI elements after file is loaded
function showUIAfterLoad(filename) {
  document.body.classList.add("loaded");

  document.getElementById("fileName").textContent = filename;

  document.getElementById("tagLine").classList.remove("visible");
  document.getElementById("pdfInputContainer").classList.remove("visible");

  document.getElementById("fileName").classList.add("visible");
  document.getElementById("contentWrapper").classList.add("visible");
}

// Reset UI and canvas
function resetApp() {
  document.body.classList.remove("loaded");

  document.getElementById("fileName").textContent = "";
  document.getElementById("pdfInput").value = "";

  document.getElementById("tagLine").classList.add("visible");
  document.getElementById("pdfInputContainer").classList.add("visible");

  document.getElementById("fileName").classList.remove("visible");
  document.getElementById("contentWrapper").classList.remove("visible");

  liveMode = false;
  curVisPage = 0;
  curProcPage = 1;
  totalPages = 0;
  inPdf = null;
  // let diffPdf = null;
  outPdfBlob = null;
  outPdf = null;

  // Each element will contain 3 images: in, diff, out
  allImgArr = [];

  ["originalCanvas", "diffCanvas", "finalCanvas"].forEach((id) => {
    const canvas = document.getElementById(id);
    if (canvas !== null) {
      const context = canvas.getContext("2d");
      context.clearRect(0, 0, canvas.width, canvas.height);
    }
  });

  const RoughCanvasCxt = RoughCanvas.getContext("2d");
  RoughCanvasCxt.clearRect(0, 0, RoughCanvas.width, RoughCanvas.height);
}

// Process PDF
async function processPdf(file) {
  try {
    inPdf = await loadPdf(file);

    const isScanned = await isScannedPDF(inPdf);
    if (!isScanned) {
      alert("This PDF is not a scanned PDF (contains text).");
      resetApp();
      return;
    }

    await processAllPages(inPdf);
  } catch (err) {
    console.error("Error loading PDF:", err);
    alert("Failed to load PDF. Please try a valid file.");
  }
}

// Load the PDF file using pdf.js
async function loadPdf(file) {
  const loadingTask = pdfjsLib.getDocument(URL.createObjectURL(file));
  const pdfDoc = await loadingTask.promise;
  return pdfDoc;
}

// Check if PDF is scanned (image-only)
async function isScannedPDF(pdfDoc) {
  const numPages = pdfDoc.numPages;

  // Check the first few pages (e.g. up to 3) to determine
  const pagesToCheck = Math.min(3, numPages);

  for (let i = 1; i <= pagesToCheck; i++) {
    const page = await pdfDoc.getPage(i);
    const textContent = await page.getTextContent();

    if (textContent.items && textContent.items.length > 10) {
      // If there's significant text, it's likely not scanned
      return false;
    }
  }

  return true;
}

// Iterate through all pages to process
async function processAllPages(pdfDoc) {
  totalPages = pdfDoc.numPages;
  outPdf = await createPDFDocument();
  // diffPdf = await createPDFDocument();

  for (let i = 1; i <= totalPages; i++) {
    const { inImageData, diffImageData, outImageData, width, height } =
      await processPage(pdfDoc, i);
    allImgArr.push([inImageData, diffImageData, outImageData]);
    curProcPage++;
    if (!liveMode) {
      document.getElementById("showNextButton").disabled = false;
    }
    await addImagePageToPDF(outPdf, outImageData, width, height);
    // await addImagePageToPDF(diffPdf, diffImageData, width, height);
  }
  outPdfBlob = await finalizePDF(outPdf);

  document.getElementById("downloadButton").disabled = false;
  // alert("Processing completed.");
}

// Process a single page: extract, clean, add
async function processPage(pdfDoc, pageNumber) {
  const page = await pdfDoc.getPage(pageNumber);

  // Prepare canvas
  // Adjust scale for quality
  const viewport = page.getViewport({ scale: 3.0 });
  let canvas;
  let context;

  if (liveMode || curProcPage === 1) {
    canvas = document.getElementById("originalCanvas");
  } else {
    canvas = RoughCanvas;
  }

  canvas.width = viewport.width;
  canvas.height = viewport.height;
  context = canvas.getContext("2d", { alpha: true });

  // Render the page onto canvas
  await page.render({ canvasContext: context, viewport }).promise;

  // Extract ImageData from canvas
  const inImageData = context.getImageData(0, 0, canvas.width, canvas.height);

  // Remove black bars
  const { cleanImageData: outImageData, diffImageData } =
    await removeBlackBarsOpenCV(inImageData);

  if (liveMode || curProcPage === 1) {
    // Display cleaned image on diffCanvas
    const finalCanvas = document.getElementById("finalCanvas");
    const finalCanvasCtx = finalCanvas.getContext("2d");

    finalCanvas.width = canvas.width;
    finalCanvas.height = canvas.height;
    finalCanvasCtx.putImageData(outImageData, 0, 0);

    const diffCanvas = document.getElementById("diffCanvas");
    const diffCanvasCtx = diffCanvas.getContext("2d");

    diffCanvas.width = canvas.width;
    diffCanvas.height = canvas.height;
    diffCanvasCtx.putImageData(diffImageData, 0, 0);

    curVisPage++;

    if (curVisPage === totalPages) {
      document.getElementById("showNextButton").disabled = true;
      document.getElementById("showPrevButton").disabled = false;
    }
  }

  // Store or return cleaned data for further use
  return {
    inImageData: inImageData,
    diffImageData: diffImageData,
    outImageData: outImageData,
    width: canvas.width,
    height: canvas.height,
  };
}

// Add an image-based page to a PDF
async function addImagePageToPDF(pdfDoc, imageData, width, height) {
  // Convert ImageData to PNG blob
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { alpha: true });
  context.putImageData(imageData, 0, 0);

  const pngBlob = await new Promise(
    (resolve) => canvas.toBlob(resolve, "image/png"),
    // 0.95 = near-lossless
    // canvas.toBlob(resolve, "image/jpeg", 0.95)
  );
  const pngBytes = new Uint8Array(await pngBlob.arrayBuffer());

  // Embed image in PDF
  const embeddedImage = await pdfDoc.embedPng(pngBytes);
  const page = pdfDoc.addPage([width, height]);
  page.drawImage(embeddedImage, {
    x: 0,
    y: 0,
    width: width,
    height: height,
  });
}

// Extract page as image from PDF
async function extractPageImage(pdfDoc, pageNumber) {
  const page = await pdfDoc.getPage(pageNumber);

  const viewport = page.getViewport({ Scale: 2.0 });

  // Create a canvas for rendering
  const canvas = document.createElement("canvas");
  canvas.width = viewport.width;
  canvas.height = viewport.height;
  const context = canvas.getContext("2d", { alpha: true });

  const renderContext = {
    canvasContext: context,
    viewport: viewport,
  };

  // Render the page
  await page.render(renderContext).promise;
  const imgData = context.getImageData(0, 0, canvas.width, canvas.height);

  return {
    canvas: canvas,
    imageData: imgData,
    width: canvas.width,
    height: canvas.height,
  };
}

// Remove black bars using OpenCV
async function removeBlackBarsOpenCV(imageData) {
  cv = cv instanceof Promise ? await cv : cv;

  return new Promise((resolve) => {
    if (cv.Mat === undefined) {
      cv["onRuntimeInitialized"] = () => {
        resolve(processImage());
      };
    } else {
      resolve(processImage());
    }

    function processImage() {
      let src = cv.matFromImageData(imageData);

      const gray = new cv.Mat();
      const binary = new cv.Mat();
      const morph = new cv.Mat();
      const cleaned = src.clone();

      const height = src.rows;
      const width = src.cols;

      // --- Constants ---
      const THRESHOLD_VALUE = 20;
      const MORPH_KERNEL_SIZE = new cv.Size(15, 15);
      const DILATION_ITERATIONS = 1;
      const MIN_BORDER_AREA = 10000;
      const EDGE_PROXIMITY_RATIO = 0.1;
      const CENTER_PROXIMITY_RATIO = 0.4;
      const EXTREMELY_SENSITIVE_ZONE = 0.01;

      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      // --- Step 1: Mask dark areas ---
      cv.threshold(gray, binary, THRESHOLD_VALUE, 255, cv.THRESH_BINARY_INV);
      const kernel = cv.getStructuringElement(cv.MORPH_RECT, MORPH_KERNEL_SIZE);
      cv.morphologyEx(binary, morph, cv.MORPH_CLOSE, kernel);
      for (let i = 0; i < DILATION_ITERATIONS; i++) {
        cv.dilate(morph, morph, kernel);
      }

      // --- Step 2: Text mask ---
      const textBinary = new cv.Mat();
      cv.adaptiveThreshold(
        gray,
        textBinary,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY_INV,
        15,
        10,
      );
      const textKernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        new cv.Size(30, 3),
      );
      const textMask = new cv.Mat();
      cv.dilate(textBinary, textMask, textKernel, new cv.Point(-1, -1), 2);
      cv.morphologyEx(
        textMask,
        textMask,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5)),
      );

      // --- Step 3: Contour Detection ---
      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(
        morph,
        contours,
        hierarchy,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
      );

      const cx = Math.floor(width / 2);
      const cy = Math.floor(height / 2);
      const centerW = Math.floor(CENTER_PROXIMITY_RATIO * width);
      const centerH = Math.floor(CENTER_PROXIMITY_RATIO * height);
      const x1 = cx - centerW;
      const y1 = cy - centerH;
      const x2 = cx + centerW;
      const y2 = cy + centerH;

      const sx1 = Math.floor(width * EXTREMELY_SENSITIVE_ZONE);
      const sy1 = Math.floor(height * EXTREMELY_SENSITIVE_ZONE);
      const sx2 = width - sx1;
      const sy2 = height - sy1;

      let targeted_mask = new cv.Mat.zeros(src.rows, src.cols, src.type());

      for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        const rect = cv.boundingRect(cnt);
        const area = rect.width * rect.height;

        const spansFullWidth = rect.width >= width * 0.98;
        const spansFullHeight = rect.height >= height * 0.98;

        const touchesSensitiveZone =
          rect.x <= sx1 ||
          rect.x + rect.width >= sx2 ||
          rect.y <= sy1 ||
          rect.y + rect.height >= sy2;

        const isNearEdge =
          rect.x < EDGE_PROXIMITY_RATIO * width ||
          rect.x + rect.width > (1 - EDGE_PROXIMITY_RATIO) * width ||
          rect.y < EDGE_PROXIMITY_RATIO * height ||
          rect.y + rect.height > (1 - EDGE_PROXIMITY_RATIO) * height ||
          spansFullWidth ||
          spansFullHeight;

        // const isNearCenter =
        //   rect.x <= x2 &&
        //   rect.x + rect.width >= x1 &&
        //   rect.y <= y2 &&
        //   rect.y + rect.height >= y1;

        const isNearCenter =
          rect.x + rect.width > x1 &&
          rect.x < x2 &&
          rect.y + rect.height > y1 &&
          rect.y < y2;

        const barMask = new cv.Mat.zeros(height, width, cv.CV_8UC1);
        cv.drawContours(barMask, contours, i, new cv.Scalar(255), -1);
        const overlap = new cv.Mat();
        cv.bitwise_and(barMask, textMask, overlap);
        const overlapsTextZone = cv.countNonZero(overlap) > 0;
        barMask.delete();
        overlap.delete();

        const forceRemove = touchesSensitiveZone;
        const regularRemove =
          area > MIN_BORDER_AREA &&
          isNearEdge &&
          !isNearCenter &&
          !overlapsTextZone;

        if (forceRemove || regularRemove) {
          const color = forceRemove
            ? new cv.Scalar(255, 0, 0, 255)
            : new cv.Scalar(0, 0, 255, 255);

          cv.drawContours(targeted_mask, contours, i, color, -1);

          cv.drawContours(
            cleaned,
            contours,
            i,
            new cv.Scalar(255, 255, 255, 255),
            -1,
          );
        }
      }

      const targetedMaskData = new ImageData(
        new Uint8ClampedArray(targeted_mask.data),
        cleaned.cols,
        cleaned.rows,
      );

      const cleanedImageData = new ImageData(
        new Uint8ClampedArray(cleaned.data),
        cleaned.cols,
        cleaned.rows,
      );

      // Cleanup
      src.delete();
      gray.delete();
      binary.delete();
      morph.delete();
      cleaned.delete();
      contours.delete();
      hierarchy.delete();
      textBinary.delete();
      textKernel.delete();
      textMask.delete();
      targeted_mask.delete();

      return {
        cleanImageData: cleanedImageData,
        diffImageData: targetedMaskData,
      };
    }
  });
}

// Create a new PDF document using pdf-lib
async function createPDFDocument() {
  const pdfDoc = await PDFLib.PDFDocument.create();
  return pdfDoc;
}

// Finalize the PDF and return Blob/URL
async function finalizePDF(pdfDoc) {
  const pdfBytes = await pdfDoc.save();
  return new Blob([pdfBytes], { type: "application/pdf" });
}

// Make the final PDF downloadable via blob URL
function makePDFDownloadable(blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "cleaned_output.pdf";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// TODO: show progress messages
function showStatusMessage(message) { }

// Render page on specific canvas
function renderPageOnCanvas(canvas, canvasID, pageNo) {
  const context = canvas.getContext("2d");
  const img = allImgArr[pageNo - 1][canvasID];

  canvas.width = img.width;
  canvas.height = img.height;
  context.putImageData(img, 0, 0);
}

// Process the current page only
function renderAllCanvases(pageNo = curVisPage) {
  if (!allImgArr) return;

  const pageImgs = allImgArr[pageNo - 1];
  if (!pageImgs || pageImgs.length === 0) {
    console.error(`No image data for page ${pageNo}`);
    return;
  }

  const canvasOrig = document.getElementById("originalCanvas");
  if (canvasOrig !== null) {
    renderPageOnCanvas(canvasOrig, 0, pageNo);
  }

  const canvasDiff = document.getElementById("diffCanvas");
  if (canvasDiff !== null) {
    renderPageOnCanvas(canvasDiff, 1, pageNo);
  }

  const canvasFinal = document.getElementById("finalCanvas");
  if (canvasFinal !== null) {
    renderPageOnCanvas(canvasFinal, 2, pageNo);
  }
}

// Navigate to next page
function showNextPage() {
  if (curVisPage <= totalPages) {
    if (curVisPage < curProcPage - 1) {
      document.getElementById("showPrevButton").disabled = true;
      document.getElementById("showNextButton").disabled = true;
      renderAllCanvases(curVisPage + 1);
      curVisPage++;
      document.getElementById("showPrevButton").disabled = false;
      if (curVisPage === totalPages) {
        document.getElementById("showNextButton").disabled = true;
      } else {
        document.getElementById("showNextButton").disabled = false;
      }
    } else if (curVisPage === totalPages && totalPages === curProcPage - 1) {
      console.log("Reached the end.");
    } else {
      alert("Page not yet processed.");
    }
  }
}

// Navigate to previous page
function showPreviousPage() {
  if (curVisPage > 1) {
    document.getElementById("showPrevButton").disabled = true;
    document.getElementById("showNextButton").disabled = true;
    renderAllCanvases(curVisPage - 1);
    curVisPage--;
    document.getElementById("showNextButton").disabled = false;
    if (curVisPage <= 1) {
      document.getElementById("showPrevButton").disabled = true;
    } else {
      document.getElementById("showPrevButton").disabled = false;
    }
  } else {
    console.log("Reached the start.");
  }
}

// Download the processed PDF
function downloadOutput() {
  if (!outPdfBlob) {
    alert("No output PDF available yet.");
    return;
  }
  makePDFDownloadable(outPdfBlob);
}
