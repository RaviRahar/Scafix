# Scafix

A tool to fix your pdf scans. Currently only fixes large black scan lines on the margins of pages.
All processing is done in your browser.

[Link to online instance](https://ravirahar.github.io/scafix/)

Also check out python script. That version can use jbig2 images. The main difference in pipelines can be summarized as:

Online Version: Renders page -> converts page to png -> processing -> renders png -> converts png to page
Python Script: extracts jbig2 images (or any other type) -> converts to png -> processing -> converts to jbig2 -> converts to pdf

The benefit is that python script version creates pdfs that are smaller in size. Use DPI of original images. Hence, final pdf is as close to original as possible.

Online Version Uses:

- PDF.js
- OpenCV.js (self compiled version for smaller size)
- PDF-LIB

Python Version Uses:

- pdfimages (poppler)
- jbig2enc
- jbig2topdf (modified version) (included in jbig2enc repo)
- opencv-python
