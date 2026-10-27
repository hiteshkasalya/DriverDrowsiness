import os
import re

html_path = "IEEE_Report_Driver_Drowsiness.html"
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

img_srcs = re.findall(r'src=["\']([^"\']+)["\']', html)
print("--- Image Verification ---")
for src in img_srcs:
    print(f"Image: {src} -> Exists: {os.path.exists(src)} ({os.path.getsize(src)/1024:.1f} KB)")

print("\n--- Authors Verification ---")
for author in ["Hitesh Kasalya", "Aaryan Andhare", "Siddhi Zore", "Prachi Dadlani", "Anita Thengade"]:
    print(f"Author/Guide: {author} -> Found: {author in html}")

pdf_path = "IEEE_Report_Driver_Drowsiness.pdf"
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

page_count = len(re.findall(rb'/Type\s*/Page\b', pdf_bytes))
print(f"\n--- PDF Verification ---")
print(f"PDF File: {pdf_path}")
print(f"File Size: {len(pdf_bytes)/1024:.1f} KB")
print(f"Page Count: {page_count} pages (Target: 9-10 pages)")
