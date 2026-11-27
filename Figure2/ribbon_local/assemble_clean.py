"""Assemble the newly rendered panels into the reference Figure 2 layout."""

from pathlib import Path
import xml.etree.ElementTree as ET
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure2_compositions/clean_export"
writer = PdfWriter()
page = writer.add_blank_page(width=864, height=945.5)
page.merge_transformed_page(
    PdfReader(OUT / "panels_DE_clean.pdf").pages[0], Transformation().translate(ty=-18)
)
page.merge_transformed_page(
    PdfReader(OUT / "panels_AB_redesigned.pdf").pages[0],
    Transformation().translate(ty=599),
)
page.merge_transformed_page(
    PdfReader(ROOT / "ribbon_local/panel_C_dense.pdf").pages[0],
    Transformation().translate(ty=278.2),
)
writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)
with (OUT / "figure2_clean.pdf").open("wb") as f:
    writer.write(f)
ns = "http://www.w3.org/2000/svg"
ET.register_namespace("", ns)
root = ET.Element(
    "{" + ns + "}svg",
    {
        "width": "864pt",
        "height": "945.5pt",
        "viewBox": "0 0 864 945.5",
        "version": "1.1",
    },
)
ET.SubElement(
    root, "{" + ns + "}rect", {"width": "864", "height": "945.5", "fill": "white"}
)


def append(path, prefix, y, parent):
    r = ET.parse(path).getroot()
    ids = {e.get("id"): prefix + e.get("id") for e in r.iter() if e.get("id")}
    for e in r.iter():
        for key, val in list(e.attrib.items()):
            if key == "id":
                e.set(key, ids[val])
            elif val.startswith("url(#") and val[5:-1] in ids:
                e.set(key, "url(#" + ids[val[5:-1]] + ")")
            elif val.startswith("#") and val[1:] in ids:
                e.set(key, "#" + ids[val[1:]])
    # Nested pt-sized SVG viewports rescale panels by 96/72 in SVG readers.
    # All panel viewBoxes already use PDF points; preserve those user units.
    group = ET.SubElement(
        parent,
        "{" + ns + "}g",
        {"id": prefix + "panel", "transform": f"translate(0 {y})"},
    )
    for child in list(r):
        group.append(child)


append(OUT / "panels_DE_clean.svg", "cleanDE_", -8, root)
append(OUT / "panels_AB_redesigned.svg", "cleanAB_", 0, root)
append(ROOT / "ribbon_local/panel_C_dense.svg", "cleanC_", 364.5, root)
ET.ElementTree(root).write(
    OUT / "figure2_clean.svg", encoding="utf-8", xml_declaration=True
)
doc = pdfium.PdfDocument(str(OUT / "figure2_clean.pdf"))
doc[0].render(scale=2).to_pil().save(OUT / "figure2_clean.png")
print(OUT / "figure2_clean.pdf")
