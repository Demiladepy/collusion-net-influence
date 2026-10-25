"""Render REPORT.md as a PDF in the layout of the official Apart sprint template (via headless Chrome).

Template specs (from the template's own HTML export): US Letter, 1in margins, Old Standard TT throughout,
11pt body at 1.15 line height, 20pt bold title between two rules, author grid, "With Apart Research",
centred "Abstract" heading over a narrowed italic abstract, sprint footnote on page 1.

    python build_pdf.py               -> report/Apart_submission.pdf   (for submission)
    python build_pdf.py --portfolio   -> report/Apart_portfolio.pdf    (same paper with a cover page)
"""
import argparse
import html
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE / "REPORT.md"
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Old+Standard+TT:ital,wght@0,400;0,700;1,400&display=swap');
@page { size: Letter; margin: 0.9in 1in 0.9in 1in;
        @bottom-center { content: counter(page); font: 9pt "Old Standard TT", serif; color: #555; } }
@page cover { margin: 0; @bottom-center { content: none; } }
@page front { @bottom-center { content: none; } }
* { box-sizing: border-box; }
html { font: 10.5pt/1.32 "Old Standard TT", Georgia, serif; color: #000; }
body { margin: 0; }
p { margin: 0 0 6.5pt; text-align: justify; hyphens: auto; }
a { color: #1155cc; text-decoration: none; }
b, strong { font-weight: 700; }

/* Front page, as in the template */
.front { page: front; break-after: page; height: 9.2in; display: flex; flex-direction: column; }
.front .rule { border-top: 1.5pt solid #000; }
.front h1.title { font-size: 19pt; font-weight: 700; text-align: center; text-transform: uppercase;
                  letter-spacing: .02em; line-height: 1.2; margin: 18pt 0.2in 16pt; }
.front h1.title sup { font-size: 10pt; }
.authors { display: flex; justify-content: center; gap: 0 54pt; flex-wrap: wrap; margin: 24pt 0 14pt; text-align: center; }
.author { min-width: 2in; }
.author .name { font-size: 11.5pt; }
.author .aff { font-size: 10.5pt; }
.author .link { font-size: 9.5pt; }
.with { text-align: center; margin-bottom: 22pt; font-size: 10.5pt; }
.with b { display: block; }
.abstract-h { text-align: center; font-size: 14pt; font-weight: 700; margin: 4pt 0 10pt; }
.abstract { font-style: italic; text-align: justify; margin: 0 0.55in; font-size: 10.5pt; line-height: 1.38; }
.footnote { margin-top: auto; font-size: 8.5pt; padding-top: 6pt; border-top: 0.6pt solid #000; width: 2.2in; white-space: nowrap; }

h2 { font-size: 14pt; font-weight: 700; margin: 15pt 0 5pt; break-after: avoid; }
h3 { font-size: 11.5pt; font-weight: 700; margin: 10pt 0 4pt; break-after: avoid; }
h2 + h3 { margin-top: 4pt; }
ul, ol { margin: 0 0 7pt; padding-left: 20pt; }
li { margin-bottom: 3pt; text-align: justify; }

figure { margin: 8pt 0 10pt; break-inside: avoid; text-align: center; }
figure img { display: block; margin: 0 auto 4pt; }
figcaption, .tcap { font-size: 9pt; line-height: 1.3; text-align: justify; color: #222; }
.tblock { break-inside: avoid; margin: 8pt 0 10pt; }
.tcap { margin-bottom: 4pt; }
table { width: 100%; border-collapse: collapse; font-size: 8.8pt; line-height: 1.25; font-variant-numeric: tabular-nums; }
thead th { border-top: 1.2pt solid #000; border-bottom: 0.7pt solid #000; font-weight: 700; padding: 4pt 5pt; text-align: center; vertical-align: bottom; }
thead th:first-child { text-align: left; }
tbody td { padding: 3pt 5pt; text-align: center; border-bottom: 0.3pt solid #d6d6d6; }
tbody td:first-child { text-align: left; }
tbody tr:last-child td { border-bottom: 1.2pt solid #000; }
td.hi { background: #e8f3ec; font-weight: 700; }
td .ci { color: #666; font-size: 7.6pt; margin-left: 2pt; white-space: nowrap; }

.refs li { text-align: left; font-size: 9.5pt; line-height: 1.3; margin-bottom: 4pt; overflow-wrap: anywhere; }
.appendix { break-before: page; }
.small p, .small li { font-size: 9.5pt; }

/* Portfolio cover (not part of the submission) */
.cover { page: cover; break-after: page; height: 11in; padding: 1.3in 1.1in 1in; display: flex; flex-direction: column;
         background: #f4f1ea; position: relative; }
.cover .kicker { font-size: 10pt; letter-spacing: .22em; text-transform: uppercase; color: #6b5d45; }
.cover h1 { font-size: 34pt; line-height: 1.1; margin: 20pt 0 18pt; font-weight: 700; }
.cover .sub { font-size: 14pt; font-style: italic; color: #3a352c; max-width: 5.4in; line-height: 1.35; }
.cover .grid { margin-top: 36pt; display: grid; grid-template-columns: repeat(4, 16pt); gap: 6pt; }
.cover .grid span { width: 16pt; height: 16pt; border: 1.2pt solid #2b2620; border-radius: 50%; }
.cover .grid span.on { background: #2b2620; }
.cover .meta { margin-top: auto; border-top: 1.2pt solid #2b2620; padding-top: 14pt; display: flex; justify-content: space-between; font-size: 11pt; }
.cover .meta b { display: block; font-size: 13pt; }
"""


def inline(text):
    t = html.escape(text, quote=False)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*]+)\*(?!\w)", r"<i>\1</i>", t)
    t = re.sub(r"(https?://[^\s<)]+)", r'<a href="\1">\1</a>', t)
    return t


def cell(val, header=False):
    """Table cell: '19/20 [.76, .99]' -> count plus a small grey interval; '**x**' -> highlighted."""
    hi = val.startswith("**") and val.endswith("**")
    v = val.strip("*").strip()
    m = re.match(r"^(.*?)\s*(\[[^\]]+\])$", v)
    body = inline(m.group(1)) + f'<span class="ci">{html.escape(m.group(2))}</span>' if m else inline(v)
    if header:
        return f"<th>{body}</th>"
    return f'<td class="hi">{body}</td>' if hi else f"<td>{body}</td>"


def parse_front(md):
    front = {}
    lines = md.splitlines()
    i = 0
    while i < len(lines) and re.match(r"^[a-z_]+:", lines[i]):
        k, v = lines[i].split(":", 1)
        front[k.strip()] = v.strip()
        i += 1
    text = re.sub(r"<!--.*?-->", "", "\n".join(lines[i:]), flags=re.S)
    m = re.search(r"^## Abstract\s*\n(.*?)(?=^## )", text, flags=re.S | re.M)
    front["abstract"] = " ".join(m.group(1).split())
    return front, text[m.end():]


def body_html(text):
    out, in_appendix, pending_cap = [], False, None
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l.rstrip() for l in block.strip().splitlines()]
        if not lines:
            continue
        if pending_cap and lines[0].startswith("|"):
            lines = [pending_cap] + lines  # caption written as its own paragraph above the table
        pending_cap = None
        if lines[0].startswith("*Table") and len(lines) == 1:
            pending_cap = lines[0]
            continue
        if lines[0].startswith("## "):
            title = lines[0][3:].strip()
            cls = ""
            if title.startswith("Appendix") and not in_appendix:
                in_appendix, cls = True, ' class="appendix"'
            out.append(f"<h2{cls}>{inline(title)}</h2>")
            lines = lines[1:]
        elif lines[0].startswith("### "):
            out.append(f"<h3>{inline(lines[0][4:].strip())}</h3>")
            lines = lines[1:]
        if not lines:
            continue
        if lines[0].startswith("*Table"):
            cap = lines[0].strip("*")
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines[1:] if l.startswith("|") and not re.match(r"^\|\s*-", l)]
            head = "".join(cell(c, True) for c in rows[0])
            bodyrows = "".join("<tr>" + "".join(cell(c) for c in r) + "</tr>" for r in rows[1:])
            out.append(f'<div class="tblock"><div class="tcap">{inline(cap)}</div><table><thead><tr>{head}</tr></thead><tbody>{bodyrows}</tbody></table></div>')
            continue
        m = re.match(r"^!\[(.*)\]\((.*)\)$", lines[0])
        if m:
            src = (HERE / m.group(2)).resolve().as_uri()
            width = "6.5in" if "fig1" in m.group(2) else "4.4in"
            out.append(f'<figure><img src="{src}" style="width:{width}"><figcaption>{inline(m.group(1))}</figcaption></figure>')
            continue
        if re.match(r"^(- |\d+\. )", lines[0]):
            tag = "ol" if re.match(r"^\d+\. ", lines[0]) else "ul"
            items, cur = [], None
            for l in lines:
                mm = re.match(r"^(?:- |\d+\. )(.*)$", l)
                if mm:
                    cur = [mm.group(1)]
                    items.append(cur)
                elif cur is not None:
                    cur.append(l.strip())
            cls = ' class="refs"' if any("arXiv" in " ".join(i) or "doi.org" in " ".join(i) for i in items) else ""
            out.append(f"<{tag}{cls}>" + "".join(f"<li>{inline(' '.join(i))}</li>" for i in items) + f"</{tag}>")
            continue
        out.append(f"<p>{inline(' '.join(l.strip() for l in lines))}</p>")
    return "\n".join(out)


def page_html(front, body, portfolio):
    authors = []
    for k in ("author", "author2", "author3"):
        if front.get(k):
            n = k[len("author"):]
            link = front.get("link" + n, "")
            authors.append(f'<div class="author"><div class="name">{inline(front[k])}</div>'
                           f'<div class="aff">{inline(front.get("affiliation" + n, ""))}</div>'
                           + (f'<div class="link">{inline(link)}</div>' if link else "") + "</div>")
    cover = ""
    if portfolio:
        dots = "".join(f'<span class="{"on" if i in (1, 6, 11, 12) else ""}"></span>' for i in range(16))
        cover = f'''<section class="cover"><div class="kicker">AI Collusion Research Sprint · October 2026</div>
<h1>{inline(front["title"])}</h1><div class="sub">{inline(front.get("tagline", ""))}</div>
<div class="grid">{dots}</div>
<div class="meta"><div><b>{inline(front["author"])}</b>{inline(front.get("affiliation", ""))}</div>
<div style="text-align:right"><b>Research sprint</b>Apart Research, 23–25 October 2026</div></div></section>'''
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(front['title'])}</title>
<style>{CSS}</style></head><body>{cover}
<section class="front"><div class="rule" style="margin-top:0.35in"></div>
<h1 class="title">{inline(front['title'])}<sup>1</sup></h1><div class="rule"></div>
<div class="authors">{''.join(authors)}</div>
<div class="with"><b>With</b>Apart Research</div>
<div class="abstract-h">Abstract</div><div class="abstract">{inline(front['abstract'])}</div>
<div class="footnote"><sup>1</sup> Research conducted at the <a href="https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25">AI Collusion Research Sprint</a>, October 2026</div>
</section>
{body}
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", action="store_true")
    args = ap.parse_args()
    front, text = parse_front(SOURCE.read_text(encoding="utf-8"))
    name = "Apart_portfolio" if args.portfolio else "Apart_submission"
    html_path, pdf_path = HERE / f"{name}.html", HERE / f"{name}.pdf"
    html_path.write_text(page_html(front, body_html(text), args.portfolio), encoding="utf-8")
    chrome = next(c for c in CHROME if Path(c).exists())
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()], check=True, capture_output=True)
    print(f"wrote {pdf_path.name}; abstract {len(front['abstract'].split())} words")


if __name__ == "__main__":
    main()
