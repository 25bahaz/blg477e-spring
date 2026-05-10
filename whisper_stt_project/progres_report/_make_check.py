"""Build a fallback `article`-class version of progress_report.tex
so we can compile it in environments that don't have IEEEtran.cls.

Usage:  python3 report/_make_check.py
Result: report/check.tex
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src = (ROOT / "progress_report.tex").read_text()

src = src.replace(
    "\\documentclass[conference]{IEEEtran}",
    "\\documentclass[twocolumn,10pt,a4paper]{article}\n"
    "\\usepackage[a4paper,margin=2cm]{geometry}"
)

start = src.find("\\title{")
end = src.find("\\begin{document}")
if start != -1 and end != -1:
    new_block = (
        "\\title{Speech-to-Text Generation with OpenAI Whisper:\\\\\n"
        "A Multimedia Computing Project --- Progress Report}\n"
        "\\author{[Your Name], Student ID: [Your ID]\\\\\n"
        "Department of Computer Engineering, [Your University]\\\\\n"
        "Email: [your.email@example.com]}\n"
        "\\date{\\today}\n\n"
    )
    src = src[:start] + new_block + src[end:]

src = src.replace("\\begin{IEEEkeywords}", "\\noindent\\textbf{Keywords ---} ")
src = src.replace("\\end{IEEEkeywords}", "\n")

(ROOT / "check.tex").write_text(src)
print("wrote", ROOT / "check.tex")
