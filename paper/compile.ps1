$ErrorActionPreference = "Stop"

if (Get-Command latexmk -ErrorAction SilentlyContinue) {
    latexmk -pdf template.tex
    exit $LASTEXITCODE
}

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    throw "Neither latexmk nor pdflatex is available on PATH."
}

pdflatex template.tex
bibtex template
pdflatex template.tex
pdflatex template.tex
