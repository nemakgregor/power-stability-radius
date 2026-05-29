# Paper Build

This folder is a self-contained LaTeX paper project.

## Files

- `template.tex` -- main MDPI manuscript, based on the supplied MDPI template.
- `manuscript.tex` -- journal-agnostic draft retained as a working backup.
- `references.bib` -- bibliography database.
- `figures/` -- figures used by the manuscript.
- `tables/` -- LaTeX table fragments included by the manuscript.
- `reviewer_simulation.md` -- pre-submission reviewer-risk notes, not compiled.

## Build

Preferred:

```bash
latexmk -pdf template.tex
```

Fallback:

```bash
pdflatex template.tex
bibtex template
pdflatex template.tex
pdflatex template.tex
```

On Windows PowerShell:

```powershell
./compile.ps1
```

The current environment used to create this draft did not have `latexmk`,
`pdflatex`, or `tectonic` installed, so only path and citation-key checks were
run here.
