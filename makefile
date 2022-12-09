.PHONY: clean

report.pdf: report.tex
	pdflatex report.tex 2>&1

clean:
	rm -rf report.aux report.pdf part1/__pycache__ part2/__pycache__
