report.pdf: report.tex
	@echo -n Compiling $@ ...
	@pdflatex report.tex 2>&1 >/dev/null
	@echo done
	@rm *.log *.aux
