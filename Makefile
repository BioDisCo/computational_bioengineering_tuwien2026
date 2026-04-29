.PHONY: pdf book preview clean

pdf: overview.pdf

overview.pdf: overview.md
	pandoc overview.md -o overview.pdf

book:
	quarto render

preview:
	quarto preview

clean:
	rm -f overview.pdf
	rm -rf _book _freeze

.DEFAULT_GOAL := pdf
