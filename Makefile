.PHONY: pdf clean

pdf: overview.pdf

overview.pdf: overview.md
	pandoc overview.md -o overview.pdf

clean:
	rm -f overview.pdf

.DEFAULT_GOAL := pdf
