.PHONY: docs deps git startup tests

deps:
	. bin/activate && pip install -r requirements.txt

docs:
	black --line-length 100 fragmentation_uncertainty
	cd docs && sphinx-apidoc -o ./source ../fragmentation_uncertainty -f && make html

git:
	git add --all
	git commit -m "update"
	git push origin

startup:	
	conda deactivate
	source bin/activate
	. bin/activate 

tests: