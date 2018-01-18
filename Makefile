all: install test

install:
	python setup.py install

test:
	-python ./pydlm/tests/testOdlm.py
	-python ./pydlm/tests/func/test_odlm.py
	-./test.py
