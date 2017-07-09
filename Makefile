package=ftperiodogram

clean : 
	rm $(package)/*pyc $(package)/tests/*pyc

test : 
	py.test $(package)

test-coverage :
	py.test --cov=$(package) $(package)

test-coverage-report :
	py.test --cov-report term-missing --cov=$(package) $(package)


