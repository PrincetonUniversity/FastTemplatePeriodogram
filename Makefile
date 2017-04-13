test : 
	py.test ftperiodogram

test-coverage :
	py.test --cov=ftperiodogram ftperiodogram


test-coverage-report :
	py.test --cov-report term-missing --cov=ftperiodogram ftperiodogram
