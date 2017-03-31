test : 
	py.test tempfit

test-coverage :
	py.test --cov=tempfit tempfit


test-coverage-report :
	py.test --cov-report term-missing --cov=tempfit tempfit
