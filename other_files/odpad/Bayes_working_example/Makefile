default:
	python3 setup.py build_ext --inplace

run:
	LD_LIBRARY_PATH=$(shell pwd)/lib mpirun -n 7 --oversubscribe python3 stats.py

dummy:
	LD_LIBRARY_PATH=$(shell pwd)/lib mpirun -n 2 --oversubscribe python3 dummy_example.py

clean:
	rm -f pympi.cpp
	rm -f *.so
	rm -rf build
