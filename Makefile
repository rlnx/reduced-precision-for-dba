CXXFLAGS = \
	-pedantic -Wall -Wextra -std=c++17 \
	-Wno-unused-parameter \
	-Wno-unused-variable \
	-O2 -g -fopenmp -xhost

estimate_sigma: estimate_sigma.cpp floats.hpp Makefile
	icc $(CXXFLAGS) $< -o $@ -lstdc++ -lsycl
