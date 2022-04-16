all: cholmod_c_test

LIB2 = -lcholmod -lamd -lcolamd -lsuitesparseconfig \
	-llapack -lblas -lm -lrt
I = -I/usr/include/suitesparse
C = cc -O3 -fexceptions -fPIC -fopenmp -DNPARTITION -DNCAMD
RM = rm -f
CLEAN = *.o

cholmod_c_test: cholmod_c_test.c
	$(C) -o cholmod_c_test $(I) cholmod_c_test.c $(LIB2)

clean: 
	- $(RM) -r $(CLEAN)
	- $(RM) cholmod_c_test