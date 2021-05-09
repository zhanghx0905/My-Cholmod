all: cholmod_c_test

include ../lib/SuiteSparse_config.mk

LIB2 = $(LDFLAGS) -lcholmod -lamd -lcolamd -lsuitesparseconfig \
	$(LIB_WITH_PARTITION) $(LAPACK) $(BLAS) $(LDLIBS)
I = -I/usr/include/suitesparse
C = $(CC) $(CF) $(CHOLMOD_CONFIG) $(CONFIG_PARTITION)

cholmod_c_test: cholmod_c_test.c
	$(C) -o cholmod_c_test $(I) cholmod_c_test.c $(LIB2)

clean: 
	- $(RM) -r $(CLEAN)
	- $(RM) cholmod_c_test