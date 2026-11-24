# EQ FORTRAN LIB FOR EQUILIBRIA RESOLUTION
# by Salvador Blasco <salvador.blasco@uv.es>

TESTDIR=tests

FC = gfortran
F2PY = uv run f2py
RM = rm 

FFLAGS = -Wall -O2 -fPIC
LLAPACK = -llapack -lblas

# cpluss.f90 mb.f90 si.f90 nr.f90 pcf.f90 outer.f90 eq.f90 err.f90
EQ_SRCS = $(wildcard ./*.f90)

.PHONY: pymod


MODNAME=fsolver
SIGNATURE=$(MODNAME).pyf

pymod: $(EQ_SRCS)
	$(F2PY) -c $(SIGNATURE) -m $(MODNAME) $(EQ_SRCS) $^ --f90flags="$(FFLAGS)" $(LLAPACK)

signature:
	$(F2PY) --overwrite-signature -h $(SIGNATURE) eq.f90 

clean:
	$(RM) *.o *.mod
