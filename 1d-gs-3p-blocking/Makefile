

EXE=exe-1d-gs-3p-blocking

include ../common.make

all: $(EXE)

$(EXE): *.c *.h
	$(CC) $(OPT_FLAGS) $(OMP_FLAGS) $^ -o $(EXE)

clean:
	rm -f  exe* 



