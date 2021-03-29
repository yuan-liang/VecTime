

EXE=exe-3d-jacobi-7p-blocking

include ../common.make

all: $(EXE)

$(EXE):  *.h *.c
	$(CC) $(OPT_FLAGS) $(OMP_FLAGS) $^ -o $(EXE)

clean:
	rm -f  exe* 


