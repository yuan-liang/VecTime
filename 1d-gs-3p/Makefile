
EXE=exe-1d-gs-3p

include ../common.make

all: $(EXE)

$(EXE): main.c vectime.c naive_scalar.c define.h
	$(CC) $(OPT_FLAGS) $^ -o $(EXE)

clean:
	rm -f  exe* 

