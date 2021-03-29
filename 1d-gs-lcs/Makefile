
EXE=exe-1d-gs-lcs

include ../common.make

all: $(EXE)

$(EXE): main.c define.h naive_scalar.c vectime.c
	$(CC) $(OPT_FLAGS) $^ -o $(EXE)

clean:
	rm -f  exe* 

