.PHONY: all clean bench

all:
	morloc make -o test test.loc
	./test

clean:
	rm -rf test test-build pools/ bench bench-cross *-build* *.dat *.log

bench:
	bash ../parallel/bench/run.sh cpp
