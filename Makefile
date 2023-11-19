build:
	mkdir build
	cmake -D CMAKE_BUILD_TYPE=Release -S . -B build
	cd build && $(MAKE)

update:
	cmake -D CMAKE_BUILD_TYPE=Release -S . -B build
	cd build && $(MAKE)

debug:
	mkdir build
	cmake -D CMAKE_BUILD_TYPE=Debug -S . -B build
	cd build && $(MAKE)

build/input.h5:
	cd build && \
	./gen 4096 input.h5

run: build build/input.h5
	cd build && \
	./nbody0 4096 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5 && \
	./nbody1 4096 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5 && \
	./nbody2 4096 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5 && \
	./nbody3 4096 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5 && \
	./nbody4 4096 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5

test: build build/input.h5
	./runTests.sh build/nbody0
	./runTests.sh build/nbody1
	./runTests.sh build/nbody2
	./runTests.sh build/nbody3
	./runTests.sh build/nbody4
	cd build && \
	./nbodyCpu 4096 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5 && \
	./nbody4 4096 0.01f 100 512 5 2048 128 input.h5 outputGpu.h5 && \
	../compare.sh outputCpu.h5 outputGpu.h5

vtest:
	$(MAKE) test > testout.txt
	cat testout.txt | grep ERROR 

clean:
	rm -rf build
