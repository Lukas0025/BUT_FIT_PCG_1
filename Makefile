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

test: build build/input.h5
	cd build && \
	./nbodyCpu 4096 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5 && \
	./nbody0 4096 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5

clean:
	rm -rf build
