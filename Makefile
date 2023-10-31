build: clean
	mkdir build
	cmake -D CMAKE_BUILD_TYPE=Release -S . -B build

debug: clean
	mkdir build
	cmake -D CMAKE_BUILD_TYPE=Debug -S . -B build

clean:
	rm -rf build
