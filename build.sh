cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=true ..
cmake --install .
cmake --build .
cd ..
./build/aml
