@echo off
cd build

echo == Make
cmake -G"MinGW Makefiles" ../
echo == Install
cmake --install .
echo == Build
cmake --build .

echo == Execute
aml.exe

cd ..
