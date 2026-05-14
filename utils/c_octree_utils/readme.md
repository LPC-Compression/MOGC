## compile
c++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) octreeutils.cpp -o octreeutils$(python3-config --extension-suffix)

PYTHONPATH=. pybind11-stubgen octreeutils -o ./