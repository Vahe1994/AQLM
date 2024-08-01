(rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake -DCMAKE_PREFIX_PATH=/Users/blacksamorez/reps/libtorch ..)
cmake --build cmake-out -j9
rm -rf executorch
