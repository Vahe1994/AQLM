# (rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake -DBUILD_BINDING=ON -DCMAKE_PREFIX_PATH=/Users/blacksamorez/reps/libtorch ..) # -DCMAKE_TOOLCHAIN_FILE=/Users/blacksamorez/Library/Android/sdk/ndk/27.0.12077973/build/cmake/android.toolchain.cmake  -DANDROID_ABI=arm64-v8a
cmake --build cmake-out -j9
# rm -rf executorch
