# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


# Pre-installation script executed before server installation

echo "Building dlib ..."
build_dir="/lab/.sys/lib/dlib-cpp/build"

if [ ! -d "$build_dir" ]; then
    mkdir -p $build_dir

    cd $build_dir
    cmake -DUSE_AVX_INSTRUCTIONS=ON -DBUILD_SHARED_LIBS=1 ..
    cmake --build . --config Release
    make
fi

echo "Build done!"
echo "Installing dlib ..."

cd $build_dir
make install
<<<<<<< HEAD
ldconfig

touch $ready_file
=======
ldconfig
>>>>>>> [refactoring] simplify post-install hook
