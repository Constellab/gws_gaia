# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


echo "Building dlib ..."

build_dir="/app/lab/.gws/externs/dlib-cpp/build"
ready_file="$build_dir/READY"
if [ ! -f "$ready_file" ]; then
    if [ ! -d "$build_dir" ]; then
        mkdir -p $build_dir
    fi
    
    cd $build_dir
    cmake -DUSE_AVX_INSTRUCTIONS=ON -DBUILD_SHARED_LIBS=1 ..
    cmake --build . --config Release
    make
    touch $ready_file
fi

echo "Build done!"
echo "Installing dlib ..."

cd $build_dir
make install
ldconfig