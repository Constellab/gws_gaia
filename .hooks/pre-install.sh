# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


# Pre-installation script executed before server installation

echo "Building dlib ..."

build_dir="/lab/.gws/externs/dlib-cpp/build"
ready_file="$build_dir/READY"

if [ -d "$build_dir" ]; then
    if [ ! -f "$ready_file" ]; then
        n=1
        while [ ! -f "$ready_file" ] && [ $n -le 60 ]; do
            echo "$n/60 - A dlib build is already in progress. Sleep 10 secs ..."
            sleep 10
            n=$(( $n + 1 ))
        done
    fi
else
    mkdir -p $build_dir
fi

if [ ! -f "$ready_file" ]; then
    cd $build_dir
    cmake -DUSE_AVX_INSTRUCTIONS=ON -DBUILD_SHARED_LIBS=1 ..
    cmake --build . --config Release
    make
fi

echo "Build done!"
echo "Installing dlib ..."

cd $build_dir
make install
ldconfig

touch $ready_file