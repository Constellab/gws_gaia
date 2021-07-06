# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


# Pre-installation script executed before server installation

echo "Building dlib ..."

build_dir="/lab/.gws/externs/dlib-cpp/build"
in_progress_file="$build_dir/IN_PROGRESS"
ready_file="$build_dir/READY"

if [ -d "$build_dir" ]; then
    if [ ! -f "$ready_file" ]; then
        n=1
        while [ ! -f "$ready_file" ] && [ $n -le 30 ]; do
            echo "$n/30 - A dlib build is already in progress. Sleep 10 secs ..."
            sleep 10
            n=$(( $n + 1 ))
        done
    fi
fi

if [ ! -f "$ready_file" ]; then
    if [ ! -d "$build_dir" ]; then
        mkdir -p $build_dir
    fi
    
    cd $build_dir

    touch $in_progress_file
    cmake -DUSE_AVX_INSTRUCTIONS=ON -DBUILD_SHARED_LIBS=1 ..
    cmake --build . --config Release
    make
    touch $ready_file
    rm -r $in_progress_file
fi

echo "Build done!"
echo "Installing dlib ..."

cd $build_dir
make install
ldconfig