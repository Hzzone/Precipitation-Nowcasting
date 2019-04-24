Python Wrapper of VarFlow
-------------------------
A python wrapper of VarFlow (http://sourceforge.net/projects/varflow).

To install, first create a directory "build" and then run cmake.

On windows, follows the following command:
```bash
mkdir build
cmake -G "Visual Studio 14 2015 Win64" ^
-DCMAKE_BUILD_TYPE=Release ^
-DCMAKE_CONFIGURATION_TYPES="Release" ^
..
```

On linux, follows the following command
```bash
mkdir build
cd build
cmake ..
```

If OpenCV is not found, you can try adding the `-DOpenCV_DIR` flag. The following is an example:
```bash
mkdir build
cd build
cmake -DOpenCV_DIR=/usr/local/software/opencv/share/OpenCV ..
make
```

Run the example in `varflow/varflow.py` to validate the installation.

After that, you can install the package via
```bash
python setup.py develop
```
