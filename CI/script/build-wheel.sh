#!/bin/bash

# build wheel
#PATH=/opt/usr/bin/:/opt/python/cp27-cp27m/bin:${PATH}  pip wheel ./unsio/ -w wheelhouse

#rm -f wheel/*pyds9plugin*manylinux*.whl wheel/*pyds9plugin*tar.gz wheelhouse/*pyds9plugin*manylinux*.whl
rm -f wheel/*pyds9plugin*manylinux*.whl

# build wheel
# for PYBIN in /opt/python/*/bin/; do
#     echo "Compiling using pip version ${PYBIN}...."
#     PATH=/opt/usr/bin/:${PYBIN}:${PATH}   pip wheel --no-deps ./ -w wheelhouse	
# done

for PYBIN in /opt/python/*/bin/python; do
	echo "Compiling using pip version ${PYBIN}...."
	${PYBIN} setup_cython.py bdist_wheel
done

#python3 setup_cython.py bdist_wheel
