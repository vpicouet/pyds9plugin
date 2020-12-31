#!/bin/bash

# Activate python environement
# Build wheel



rm -f dist/*yds9plugin*osx*.whl 

for PYENV in /Users/grunner/venv/unsio38*; do
    echo "Activate ${PYENV}...."
    which python
    which pip
    source ${PYENV}/bin/activate
	pip install cython 
	rm -rf build
    python setup_cython.py bdist_wheel
    # pip install cython --user
    # pip wheel --no-deps ./ -w wheelhouse
    deactivate
done

#ls -ltrh buil/


# for folder in buil/; do
# 	echo "Checking what is in build"
# 	ls $folder/*s
# done


#pip install cython --user
#python3 setup_cython.py bdist_wheel




#pip install delocate --user
#source /Users/grunner/venv/unsio37/bin/activate

#mv * dist/
# ls 
# echo 'ls dist'
# ls dist
# #cp *.whl dist/
# echo 'repairing wheel'
# delocate-wheel dist/pyds9plugin*.whl 

# deactivate
# #cp *.whl wheel/
# #cp *.whl wheel/