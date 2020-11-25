#!/bin/bash

#rm -f wheel/*pyds9plugin*osx*.whl wheelhouse/*pyds9plugin*osx*.whl
# Activate python environement
# Build wheel
# for PYENV in ${HOME}/venv/unsio*; do
#     echo "Activate ${PYENV}...."
#     source ${PYENV}/bin/activate
#     pip wheel --no-deps ./ -w wheelhouse
#     deactivate
# done
pip install cython --user
#pip install delocate --user
source /Users/grunner/venv/unsio37/bin/activate

python3 setup_cython.py bdist_wheel
ls 
echo 'ls dist'
ls dist
#cp *.whl dist/
echo 'repairing wheel'
delocate-wheel dist/pyds9plugin*.whl 

deactivate
#cp *.whl wheel/
#cp *.whl wheel/