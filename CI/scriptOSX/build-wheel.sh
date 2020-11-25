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
python3 setup_cython.py bdist_wheel
ls 
ls dist
cp *.whl dist/
#cp *.whl wheel/
#cp *.whl wheel/