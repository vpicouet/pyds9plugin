FROM quay.io/pypa/manylinux2010_x86_64


# Install numpy minimum requirements for swig
# Install twin
RUN                               \
  /opt/python/cp27-cp27m/bin/pip  install numpy==1.8.2  twine    &&\
  /opt/python/cp27-cp27mu/bin/pip install numpy==1.8.2  twine    &&\
  /opt/python/cp34-cp34m/bin/pip  install numpy==1.8.2  twine    &&\
  /opt/python/cp35-cp35m/bin/pip  install numpy==1.9.0  twine    &&\
  /opt/python/cp36-cp36m/bin/pip  install numpy==1.11.3 twine    &&\
  /opt/python/cp37-cp37m/bin/pip  install numpy==1.14.5 twine    &&\
  /opt/python/cp38-cp38/bin/pip   install numpy==1.17.3 twine

ENV PATH /opt/usr/bin/:$PATH
