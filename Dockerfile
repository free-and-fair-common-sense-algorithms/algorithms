FROM jupyter/datascience-notebook:9f9e5ca8fe5a
RUN pip install \
  # pandas=1.0.0 \
  git+https://github.com/free-and-fair-common-sense-algorithms/sklearn_transformers.git#egg=sklearn_transformers
