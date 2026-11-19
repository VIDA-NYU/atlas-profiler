Installation
============

Install Atlas Profiler from PyPI:

.. code-block:: bash

   pip install atlas-profiler

For development:

.. code-block:: bash

   git clone https://github.com/VIDA-NYU/atlas-profiler.git
   cd atlas-profiler
   pip install -e .

Requirements
------------

- Python 3.10 or later.
- polars 1.44 or later; polars 2.x works too.
- PyTorch and transformers for the spatial classifier. They are imported only when the classifier runs
  (``geo_classifier=True``, the default), and its model files are downloaded on first use into
  ``$XDG_CACHE_HOME/atlas_profiler/model`` (``~/.cache/atlas_profiler/model`` by default).

Running the tests
-----------------

.. code-block:: bash

   pip install pytest
   pytest -q

   # Corpus tests on the UrbanTrace data lake (opt-in; files are checked against a pinned snapshot)
   ATLAS_PROFILER_TEST_DATA_DIR=../urbantrace/data pytest -q tests/corpus

   # Time and peak memory of the whole-file column statistics
   python benchmarks/csv_column_stats.py data.csv

``tests/TEST_PLAN.md`` describes the test layers and cases.

Building the documentation
--------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html