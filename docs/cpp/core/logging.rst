.. _cpp_core_logging:

Logging API
===========

The core C++ library offers an API for managing :ref:`logging <logging_top>` in the ``svs/core/logging.h`` header.
Use of the logging API will supersede environment variable initialization (if applicable).

Logging Functions
-----------------

.. doxygenfunction:: svs::logging::log(logger_ptr&, Level, fmt::format_string<Args...>, Args&&...)
   :project: SVS

.. doxygenfunction:: svs::logging::log(Level, fmt::format_string<Args...>, Args&&...)
   :project: SVS


Extra Definitions
-----------------

.. doxygenenum:: svs::logging::Level
   :project: SVS

.. doxygentypedef:: svs::logging::logger_ptr
   :project: SVS

.. doxygentypedef:: svs::logging::sink_ptr
   :project: SVS

.. doxygenfunction:: svs::logging::global_logger()
   :project: SVS

.. doxygenfunction:: svs::logging::get
   :project: SVS

.. doxygenfunction:: svs::logging::set(const logger_ptr&)
   :project: SVS

.. doxygenfunction:: svs::logging::set(logger_ptr&&)
   :project: SVS

Sinks
-----

.. doxygenfunction:: svs::logging::null_sink
   :project: SVS

.. doxygenfunction:: svs::logging::stdout_sink
   :project: SVS

.. doxygenfunction:: svs::logging::stderr_sink
   :project: SVS

.. doxygenfunction:: svs::logging::file_sink
   :project: SVS


