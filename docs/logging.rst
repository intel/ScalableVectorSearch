.. Copyright (C) 2024 Intel Corporation
..
.. This software and the related documents are Intel copyrighted materials,
.. and your use of them is governed by the express license under which they
.. were provided to you ("License"). Unless the License provides otherwise,
.. you may not use, modify, copy, publish, distribute, disclose or transmit
.. this software or the related documents without Intel's prior written
.. permission.
..
.. This software and the related documents are provided as is, with no
.. express or implied warranties, other than those that are expressly stated
.. in the License.

.. _logging_top:

Logging
#######

SVS uses the excellent `spdlog <https://github.com/gabime/spdlog>` to support logging functionality.
Logging levels (in order of increasing severity) are:

.. table:: SVS Logging Levels
   :width: 100

   +-----------+-------------------------+-----------------------------------------------------+
   | Log Level | Environment Variables   | Descriptions                                        |
   +===========+=========================+=====================================================+
   | Trace     | ``trace``, ``TRACE``    | Tracing control flow through functions.             |
   |           |                         | Potentially extremely verbose.                      |
   +-----------+-------------------------+-----------------------------------------------------+
   | Debug     | ``debug``, ``DEBUG``    | Verbose logging useful for debugging.               |
   +-----------+-------------------------+-----------------------------------------------------+
   | Info      | ``info``, ``INFO``      | Informative prints. This level is used by           |
   |           |                         | long-running process to provide periodic updates    |
   |           |                         | that progress is being made.                        |
   +-----------+-------------------------+-----------------------------------------------------+
   | **Warn**  | ``warn``, ``WARN``      | Diagnostic warnings about feature deprecation and   |
   | (default) |                         | other situations upon which the user may need to    |
   |           |                         | take action.                                        |
   +-----------+-------------------------+-----------------------------------------------------+
   | Error     | ``error``, ``ERROR``    | Hard program errors that which may be recoverable   |
   |           |                         | but likely require intervention from the user.      |
   +-----------+-------------------------+-----------------------------------------------------+
   | Critical  | ``critical``,           | Critical errors that must be addressed and will     |
   |           | ``CRITICAL``            | likely lead to program termination.                 |
   +-----------+-------------------------+-----------------------------------------------------+
   | Off       | ``off``, ``OFF``        | Disable loging.                                     |
   +-----------+-------------------------+-----------------------------------------------------+

Each logging level suppresses all diagnostic messages with a lower severity.

If SVS is compiled using ``-DSVS_INITIALIZE_LOGGER=YES``, then the environment variable ``SVS_LOG_LEVEL`` can be used set the initial logging level.
The string values for these variables is provided in the table above.
For example, starting an SVS application with

.. code::

    SVS_LOG_LEVEL=debug ./application

Will initialize logging at the "debug" level, passing through messages with severity "debug" or higher.
If this environment variable is not supplied or if it is the empty string, then the default logging level of "warn" will be used.
Using any other string value for the environment variable results in unspecified behavior.

.. HINT::

   Generally, if SVS doesn't understand a string value, it will default to "warn".
   However, this behvaior should not be relied upon.

Logging Sinks
-------------

By default, log messages will be sent to ``stdout``.
This behavior can be changed if desired using the ``SVS_LOG_SINK`` environment variable.
Values for this variable are shown below.

.. table:: SVS Built-in Sinks
   :width: 100

   +-------------+-----------------------+-----------------------------------------------------+
   | Sink        | Enironment Variable   | Description                                         |
   +=============+=======================+=====================================================+
   | **stdout**  | ``stdout``            | Send all log messages to ``stdout``.                |
   | (default)   |                       |                                                     |
   +-------------+-----------------------+-----------------------------------------------------+
   | stderr      | ``stderr``            | Send all log messages to ``stderr``.                |
   +-------------+-----------------------+-----------------------------------------------------+
   | null        | ``null``              | Send all log messages into the abyss (no logging    |
   |             |                       | whatsoever).                                        |
   +-------------+-----------------------+-----------------------------------------------------+
   | file        | ``file:path/to/file`` | Write log messages to the file ``path/to/file``.    |
   |             |                       | SVS will try to make  all paths necessary to create |
   |             |                       | the file. Insufficient permissions will result in   |
   |             |                       | an exception.                                       |
   |             |                       |                                                     |
   |             |                       | The contents of any existing file with the exact    |
   |             |                       | same path will be deleted.                          |
   +-------------+-----------------------+-----------------------------------------------------+

For example, to send all log messages to a file ``~/svslog.txt``, an application linked with SVS could be started as

.. code::

    SVS_LOG_SINK=file:~/log.txt ./application

.. NOTE::

   Writes to a file are buffered and periodically flushed.
   If an SVS application is interrupted and that brings down the application, it is possible that some log messages could be lost.

Logging API
-----------

Both the C++ library and ``svs`` provide APIs for configuring logging, with the former accepting any instance of a ``std::shared_ptr<spdlog::logger>``.
Configuration made through the API superseded those configured using the environment variables.

* :ref:`Link to C++ API <cpp_core_logging>`
* :ref:`Link to svs API <python_logging>`

.. HINT::

   Environment variable based initialization can be disabled by compiling SVS applications
   with ``-DSVS_INITIALIZE_LOGGER=NO``. The global logger will then have a null sink and
   no messages will be logged.
