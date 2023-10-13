.. _cpp_lib_saveload:

Data Structure Saving and Loading
=================================

This section describes the data structure saving and loading support and infrastructure.
We use the `toml++ <https://https://github.com/marzer/tomlplusplus>` library to assist with object saving and reloading.
Objects opt into this infrastructure by providing a special `save()` and `static load()` member functions.
The expected signatures and semantics of these functions will be described in this section with an API reference at the bottom.

Context Free Saving and Loading
-------------------------------

Many classes to be saved are simple enough that they may be stored entirely inside a TOML table.
We call these classes "context free" because their saving format does not depend on the directory in which an object is being saved.
The example below demonstrates a simple class implementing context free loading and saving.

.. literalinclude:: ../../../../examples/cpp/saveload.cpp
   :language: cpp
   :start-after: [context-free]
   :end-before: [context-free]

There are several things to note.
First, each class is expected to supply version (in the form of an ``svs::lib::Version``) information along with its serialization form.
The version is supplied upon reloading.
This enables classes to evolve while maintaining backwards compatibility with previously saved versions.

.. NOTE::
   It is expected that object saving will **not** make backwards incompatible changes to their saved format without incrementing the major version of the library!

   Making a breaking change to a class' saved format will also break all classes that
   transitively use this class.

Next, the object returned from the ``save()`` method is a ``svs::lib::SaveTable``, which in practice is a thin wrapper around a ``toml::table``.
The table should contain the relevant data required to reconstruct the object upon loading.
Library facilities will take care of storing the version information.

.. NOTE::
   The ``toml::table`` class stored entries as key-value pairs.
   Keys beginning with two underscores "__" are reserved by the SVS saving infrastructure.
   Outside of that, classes are free to use whatever names they like.

Finally, loading is expected to take a ``toml::table`` and version and return a reconstructed object.
The table given to ``load`` will match that given by ``save``, potentially with the addition of some reserved names (see the note above).

Implementing Save and Load
^^^^^^^^^^^^^^^^^^^^^^^^^^

The implementation of ``save`` is given below.

.. literalinclude:: ../../../../examples/cpp/saveload.cpp
   :language: cpp
   :start-after: [context-free-saving]
   :end-before: [context-free-saving]

There is not much too it.
Each member of ``ContextFreeSaveable`` is stored as a key-value pair ``svs::lib::SaveTable``.
The version is passed as the first argument to the constructor of ``svs::lib::SaveTable`` and the entries are passed as a ``std::initializer_list`` of key-value pairs.
Keys are string-like and values should be obtained through calls to ``svs::lib::save()``.
The example shows two equivalent ways of calling ``svs::lib::save()``.
First is a direct invocation that specified the key name ("a") and passes the member ``a_`` to ``svs::lib::save()``.
The other uses the convenience macro ``SVS_LIST_SAVE_`` to automatically derive the key based on the member name.

.. NOTE::
   When using the ``SVS_LIST_SAVE_`` and ``SVS_MEMBER_LOAD_AT_`` helper macros that end in underscores, a trailing underscore will be automatically appended to the target variable name.

Loading is also straightforward.

.. literalinclude:: ../../../../examples/cpp/saveload.cpp
   :language: cpp
   :start-after: [context-free-loading]
   :end-before: [context-free-loading]

A version check is performed (this is where backwards compatibility may be implemented) and the relevant fields are recovered from the TOML table.
The ``svs::lib::load_at`` method is used to extract the element from the table at a specific key.
Alternatively, the macro ``SVS_LOAD_MEMBER_AT_`` can be used to automatically determine the type of the object to load.

Using Save and Load
^^^^^^^^^^^^^^^^^^^

Saving and restoring and object to/from disk is easy.

.. literalinclude:: ../../../../examples/cpp/saveload.cpp
   :language: cpp
   :start-after: [saving-and-reloading-context-free]
   :end-before: [saving-and-reloading-context-free]

The example above shows constructing a ``ContextFreeSavable``, saving it to a serialized form using ``svs::lib::save``, and reloading it with ``svs::lib::load``.
It further shows that we can save and reload the data structure to disk using ``svs::lib::save_to_disk`` and ``svs::lib::load_from_disk`` respectively.
Note that a directory is required instead of a simple ``.toml`` file because in the next section, we will discuss contextual saving, which may require multiple files.
By storing the saved object in a directory, we maintain the same API.

One advantage of context free saving is that we can save an entire object inside a TOML table.
This allows usage like the following example, which can be used to construct more advanced object saving in testing and benchmarking pipelines.

.. literalinclude:: ../../../../examples/cpp/saveload.cpp
   :language: cpp
   :start-after: [saving-to-table]
   :end-before: [saving-to-table]

Contextual Saving and Loading
-----------------------------

Context free saving and loading is great for small key-value-like data structures.
However, larger data structures like datasets and graphs can carry significant binary state unsuitable for storage in a TOML file.
Instead, it is preferable to store this state in one or more auxiliary binary files that are rediscovered from the TOML configuration when loading.
These data structures are "contextual" because they require run-time context in the form of the directory being processed in addition to the TOML format.

The example class definition below shows a class that implements contextual saving and loading.
The motivation for contextual loading is the existence of the ``std::vector<float> data_`` member.
If the size of this vector is large, saving it in a TOML file is space and time inefficient.

.. literalinclude:: ../../../../examples/cpp/saveload.cpp
   :language: cpp
   :start-after: [contextual-loading]
   :end-before: [contextual-loading]

Objects implementing contextual saving and loading have "save" and "load" methods.
However, this time they require a :cpp:class:`svs::lib::SaveContext` and :cpp:class:`svs::lib::LoadContext` respectively.
The :cpp:class:`svs::lib::SaveContext` class provides a way of obtaining the saving directory as well as facilities to generate unique filenames to avoid name clashing.
The :cpp:class:`svs::lib::LoadContext` provides the working directory when loading.
Together, these classes facilitate the generation of saved objects in a relocatable manner.

Implementing Contextual Saving and Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code snippet below shows the implementation of the contextual save method.

.. literalinclude:: ../../../../examples/cpp/saveload.cpp
   :language: cpp
   :start-after: [contextual-saving-impl]
   :end-before: [contextual-saving-impl]

To save the ``data_`` member, the ``generate_name()`` method is used to generate a unique file name in the saving directory.
The contents of the vector are then saved directly to this file.
We need to find this file when reloading the data structure.
However, the variable ``fullpath`` returned by ``generate_name()`` is an absolute path.
When we store this filepath in the TOML table, we need to ensure that we only store the final filename.
When reloading, the full path will be recreated using the :cpp:class:`svs::lib::LoadContext`.

This example also demonstrates another important concept: recursive saving.
The ``Saveable`` class has a ``ContextFreeMember``.
To save the member, ``svs::lib::save`` is used, which will perform all the necessary steps to save that member class and return its generated TOML table, which can then be nested inside the ``Saveable``'s TOML table.

Reloading is similar.

.. literalinclude:: ../../../../examples/cpp/saveload.cpp
   :language: cpp
   :start-after: [contextual-loading-impl]
   :end-before: [contextual-loading-impl]

Here we see the directory obtained from the load context combined with the file name stored in the TOML table to recreate the full filepath for the saved binary data.
The function :cpp:func:`svs::lib::load` is used to load the saveable subobject.
End to end saving is shown below.

.. literalinclude:: ../../../../examples/cpp/saveload.cpp
   :language: cpp
   :start-after: [saving-and-reloading-contextual]
   :end-before: [saving-and-reloading-contextual]

General Guidelines
------------------

* Prefer context-free loading and saving if possible.
  It is more flexible and allows for more uses than contextual saving.
* Use :cpp:func:`svs::lib::save` and :cpp:func:`svs::lib::load` to save and reload saveable sub-objects.

STL Support
-----------

The saving and loading infrastructure has support for several built-in types, including ``std::vector``.
The example below demonstrates the use of ``std::vector``.

.. literalinclude:: ../../../../examples/cpp/saveload.cpp
   :language: cpp
   :start-after: [saving-and-reloading-vector]
   :end-before: [saving-and-reloading-vector]

The list of built-in types is:

.. table:: STL Support
   :width: 100

   +--------------------------------------------+-------------------------------------------------------------------------------------------------------------------+
   | Type Class                                 | Notes                                                                                                             |
   +--------------------------------------------+-------------------------------------------------------------------------------------------------------------------+
   | Integers                                   | Will error if the conversion from TOML's ``int64_t`` type is lossy.                                               |
   +--------------------------------------------+-------------------------------------------------------------------------------------------------------------------+
   | ``float``, ``double``                      | Lossy conversion allowed for ``float`` to support literals like "1.2".                                            |
   +--------------------------------------------+-------------------------------------------------------------------------------------------------------------------+
   | ``std::string``, ``std::filesystem::path`` |                                                                                                                   |
   +--------------------------------------------+-------------------------------------------------------------------------------------------------------------------+
   | ``std::vector<T, Alloc>``                  | Can optionally take an allocator as the first non-context argument. Loading is contextual if ``T`` is contextual. |
   +--------------------------------------------+-------------------------------------------------------------------------------------------------------------------+

Advanced Features
-----------------

Load Helpers
^^^^^^^^^^^^

Until now, it has been assumed that the class to be loaded implements a static ``load`` method.
However, this is not always convenient nor least verbose.
All of the load methods described so far can take an instance of a class for the first argument.
As long as this object has an appropriate ``load()`` method as described above, it can be used.
In fact, the return type is not constrained, so this "load helper" may be used to create any other class.

Load Argument Forwarding
^^^^^^^^^^^^^^^^^^^^^^^^

For all loading methods, an arbitrary number of trailing arguments can be appended to any call
These arguments will be forwarded to the final ``load()`` method.
This allows run-time context that isn't necessary to be saveable or that can change from run to run (for example: allocator) to be given.

Load and Save Override
^^^^^^^^^^^^^^^^^^^^^^

It is occasionally useful to use a lambda to implement ad-hoc loading and saving of some sub-components of a larger class.
This can be be done by passing the lambda to the :cpp:class:`svs::lib::SaveOverride` and :cpp:class:`svs::lib::LoadOverride` classes respectively and passing these to the various saving and loading methods.

Power-User Functionality
^^^^^^^^^^^^^^^^^^^^^^^^

Saving and loading plumbing for a class ``T`` passes through :cpp:class:`svs::lib::Saver<T>` and :cpp:class:`svs::lib::Loader<T>` proxy classes.
If ``T`` implements member ``save`` and ``load`` methods, then the default definition for these proxy classes will "do the right thing" and call those methods.
Alternatively, classes may chose to explicitly specialize these classes.

See the documentation on those classes for details.
API Reference
-------------

.. doxygenclass:: svs::lib::SaveContext
   :project: SVS
   :members:

.. doxygenclass:: svs::lib::LoadContext
   :project: SVS
   :members:

.. doxygenclass:: svs::lib::SaveTable
   :project: SVS
   :members:

Saving Related Methods
^^^^^^^^^^^^^^^^^^^^^^

.. doxygengroup:: save_group
   :project: SVS
   :members:
   :content-only:

.. doxygenfunction:: svs::lib::save_to_table
   :project: SVS

.. doxygenfunction:: svs::lib::save_to_disk
   :project: SVS

Loading Related Methods
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygengroup:: load_group
   :project: SVS
   :members:
   :content-only:

.. doxygengroup:: load_from_disk_group
   :project: SVS
   :members:
   :content-only:

Save and Load Overrides
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: svs::lib::SaveOverride
   :project: SVS
   :members:

.. doxygenclass:: svs::lib::LoadOverride
   :project: SVS
   :members:

