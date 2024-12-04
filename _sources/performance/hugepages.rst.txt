.. _hugepages:

Huge Pages
==========

SVS explicitly uses huge pages to reduce the virtual memory overheads. By doing so, it can achieve significant
performance gains for large datasets. For example, for the standard 96-dimensional dataset `Deep <http://sites.skoltech.ru/compvision/noimi/>`_, we observe throughput
improvements of 20% to 90% when searching on 100 million to 1 billion points [ABHT23]_ (see :ref:`huge_pages_usage` for details).

.. contents::
   :local:
   :depth: 1

What are huge pages and why are they useful?
--------------------------------------------

Modern computer systems use virtual memory to create the illusion that each program is running in its own address space.
With the help of the operating system (OS) and underlying hardware, address translation tables are created for each process.
These translation tables convert the memory address used by the process (the "virtual" address) into a real "physical" address of where the memory actually resides on the hardware.

Addresses are grouped into blocks, or "pages", to help this translation procedure.
By default, the size of a page is 4 kB.
Each time a process accesses a virtual memory address, the hardware/OS combination determines which page the addresses is on, and then looks up the physical address in a page table.
Since this has to be done for each address reference, CPUs have a structure called a Translation Lookaside-Buffer (TLB) that caches recently used virtual to physical page translations.
This greatly speeds up the translation process.

Now, there is a trade off to be made for pages sizes.
Larger pages mean that fewer TLB entries are required to cover a given address space.
Since the TLB entries (i.e. virtual to physical address mappings) are also stored in memory, larger page sizes can reduce time spent waiting to fill TLB misses.

However, larger page sizes also affect the granularity of memory allocations.
If, for example, 1 GB page sizes are used, this means that the minimum allocation size (ignoring library support for managing objects within a single page) is 1 GB.
This is extremely wasteful for allocating a lot of small objects.
The default page size that people have settled on is 4 KB, which strikes a balance between allocation size and TLB efficiency.

For some applications such as large scale graph-based similarity search, a TLB miss for each random access during
graph traversal is nearly certain when using typical 4096 kB pages, as the probability of the corresponding page table
entry being resident in cache is nearly zero. That's where **hugepages come into play**.
Linux based operating systems support two sizes of huge pages: 2 MB and 1 GB.
Using 2 MB or 1 GB huge pages greatly increases the probability that the missed page-table entry is in the cache. SVS
uses large contiguous block allocations and implements explicit huge page allocators to take advantage of this.

Allocating Huge Pages
---------------------

Allocating huge pages is pretty simple:

.. code-block:: sh

    sudo hugeadm --obey-mempolicy --pool-pages-min=1G:64
    sudo hugeadm --create-mounts
    hugeadm --pool-list

Lets break this down:

- ``hugeadm``: hugepage administration tool.

    - ``--pool-pages-min=1G:64``: Minimum number of pages to allocate.
      Here, we are allocating 64 ``1 GB`` huge pages.
      We can also allocate ``2 MB`` huge pages by using ``2M`` instead of ``1G``.
    - ``--create-mounts``: Creates mount points for each supported huge page size.
    - ``--pool-list``: Displays the Minimum,  Current  and  Maximum  number  of  huge pages in the system.

- ``sudo``: We're messing with OS level stuff. We need superuser privileges.

**How many huge pages should I allocate?** Enough to fit the graph and the vectors.
The sizes of the graph and vectors in bytes are given by:

.. math::

    \text{graph size} &= 4 * \text{graph_max_degree} * n

    \text{vectors size} &= d * \text{bytes_per_dimension} * n

where :math:`n` is the total number of points in the dataset, ``graph_max_degree`` is the :ref:`graph maximum out degree <graph-build-param-setting>`,
:math:`d` is the vector dimensionality, and :math:`\text{bytes_per_dimension}`
is the number of bytes used per vector dimension (e.g., 4 in the case of float32-valued vectors).

.. note::

    It's best to allocate huge pages early after a system reboot.
    Huge pages must be made of physically contiguous memory.
    If a system has been running for a while, it may be impossible for the OS to find enough free contiguous memory and the allocation will fail.


Allocating Huge Pages in NUMA Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using a server system with multiple NUMA domains, use the following command

.. code-block:: sh

    sudo numactl --cpunodebind=1 --membind=1 hugeadm --obey-mempolicy --pool-pages-min=1G:64
    sudo hugeadm --create-mounts

Lets break this down:

- ``hugeadm``: hugepage administration tool.

    - ``--obey-mempolicy``: If running under ``numactl`` to control NUMA domains, this flag forces ``hugeadm`` to obey its current NUMA policy.
      Huge pages are allocated per NUMA node, so it's important to make sure this is correct for your workload.
    - ``--pool-pages-min=1G:64``: Minimum number of pages to allocate.
      Here, we are allocating 64 ``1 GB`` huge pages.
      We can also allocate ``2 MB`` huge pages by using ``2M`` instead of ``1G``.

- ``numactl``: Select which NUMA domain to allocate on

    - ``--cpunodebind=1``: Allocate to NUMA node 1
    - ``--membind=1``: Allocate to NUMA node 1

- ``sudo``: We're messing with OS level stuff. We need superuser privileges.
