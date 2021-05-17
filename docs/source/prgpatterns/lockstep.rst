.. _prgpatterns-lockstep:

.. seealso::

   In order to follow this section, you need to understand the `CUDA programming model <http://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model>`_.

Lockstep Programming Model
==========================

.. sectionauthor:: René Widera, Axel Huebl

The *lockstep programming model* structures code that is evaluated collectively and independently by workers (physical threads).
Actual processing is described by one-dimensional index domains of *virtual workers* which can even be changed within a kernel.
Mathematically, index domains are none-injective, total functions on physical workers.

An index domain is **independent** from data but **can** be mapped to a data domain, e.g. one to one or with more complex mappings.

Code which is implemented by the *lockstep programming model* is free of any dependencies between the number of worker and processed data elements.
To simplify the implementation, each index within a domain can be seen as a *virtual worker* which is processing one data element (like the common workflow to programming CUDA).
Each *worker* :math:`i` can be executed as :math:`N_i` *virtual workers* (:math:`1:N_i`).

To transfer information from a virtual working between lock steps you can use a context variable ``CtxVar``, similar to a temporary local variable in a function.

Functors passed into lock step routines can have three different parameter signatures.

* No parameter, if the work is independent of the domain size

.. code-block:: bash

[&](){ }


* An unsigned 32bit integral parameter if the work depends on indices within the domain ``range [0,domain size)``

.. code-block:: bash

[&](uint32_t const linearIdx){}


* ``DomainIdx`` as parameter. DomainIdx is holing the linear index within the domain and meta information to access a context variables.

.. code-block:: bash

[&](pmacc::mappings::threads::DomainIdx const domIdx){}


pmacc helpers
-------------

.. doxygenstruct:: pmacc::mappings::threads::IdxConfig
   :project: PIConGPU

.. doxygenstruct:: pmacc::memory::CtxVar
   :project: PIConGPU

.. doxygenstruct:: pmacc::mappings::threads::ForEachIdx
   :project: PIConGPU

Common Patterns
---------------

Collective Loop
^^^^^^^^^^^^^^^

* each worker needs to pass a loop N times
* in this example, there are more dates than workers that process them

.. code-block:: bash

    // `frame` is a list which must be traversed collectively
    while( frame.isValid() )
    {
        uint32_t const workerIdx = cupla::threadIdx( acc ).x;
        using ParticleDomCfg = IdxConfig<
            frameSize,
            numWorker
        >;
        ForEachIdx< ParticleDomCfg > forEachParticle( workerIdx );
        forEachParticle(
           [&]( DomainIdx const domIdx )
           {
               // independent work
           }
        forEachParticle(
           [&]( uint32_t const linearIdx )
           {
               // independent work based on the linear index
           }
       );
    }


Non-Collective Loop
^^^^^^^^^^^^^^^^^^^

* each *virtual worker* increments a private variable

.. code-block:: cpp

    uint32_t const workerIdx = cupla::threadIdx( acc ).x;
    using ParticleDomCfg = IdxConfig<
        frameSize,
        numWorkers
    >;
    ForEachIdx< ParticleDomCfg > forEachParticle( workerIdx );
    memory::CtxVar< int, ParticleDomCfg > vWorkerIdx( 0 );
    forEachParticle(
        [&]( auto const domIdx )
        {
            vWorkerIdx[ domIdx ] = domIdx.lIdx();
            for( int i = 0; i < 100; i++ )
                vWorkerIdx[ domIdx ]++;
        }
    );


Create a Context Variable
^^^^^^^^^^^^^^^^^^^^^^^^^

* ... and initialize with the index of the virtual worker

.. code-block:: cpp

    uint32_t const workerIdx = cupla::threadIdx( acc ).x;
    using ParticleDomCfg = IdxConfig<
        frameSize,
        numWorkers
    >;
    memory::CtxVar< int, ParticleDomCfg > vIdx(
        workerIdx,
        [&]( DomainIdx const domIdx ) -> int32_t
        {
            return domIdx.lIdx();
        }
    );

    // is equal to

    memory::CtxVar< int, ParticleDomCfg > vIdx;
    ForEachIdx< ParticleDomCfg >{ workerIdx }(
        [&]( DomainIdx const domIdx )
        {
            vIdx[ domIdx ] = domIdx.lIdx();
        }
    );


* Data from a context variable can be accessed within independent lock steps.

.. code-block:: cpp

    uint32_t const workerIdx = cupla::threadIdx( acc ).x;
    using ExampleDomCfg = IdxConfig<
        42,
        numWorkers
    >;
    memory::CtxVar< int, ExampleDomCfg > vIdx(
        workerIdx,
        [&]( DomainIdx const domIdx ) -> int32_t
        {
            return domIdx.lIdx();
        }
    );

    ForEachIdx< ExampleDomCfg > forEachExample{ workerIdx };

    forEachExample(
        [&]( DomainIdx const domIdx )
        {
            printf( "virtual worker linear idx: %u == %u\n", vIdx[ domIdx ], domIdx.lIdx() );
        }
    );

    forEachExample(
        [&]( DomainIdx const domIdx )
        {
            printf( "nothing changed: %u == %u\n", vIdx[ domIdx ], domIdx.lIdx() );
        }
    );


Using a Master Worker
^^^^^^^^^^^^^^^^^^^^^

* only one *virtual worker* (called *master*) of all available ``numWorkers`` manipulates a shared data structure for all others

.. code-block:: cpp

    // example: allocate shared memory (uninitialized)
    PMACC_SMEM(
        finished,
        bool
    );

    uint32_t const workerIdx = cupla::threadIdx( acc ).x;
    ForEachIdx<
        IdxConfig<
            1,
            numWorkers
        >
    > onlyMaster{ workerIdx };

    // manipulate shared memory
    onlyMaster(
        [&]( )
        {
            finished = true;
        }
    );

    /* important: synchronize now, in case upcoming operations (with
     * other workers) access that manipulated shared memory section
     */
    cupla::__syncthreads( acc );
