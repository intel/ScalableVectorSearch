set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${SVS_LIB} INTERFACE Threads::Threads)
