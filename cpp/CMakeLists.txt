# hft-autofactor/cpp — static engine library + hftaf-engine CLI + tests.
set(HFTAF_SOURCES
    src/io.cpp
    src/decode.cpp
    src/session.cpp
    src/book.cpp
    src/labels.cpp
    src/output.cpp
    src/factors_snapshot.cpp
    src/factors_tick.cpp
    src/factory.cpp
    src/engine.cpp
)

add_library(hftaf STATIC ${HFTAF_SOURCES})
target_include_directories(hftaf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hftaf PUBLIC ZLIB::ZLIB)

add_executable(hftaf-engine src/main.cpp)
target_link_libraries(hftaf-engine PRIVATE hftaf)

add_subdirectory(tests)
