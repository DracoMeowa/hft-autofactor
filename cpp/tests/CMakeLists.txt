# hft-autofactor/cpp/tests — ctest targets. All tests are self-contained:
# fixtures are generated at test time (fixture.hpp writes .csv.gz via zlib),
# so nothing binary is checked in and no external tools are required.
set(HFTAF_TESTS
    test_decode
    test_book
    test_labels
    test_factors
    test_determinism
    test_cache
)

foreach(t ${HFTAF_TESTS})
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE hftaf)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
