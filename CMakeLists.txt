cmake_minimum_required(VERSION 3.16)
project(radc_xva VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RADC_USE_OPENBLAS "Prefer OpenBLAS/BLAS backend" ON)
option(RADC_USE_MKL "Enable MKL backend if available" OFF)
option(RADC_REQUIRE_BLAS "Fail configure when no BLAS backend is available" ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

if(RADC_USE_MKL)
  find_package(MKL QUIET)
  if(MKL_FOUND)
    message(STATUS "Using MKL backend")
  else()
    message(WARNING "RADC_USE_MKL=ON but MKL not found; falling back")
  endif()
endif()

if(RADC_USE_OPENBLAS)
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
  else()
    message(WARNING "RADC_USE_OPENBLAS=ON but BLAS not found")
  endif()
endif()

set(RADC_HAS_BLAS OFF)
if(MKL_FOUND OR BLAS_FOUND)
  set(RADC_HAS_BLAS ON)
endif()
if(RADC_REQUIRE_BLAS AND NOT RADC_HAS_BLAS)
  message(FATAL_ERROR "No BLAS backend found. Install OpenBLAS/MKL or configure with -DRADC_REQUIRE_BLAS=OFF for non-performance unit-test-only builds.")
endif()

add_library(radc_core STATIC
  radc/src/config.cpp
  radc/src/logging.cpp
  radc/src/timer.cpp
  radc/src/rand_range_finder.cpp
  radc/src/small_svd.cpp
  radc/src/countsketch2d.cpp
  radc/src/exposure_cert.cpp
  radc/src/compressed_protocol.cpp
  radc/src/risk_xva.cpp
  radc/src/safety_gate.cpp
)
set_target_properties(radc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(radc_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/radc/include
)

target_link_libraries(radc_core PUBLIC MPI::MPI_CXX Threads::Threads)
if(MKL_FOUND)
  target_compile_definitions(radc_core PUBLIC RADC_HAVE_CBLAS=1)
  target_link_libraries(radc_core PUBLIC MKL::MKL)
elseif(BLAS_FOUND)
  target_compile_definitions(radc_core PUBLIC RADC_HAVE_CBLAS=1)
  target_link_libraries(radc_core PUBLIC ${BLAS_LIBRARIES})
endif()

add_library(radc SHARED
  radc/src/mpi_intercept.cpp
)

target_include_directories(radc PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/radc/include
)

target_link_libraries(radc PRIVATE radc_core MPI::MPI_CXX Threads::Threads dl)
set_target_properties(radc PROPERTIES OUTPUT_NAME radc)

add_executable(bench_allreduce_matrix bench/bench_allreduce_matrix.cpp)
add_executable(bench_xva_delta_gamma bench/bench_xva_delta_gamma.cpp)
add_executable(bench_pca_like bench/bench_pca_like.cpp)
add_executable(bench_stage3_intercept_driver bench/bench_stage3_intercept_driver.cpp)

foreach(bench_target IN ITEMS bench_allreduce_matrix bench_xva_delta_gamma bench_pca_like bench_stage3_intercept_driver)
  target_include_directories(${bench_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/radc/include)
  target_link_libraries(${bench_target} PRIVATE radc_core MPI::MPI_CXX dl)
endforeach()

enable_testing()

add_executable(test_countsketch2d tests/test_countsketch2d.cpp)
add_executable(test_exposure_cert tests/test_exposure_cert.cpp)
add_executable(test_risk_xva_lipschitz tests/test_risk_xva_lipschitz.cpp)
add_executable(test_safety_gate tests/test_safety_gate.cpp)
add_executable(test_small_svd tests/test_small_svd.cpp)
add_executable(test_rand_range_finder tests/test_rand_range_finder.cpp)
add_executable(test_mpi_intercept_fallback tests/test_mpi_intercept_fallback.cpp)
add_executable(test_mpi_intercept_compression_mode tests/test_mpi_intercept_compression_mode.cpp)
add_executable(test_mpi_intercept_exact_passthrough tests/test_mpi_intercept_exact_passthrough.cpp)

foreach(test_target IN ITEMS test_countsketch2d test_exposure_cert test_risk_xva_lipschitz test_safety_gate test_small_svd test_rand_range_finder test_mpi_intercept_fallback test_mpi_intercept_compression_mode test_mpi_intercept_exact_passthrough)
  target_include_directories(${test_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/radc/include)
  target_link_libraries(${test_target} PRIVATE radc_core MPI::MPI_CXX dl)
endforeach()

add_test(NAME test_countsketch2d COMMAND test_countsketch2d)
add_test(NAME test_exposure_cert COMMAND test_exposure_cert)
add_test(NAME test_risk_xva_lipschitz COMMAND test_risk_xva_lipschitz)
add_test(NAME test_safety_gate COMMAND test_safety_gate)
add_test(NAME test_small_svd COMMAND test_small_svd)

if(MPIEXEC_EXECUTABLE)
  set(RADC_PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:radc>")
  if(APPLE)
    set(RADC_PRELOAD_ENV "DYLD_FORCE_FLAT_NAMESPACE=1")
    list(APPEND RADC_PRELOAD_ENV "DYLD_INSERT_LIBRARIES=$<TARGET_FILE:radc>")
  endif()

  add_test(
    NAME test_mpi_intercept_fallback
    COMMAND ${CMAKE_COMMAND} -E env
      RADC_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/configs/w1_synth_small.yaml
      ${RADC_PRELOAD_ENV}
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_mpi_intercept_fallback>
  )

  add_test(
    NAME test_mpi_intercept_compression_mode
    COMMAND ${CMAKE_COMMAND} -E env
      RADC_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/configs/test_stage3_intercept.yaml
      ${RADC_PRELOAD_ENV}
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_mpi_intercept_compression_mode>
  )

  add_test(
    NAME test_mpi_intercept_compression_mode_downcast
    COMMAND ${CMAKE_COMMAND} -E env
      RADC_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/configs/test_stage3_intercept_downcast.yaml
      ${RADC_PRELOAD_ENV}
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_mpi_intercept_compression_mode>
  )

  add_test(
    NAME test_mpi_intercept_compression_mode_passthrough
    COMMAND ${CMAKE_COMMAND} -E env
      RADC_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/configs/test_stage3_intercept_passthrough.yaml
      ${RADC_PRELOAD_ENV}
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_mpi_intercept_compression_mode>
  )

  add_test(
    NAME test_mpi_intercept_exact_passthrough
    COMMAND ${CMAKE_COMMAND} -E env
      RADC_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/configs/test_stage3_exact_passthrough.yaml
      ${RADC_PRELOAD_ENV}
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_mpi_intercept_exact_passthrough>
  )

  add_test(
    NAME test_rand_range_finder
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_rand_range_finder>
  )
else()
  message(WARNING "MPIEXEC_EXECUTABLE not found; MPI tests are disabled")
endif()
