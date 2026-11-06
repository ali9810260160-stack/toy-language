cmake_minimum_required(VERSION 3.20.0)

project(my-toy LANGUAGES CXX C)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")


list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR} ${LLVM_CMAKE_DIR})

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

include_directories(include)


add_definitions(${LLVM_DEFINITIONS})


add_executable(toyc-ch1 main.cpp
        include/toy/lexer.h
        include/toy/AST.h
        include/toy/Parser.h
        AST.cpp)


llvm_update_compile_flags(toyc-ch1)

target_link_libraries(toyc-ch1 PRIVATE LLVMSupport)

set(LLVM_TARGET_DEFINITIONS include/toy/Ops.td)
mlir_tablegen(include/toy/Ops.h.inc -gen-op-decls)
mlir_tablegen(include/toy/Ops.cpp.inc -gen-op-defs)
mlir_tablegen(include/toy/Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/toy/Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRToyOpsIncGen)

add_dependencies(toyc-ch1 MLIRToyOpsIncGen)
