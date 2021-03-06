cmake_minimum_required (VERSION 3.9)

project (vulkan_demos VERSION 0.0.0 DESCRIPTION "Various Vulkan (tm) demos")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(RUN_DIR ${PROJECT_SOURCE_DIR}/bin)

# Define C++ version to be used for building the project
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

configure_file("${PROJECT_SOURCE_DIR}/config.h.in" "${PROJECT_BINARY_DIR}/config.h")
include_directories("${PROJECT_BINARY_DIR}")

set(DEMOS
  basic
  model
  asModel
  particles
  )

# The Python interpreter is needed to compile shaders
include(FindPythonInterp)

# Libraries
if(WIN32)
  find_path(glfw3_DIR GLFW $ENV{PROGRAMFILES}/GLFW/ DOC "The directory where GLFW resides")
  find_path(glfw3_INCLUDE_DIRS GLFW/glfw3.h $ENV{PROGRAMFILES}/GLFW/include ${glfw3_DIR}/include DOC "The directory where GLFW/glfw3.h resides")
  find_library(glfw3_LIBRARIES NAMES glfw3dll GLFW PATHS $ENV{PROGRAMFILES}/GLFW/lib ${glfw3_DIR}/lib DOC "The GLFW library")
  include_directories(${glfw3_INCLUDE_DIRS})
else()	
  find_package(glfw3 REQUIRED)
  set(glfw3_LIBRARIES "glfw")
endif()

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

if(WIN32)
  find_path(ASSIMP_DIR Assimp $ENV{PROGRAMFILES}/Assimp/ DOC "The directory where Assimp resides")
  find_path(ASSIMP_INCLUDE_DIRS assimp/Importer.hpp $ENV{PROGRAMFILES}/Assimp/include ${ASSIMP_DIR}/include DOC "The directory where assimp/Importer.h resides")
  find_library(ASSIMP_LIBRARIES NAMES assimp Assimp PATHS $ENV{PROGRAMFILES}/Assimp/lib/x64 ${ASSIMP_DIR}/lib/x64 DOC "The assimp library")
  include_directories(${ASSIMP_INCLUDE_DIRS})
 else()
  find_package(ASSIMP REQUIRED)
  include_directories(${ASSIMP_INCLUDE_DIRS})
endif()

find_package(Threads REQUIRED )

#needed for the on-the-fly compilation of GLSL to SPV
link_directories(external)
set(GLSLANG_LIBS SPIRV SPVRemapper glslang OGLCompiler HLSL OSDependent SPIRV-Tools SPIRV-Tools-opt)

set(COMMON_LIBS ${glfw3_LIBRARIES} ${Vulkan_LIBRARIES} ${ASSIMP_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if(WIN32)
  set(GLM_ROOT_DIR "glm-0.9.8.3" CACHE STRING "The directory where glm is located")
  find_path(glm_INCLUDE_DIRS glm/glm.hpp PATHS $ENV{SYSTEMDRIVE}/${GLM_ROOT_DIR} $ENV{HOMEDRIVE}/${GLM_ROOT_DIR} $ENV{PROGRAMFILES}/${GLM_ROOT_DIR} DOC "The directory where glm.hpp resides")
  include_directories(${glm_INCLUDE_DIRS})
endif()

foreach(DEMO ${DEMOS})
  add_custom_target("compile_shader_${DEMO}" ALL "python" "compileshaders.py" "${DEMO}"
    WORKING_DIRECTORY "${RUN_DIR}/media/shaders/"
    COMMENT "Compiling shaders for ${DEMO}"
    VERBATIM
    )
  add_executable(${DEMO} src/${DEMO}/${DEMO}.cpp)
  target_link_libraries(${DEMO} ${COMMON_LIBS} ${GLSLANG_LIBS})
endforeach(DEMO)

include_directories(include)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DVK_DEBUG)
endif ()

if (WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /DVK_DEBUG")
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL GNU OR CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}         -Wall -march=native -O2 -flto")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -march=native -O2 -flto")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -Wall -march=native -O0 -flto")
endif()
