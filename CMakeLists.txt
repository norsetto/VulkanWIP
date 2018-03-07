cmake_minimum_required (VERSION 3.9)

project (vulkan_demos VERSION 0.0.0 DESCRIPTION "Various Vulkan (tm) demos")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
  
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(RUN_DIR ${PROJECT_SOURCE_DIR}/bin)

configure_file("${PROJECT_SOURCE_DIR}/config.h.in" "${PROJECT_BINARY_DIR}/config.h")
include_directories("${PROJECT_BINARY_DIR}")

set(DEMOS
  basic
  model
  )

# Libraries
find_package(PkgConfig REQUIRED)

pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

find_package (Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

set(COMMON_LIBS ${GLFW_LIBRARIES} ${Vulkan_LIBRARIES})

foreach(DEMO ${DEMOS})
  add_custom_target("compile_shader_${DEMO}" ALL "python" "compileshaders.py" "${DEMO}"
    WORKING_DIRECTORY "${RUN_DIR}/media/shaders/"
    COMMENT "Compiling shaders for ${DEMO}"
    VERBATIM
    )
  add_executable(${DEMO} src/${DEMO}/${DEMO}.cpp)
  target_link_libraries(${DEMO} ${COMMON_LIBS})
endforeach(DEMO)

include_directories(include)

if (CMAKE_CXX_COMPILER_ID STREQUAL GNU OR CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -std=c++14 -march=native -O2 -flto")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -Wall -std=c++14 -march=native -O0 -flto -DVK_DEBUG")
endif()
