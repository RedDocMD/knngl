cmake_minimum_required(VERSION 3.10)
project(knngl LANGUAGES C CXX)

add_subdirectory(extern/pybind11)

set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

pybind11_add_module(knngl
	knn.cpp
	util.cpp
	glad/src/gl.c
	glad/src/egl.c
)

target_include_directories(knngl PRIVATE glad/include)
target_link_libraries(knngl PRIVATE OpenGL::GL OpenGL::EGL GLEW::GLEW)

target_compile_options(knngl PRIVATE 
	$<$<COMPILE_LANGUAGE:CXX,C>:-Wall>
	$<$<COMPILE_LANGUAGE:CXX>:-pedantic>
	$<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
	$<$<COMPILE_LANGUAGE:C>:-std=c11>
)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/python/test.py
	${CMAKE_CURRENT_BINARY_DIR}/test.py
	COPYONLY
)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/python/bench_adult.py
	${CMAKE_CURRENT_BINARY_DIR}/bench_adult.py
	COPYONLY
)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/python/bench_adult_rpi.py
	${CMAKE_CURRENT_BINARY_DIR}/bench_adult_rpi.py
	COPYONLY
)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/python/bench_rtiot.py
	${CMAKE_CURRENT_BINARY_DIR}/bench_rtiot.py
	COPYONLY
)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/python/bench_single_time.py
	${CMAKE_CURRENT_BINARY_DIR}/bench_single_time.py
	COPYONLY
)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/python/test_adult.py
	${CMAKE_CURRENT_BINARY_DIR}/test_adult.py
	COPYONLY
)
