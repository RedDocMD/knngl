cmake_minimum_required(VERSION 3.10)
project(knngl LANGUAGES CXX)

add_subdirectory(extern/pybind11)

set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

pybind11_add_module(knngl
	knn.cpp
	util.cpp)

target_link_libraries(knngl PRIVATE OpenGL::GL OpenGL::EGL GLEW::GLEW)
target_compile_options(knngl PRIVATE -Wall -pedantic -std=c++17)
