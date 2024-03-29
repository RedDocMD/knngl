#include "util.hpp"
#include "GLFW/glfw3.h"
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>

std::string readFile(const std::string &name) {
  std::ifstream infile(name, std::ios::in | std::ios::ate);
  if (infile.fail())
    throw std::runtime_error("File opening failed: " + name);
  size_t size = infile.tellg();
  std::string inp(size, '\0');
  infile.seekg(0);
  if (!infile.read(inp.data(), size))
    throw std::runtime_error("File reading failed: " + name);
  return inp;
}

void printShaderInfoLog(GLuint shader_index) {
  GLint size = 0;
  glGetShaderiv(shader_index, GL_INFO_LOG_LENGTH, &size);
  std::string info(size, '\0');
  int actual_length;
  glGetShaderInfoLog(shader_index, size, &actual_length, info.data());
  printf("shader info log for GL index %u:\n%s\n", shader_index, info.c_str());
}

GLuint loadShader(const std::string &name, GLuint shader_type) {
  auto data = readFile(name);
  return loadShaderFromData(data, shader_type);
}

GLuint loadShaderFromData(const std::string &data, GLuint shader_type) {
  auto shader_idx = glCreateShader(shader_type);
  if (shader_idx == 0)
    throw std::runtime_error("Failed to create shader");
  std::array<const char *, 1> dataArr{data.data()};
  glShaderSource(shader_idx, 1, dataArr.data(), NULL);
  glCompileShader(shader_idx);

  int params = -1;
  glGetShaderiv(shader_idx, GL_COMPILE_STATUS, &params);
  if (GL_TRUE != params) {
    fprintf(stderr, "ERROR: GL shader index %i did not compile\n", shader_idx);
    printShaderInfoLog(shader_idx);
    throw std::runtime_error("shader compilation failed");
  }

  return shader_idx;
}

void printProgramInfoLog(GLuint program) {
  GLint size;
  glGetShaderiv(program, GL_INFO_LOG_LENGTH, &size);
  std::string info(size, '\0');
  int actual_length;
  glGetProgramInfoLog(program, size, &actual_length, info.data());
  printf("program info log for GL index %u:\n%s", program, info.c_str());
}

GLuint createProgram(const std::vector<GLuint> &shaders) {
  GLuint program = glCreateProgram();
  for (auto shader : shaders)
    glAttachShader(program, shader);
  glLinkProgram(program);

  int params = -1;
  glGetProgramiv(program, GL_LINK_STATUS, &params);
  if (GL_TRUE != params) {
    fprintf(stderr, "ERROR: could not link shader programm GL index %u\n",
            program);
    printProgramInfoLog(program);
    throw std::runtime_error("program creation failed");
  }

  for (auto shader : shaders)
    glDetachShader(program, shader);
  return program;
}

void handleGlError() {
  GLenum err;
  bool has_error = false;
  while ((err = glGetError()) != GL_NO_ERROR) {
    has_error = true;
    switch (err) {
    case GL_INVALID_ENUM:
      std::cerr << "GL_INVALID_ENUM: An unacceptable value is specified "
                   "for an enumerated "
                   "argument. The offending command is ignored and has "
                   "no other side effect than to set the error flag\n";
      break;
    case GL_INVALID_VALUE:
      std::cerr << "GL_INVALID_VALUE: A numeric argument is out of "
                   "range. The offending command is ignored and has no "
                   "other side effect than to set the error flag\n";
      break;
    case GL_INVALID_OPERATION:
      std::cerr << "GL_INVALID_OPERATION: The specified operation is not "
                   "allowed in the current state. The offending command "
                   "is ignored and has no other side effect than to set "
                   "the error flag\n";
      break;
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      std::cerr << "GL_INVALID_FRAMEBUFFER_OPERATION: The framebuffer object "
                   "is not complete. The offending command is ignored and has "
                   "no other side effect than to set the error flag\n";
      break;
    case GL_OUT_OF_MEMORY:
      std::cerr << "GL_OUT_OF_MEMORY: The framebuffer object is not "
                   "complete. The offending command is ignored and has "
                   "no other side effect than to set the error flag\n";
      break;
    case GL_STACK_UNDERFLOW:
      std::cerr << "GL_STACK_UNDERFLOW: An attempt has been made to "
                   "perform an operation that would cause an internal "
                   "stack to underflow\n";
      break;
    case GL_STACK_OVERFLOW:
      std::cerr << "GL_STACK_OVERFLOW: An attempt has been made to perform an "
                   "operation that would cause an internal stack to overflow\n";
      break;
    default:
      std::cerr << "Unknown error\n";
    }
    std::flush(std::cerr);
  }
  if (has_error)
    throw std::runtime_error("OpenGl error");
}

int GlContext::glInit(bool es) {
  if (!glfwInit()) {
    fprintf(stderr, "ERROR: could not start GLFW3\n");
    return 1;
  }

  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  GLFWwindow *window = glfwCreateWindow(1, 1, "", nullptr, nullptr);
  if (!window) {
    fprintf(stderr, "ERROR: could not open window with GLFW3\n");
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);

  printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
  printf("OpenGL Vendor: %s\n", glGetString(GL_VENDOR));
  printf("OpenGL Renderer: %s\n", glGetString(GL_RENDERER));

  glewExperimental = GL_TRUE;
  glewInit();

  return 0;
}

std::optional<GlContext> GlContext::init(bool es) {
  GlContext ctx;
  if (ctx.glInit(es))
    return {};
  return std::move(ctx);
}

GlContext::~GlContext() {
  if (init_) {
    eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(display_, context_);
    eglTerminate(display_);
  }
}
