#include "util.hpp"
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
  auto egl_version = gladLoaderLoadEGL(nullptr);
  if (egl_version == 0) {
    fprintf(stderr, "failed to load EGL\n");
    return 1;
  }

  display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (!eglInitialize(display_, nullptr, nullptr)) {
    fprintf(stderr, "failed to egl initialize\n");
    return 1;
  }

  egl_version = gladLoaderLoadEGL(display_);
  if (egl_version == 0) {
    fprintf(stderr, "failed to reload EGL\n");
    return 1;
  }

  std::string egl_extensions_st = eglQueryString(display_, EGL_EXTENSIONS);
  if (egl_extensions_st.find("EGL_KHR_create_context") == std::string::npos) {
    fprintf(stderr, "EGL_KHR_create_context not found\n");
    return 1;
  }
  if (egl_extensions_st.find("EGL_KHR_surfaceless_context") ==
      std::string::npos) {
    fprintf(stderr, "EGL_KHR_surfaceless_context not found\n");
    return 1;
  }

  auto config_bit = es ? EGL_OPENGL_ES_BIT : EGL_OPENGL_BIT;
  static std::array<EGLint, 3> config_attribs = {EGL_RENDERABLE_TYPE,
                                                 config_bit, EGL_NONE};
  EGLConfig cfg;
  EGLint count;

  if (!eglChooseConfig(display_, config_attribs.data(), &cfg, 1, &count)) {
    fprintf(stderr, "eglChooseConfig failed\n");
    return 1;
  }

  auto api = es ? EGL_OPENGL_ES_API : EGL_OPENGL_API;
  if (!eglBindAPI(api)) {
    fprintf(stderr, "eglBindAPI failed\n");
    return 1;
  }

  EGLint major = es ? 3 : 4;
  EGLint minor = es ? 2 : 3;
  static std::array<EGLint, 5> attribs = {EGL_CONTEXT_MAJOR_VERSION, major,
                                          EGL_CONTEXT_MINOR_VERSION, minor,
                                          EGL_NONE};
  context_ = eglCreateContext(display_, cfg, EGL_NO_CONTEXT, attribs.data());
  if (context_ == EGL_NO_CONTEXT) {
    fprintf(stderr, "failed to create egl context\n");
    return 1;
  }

  if (!eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, context_)) {
    fprintf(stderr, "failed to make egl context current\n");
    return 1;
  }

  auto version = gladLoaderLoadGL();
  if (version == 0) {
    fprintf(stderr, "faiedl to load GL\n");
    return 1;
  }

  // printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
  // printf("OpenGL Vendor: %s\n", glGetString(GL_VENDOR));
  // printf("OpenGL Renderer: %s\n", glGetString(GL_RENDERER));

  init_ = true;
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
    gladLoaderUnloadGL();
    eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    // eglDestroyContext(display_, context_);
    // eglTerminate(display_);
    gladLoaderUnloadEGL();
    init_ = false;
  }
}
