#pragma once

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/glew.h>
#include <algorithm>
#include <optional>
#include <string>
#include <vector>

std::string readFile(const std::string &name);
void printShaderInfoLog(GLuint shader_index);
GLuint loadShader(const std::string &name, GLuint shader_type);
GLuint loadShaderFromData(const std::string &data, GLuint shader_type);
void printProgramInfoLog(GLuint program);
GLuint createProgram(const std::vector<GLuint> &shaders);
void handleGlError();

class GlContext {
public:
  static std::optional<GlContext> init(bool es);
  ~GlContext();
  GlContext(const GlContext &) = delete;
  GlContext &operator=(const GlContext &) = delete;

  GlContext(GlContext &&ctx)
      : init_{ctx.init_}, context_{ctx.context_}, display_{ctx.display_} {
    ctx.init_ = false;
  }

  GlContext &operator=(GlContext &&ctx) {
    if (&ctx == this)
      return *this;
    if (init_)
      this->~GlContext();

    init_ = ctx.init_;
    context_ = ctx.context_;
    display_ = ctx.display_;
    ctx.init_ = false;

    return *this;
  }

private:
  bool init_{false};
  EGLContext context_{};
  EGLDisplay display_{};

  GlContext() = default;
  int glInit(bool es);
};

class Shader {
public:
  static Shader fromFile(const std::string &name, GLuint shader_type) {
    return Shader(loadShader(name, shader_type));
  }

  static Shader fromData(const std::string &data, GLuint shader_type) {
    return Shader(loadShaderFromData(data, shader_type));
  }

  ~Shader() { glDeleteShader(shader_); }

  Shader(const Shader &) = delete;
  Shader &operator=(const Shader &) = delete;

  Shader(Shader &&sh) : shader_{sh.shader_} { sh.shader_ = 0; }

  Shader &operator=(Shader &&sh) {
    if (&sh == this)
      return *this;
    this->~Shader();
    shader_ = sh.shader_;
    sh.shader_ = 0;
    return *this;
  }

  GLuint shader() const { return shader_; }

private:
  Shader(GLuint shader) : shader_{shader} {}

  GLuint shader_;
};

class Program {
public:
  static Program fromShaders(const std::vector<Shader> &shaders) {
    std::vector<GLuint> shader_ids(shaders.size());
    std::transform(shaders.begin(), shaders.end(), shader_ids.begin(),
                   [](const auto &sh) { return sh.shader(); });
    return Program(createProgram(shader_ids));
  }

  ~Program() { glDeleteProgram(program_); }

  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;

  Program(Program &&pg) : program_{pg.program_} { pg.program_ = 0; }

  Program &operator=(Program &&pg) {
    if (&pg == this)
      return *this;
    this->~Program();
    program_ = pg.program_;
    pg.program_ = 0;
    return *this;
  }

  GLuint program() const { return program_; }

private:
  Program(GLuint program) : program_{program} {}

  GLuint program_;
};
