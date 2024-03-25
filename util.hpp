#pragma once

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/glew.h>
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
  static std::optional<GlContext> init();
  ~GlContext();
  GlContext(const GlContext &) = delete;
  GlContext &operator=(const GlContext &) = delete;

  GlContext(GlContext &&ctx)
      : init_{ctx.init_}, context_{ctx.context_}, display_{ctx.display_} {
    ctx.init_ = false;
  }

  GlContext &operator=(GlContext &&ctx) {
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
  int glInit();
};
