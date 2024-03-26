#include "util.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

static GLuint make_texture(GLuint width, GLuint height) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTextureStorage2D(tex, 1, GL_RG32F, width, height);
  glBindTexture(GL_TEXTURE_2D, 0);
  return tex;
}

static GLuint vectors_to_texture(const double *data, size_t vec_dim,
                                 size_t vec_cnt, GLuint loc) {
  GLuint tex_width = vec_dim;
  GLuint tex_height = vec_cnt;
  auto tex = make_texture(vec_dim, vec_cnt);
  glTextureSubImage2D(tex, 0, 0, 0, tex_width, tex_height, GL_RG, GL_FLOAT,
                      data);
  glBindImageTexture(loc, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
  return tex;
}

static void split_double(double *data, size_t len) {
  constexpr double splitter = (1 << 29) + 1;
  for (size_t i = 0; i < len; i++) {
    double *a = data + i;
    double t = *a * splitter;
    float t_hi = t - (t - *a);
    float t_lo = *a - t_hi;
    memcpy(a, &t_lo, sizeof(t_lo));
    memcpy(reinterpret_cast<char *>(a) + sizeof(float), &t_hi, sizeof(t_hi));
  }
}

static void join_double(double *data, size_t len) {
  for (size_t i = 0; i < len; i++) {
    double *a = data + i;
    float t_lo, t_hi;
    memcpy(&t_lo, a, sizeof(t_lo));
    memcpy(&t_hi, reinterpret_cast<char *>(a) + sizeof(float), sizeof(t_hi));
    *a = static_cast<double>(t_lo) + static_cast<double>(t_hi);
  }
}

static inline __attribute__((always_inline)) GLint
get_uniform_location(GLint program, const std::string &name) {
  auto loc = glGetUniformLocation(program, name.c_str());
  if (loc == -1)
    throw std::runtime_error("failed to find location of uniform: " + name);
  handleGlError();
  return loc;
}

const std::string knn_glsl = R"(
#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(rg32f, binding = 0) uniform image2D data;
layout(rg32f, binding = 1) uniform image2D queries;
layout(rg32f, binding = 2) uniform image2D dist;

double join(in vec2 fv) {
	return double(fv.x) + double(fv.y);
}

vec2 split(in double a) {
	const double SPLITTER = (1 << 29) + 1;
	double t = a * SPLITTER;
	double t_hi = t - (t - a);
	double t_lo = a - t_hi;
	return vec2(float(t_lo), float(t_hi));
}

void main() {
        uint qidx = gl_GlobalInvocationID.x;
        uint didx = gl_GlobalInvocationID.y;
	int dim = imageSize(data).x;
	double sum = 0;
	for (int i = 0; i < dim; i++) {
		vec2 qv = imageLoad(queries, ivec2(i, qidx)).xy;
		vec2 dv = imageLoad(data, ivec2(i, didx)).xy;
		double qvd = join(qv);
		double dvd = join(dv);
		double diff = abs(qvd - dvd);
		sum += diff * diff;
	}
	double val = sqrt(sum);
	vec2 val_vec = split(val);
	vec4 pixel = vec4(val_vec.x, val_vec.y, 0, 0);
	ivec2 coord = ivec2(gl_GlobalInvocationID.yx);
	imageStore(dist, coord, pixel);
}
)";

class Knn {
public:
  static Knn create() {
    auto ctx = GlContext::init();
    if (!ctx)
      throw std::runtime_error("Failed to create OpenGL context");
    auto knn_shader = Shader::fromData(knn_glsl, GL_COMPUTE_SHADER);
    std::vector<Shader> knn_shaders;
    knn_shaders.push_back(std::move(knn_shader));
    auto knn_prog = Program::fromShaders(knn_shaders);
    return Knn(std::move(*ctx), std::move(knn_prog));
  }

  Knn(const Knn &) = delete;
  Knn &operator=(const Knn &) = delete;

  Knn(Knn &&knn)
      : ctx_{std::move(knn.ctx_)}, knn_prog_{std::move(knn.knn_prog_)} {}

  Knn &operator=(Knn &&knn) {
    if (&knn == this)
      return *this;
    ctx_ = std::move(knn.ctx_);
    knn_prog_ = std::move(knn.knn_prog_);
    return *this;
  }

  py::array_t<py::ssize_t> knn(py::array_t<double, py::array::c_style> data,
                               py::array_t<double, py::array::c_style> queries,
                               int k) {
    if (data.ndim() != 2)
      throw std::runtime_error("data doesn't have 2 dimensions");
    if (queries.ndim() != 2)
      throw std::runtime_error("data doesn't have 2 dimensions");

    auto *data_shape = data.shape();
    auto *queries_shape = queries.shape();
    if (data_shape[1] != queries_shape[1])
      throw std::runtime_error(
          "data and query vectors have different dimensions");
    if (data_shape[0] == 0)
      throw std::runtime_error("empty data array");
    if (queries_shape[0] == 0)
      throw std::runtime_error("empty queries array");

    auto data_arr = data.mutable_unchecked<2>();
    auto queries_arr = queries.mutable_unchecked<2>();

    auto *data_ptr = data_arr.mutable_data(0, 0);
    auto *queries_ptr = queries_arr.mutable_data(0, 0);
    auto data_size = data_arr.size();
    auto queries_size = queries_arr.size();

    split_double(data_ptr, data_size);
    split_double(queries_ptr, queries_size);

    auto program = knn_prog_.program();
    glUseProgram(program);

    auto data_loc = get_uniform_location(program, "data");
    auto query_loc = get_uniform_location(program, "queries");
    auto dist_loc = get_uniform_location(program, "dist");

    auto data_cnt = data_shape[0];
    auto queries_cnt = queries_shape[0];
    auto dim = data_shape[1];

    auto data_tex = vectors_to_texture(data_ptr, dim, data_cnt, data_loc);
    auto query_tex =
        vectors_to_texture(queries_ptr, dim, queries_cnt, query_loc);
    auto dist_tex = make_texture(data_cnt, queries_cnt);
    glBindImageTexture(dist_loc, dist_tex, 0, GL_FALSE, 0, GL_READ_WRITE,
                       GL_RG32F);

    glDispatchCompute(queries_cnt, data_cnt, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    std::vector<double> dist(queries_cnt * data_cnt, -1.0f);
    glGetTextureImage(dist_tex, 0, GL_RG, GL_FLOAT,
                      dist.size() * sizeof(double), dist.data());
    join_double(dist.data(), dist.size());
    join_double(data_ptr, data_size);
    join_double(queries_ptr, queries_size);

    glUseProgram(0);
    std::array<GLuint, 3> textures{data_tex, query_tex, dist_tex};
    glDeleteTextures(3, textures.data());

    py::array_t<py::ssize_t> neighbours(
        {static_cast<py::ssize_t>(queries_cnt), static_cast<py::ssize_t>(k)});
    for (py::ssize_t q = 0; q < queries_cnt; q++) {
      py::ssize_t last_min_idx = -1;
      double last_min = -1;
      for (int kval = 0; kval < k; kval++) {
        py::ssize_t min_idx = 0;
        double min_dist = std::numeric_limits<double>().max();
        for (py::ssize_t i = 0; i < data_cnt; i++) {
          auto d = dist[q * data_cnt + i];
          if (d < last_min)
            continue;
          if (d == last_min && i == last_min_idx)
            continue;
          if (d < min_dist) {
            min_dist = d;
            min_idx = i;
          }
        }
        *neighbours.mutable_data(q, kval) = min_idx;
        last_min = min_dist;
        last_min_idx = min_idx;
      }
    }

    return neighbours;
  }

private:
  Knn(GlContext ctx, Program knn_prog)
      : ctx_{std::move(ctx)}, knn_prog_{std::move(knn_prog)} {}

  GlContext ctx_;
  Program knn_prog_;
};

PYBIND11_MODULE(knngl, m) {
  m.doc() = "KNN implementation using OpenGL";

  py::class_<Knn>(m, "Knn").def(py::init(&Knn::create)).def("knn", &Knn::knn);
}
