#include "util.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <type_traits>

namespace py = pybind11;

static GLuint make_texture(GLuint width, GLuint height, bool es) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  auto internal_format = es ? GL_R32F : GL_RG32F;
  glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, width, height);
  glBindTexture(GL_TEXTURE_2D, 0);
  return tex;
}

static GLuint vectors_to_texture(const double *data, size_t vec_dim,
                                 size_t vec_cnt, GLuint loc, bool es) {
  GLuint tex_width = es ? 2 * vec_dim : vec_dim;
  GLuint tex_height = vec_cnt;
  auto tex = make_texture(tex_width, tex_height, es);
  auto format = es ? GL_RED : GL_RG;
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_width, tex_height, format,
                  GL_FLOAT, data);
  glBindTexture(GL_TEXTURE_2D, 0);
  auto img_format = es ? GL_R32F : GL_RG32F;
  glBindImageTexture(loc, tex, 0, GL_FALSE, 0, GL_READ_ONLY, img_format);
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

static GLuint get_ssbo(void *data, size_t size, GLint bind_point) {
  GLuint ssbo;
  glGenBuffers(1, &ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
  glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind_point, ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  return ssbo;
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

const std::string knn_glsl_es = R"(
#version 320 es
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(r32f, binding = 0) uniform highp image2D data;
layout(r32f, binding = 1) uniform highp image2D queries;
layout(r32f, binding = 2) uniform highp image2D dist;

double join(in float lo, in float hi) {
	return double(lo) + double(hi);
}

vec2 split(in double a) {
	const double SPLITTER = double((1 << 29) + 1);
	double t = a * SPLITTER;
	double t_hi = t - (t - a);
	double t_lo = a - t_hi;
	return vec2(float(t_lo), float(t_hi));
}

void main() {
        int qidx = int(gl_GlobalInvocationID.x);
        int didx = int(gl_GlobalInvocationID.y);
	int dim = imageSize(data).x / 2;
	float sum = 0.0f;
	for (int i = 0; i < dim; i++) {
		float qlo = imageLoad(queries, ivec2(i, qidx)).x;
		float qhi = imageLoad(queries, ivec2(i + 1, qidx)).x;
		float dlo = imageLoad(data, ivec2(i, didx)).x;
		float dhi = imageLoad(data, ivec2(i + 1, didx)).x;
		double qd = join(qlo, qhi);
		double dd = join(dlo, dhi);
		float diff = abs(float(qd - dd));
		sum += diff * diff;
	}
	float val = sqrt(sum);
	vec4 pixel = vec4(val, 0, 0, 0);
	ivec2 coord = ivec2(didx, qidx);
	imageStore(dist, coord, pixel);
}
)";

const std::string knn_ssbo = R"(
#version 430

uniform int dim;
uniform int data_cnt;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer dataBuffer {
	double buf[];
} data;

layout(std430, binding = 1) buffer queriesBuffer {
	double buf[];
} queries;

layout(std430, binding = 2) buffer outBuffer {
	double buf[];
} dist;

void main() {
	int query_idx = int(gl_GlobalInvocationID.x);
	int data_idx = int(gl_GlobalInvocationID.y);
	double sum = 0.0;
	for (int i = 0; i < dim; i++) {
		double q = queries.buf[query_idx * dim + i];
		double d = data.buf[data_idx * dim + i];
		double diff = abs(q - d);
		sum += diff * diff;
	}
	double val = sqrt(sum);
	dist.buf[query_idx * data_cnt + data_idx] = val;
}
)";

template <typename T>
static void nearest_neighbours_intern(const T *dist,
                                      py::array_t<py::ssize_t> &neighbours,
                                      py::ssize_t queries_cnt,
                                      py::ssize_t data_cnt, int k) {
  for (py::ssize_t q = 0; q < queries_cnt; q++) {
    py::ssize_t last_min_idx = -1;
    T last_min = -1;
    for (int kval = 0; kval < k; kval++) {
      py::ssize_t min_idx = 0;
      auto min_dist = std::numeric_limits<T>().max();
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
}

template <typename T>
static void find_nearest_neighbours(py::array_t<py::ssize_t> &neighbours,
                                    py::ssize_t queries_cnt,
                                    py::ssize_t data_cnt, int k,
                                    GLuint dist_tex) {
  static_assert(
      !std::is_same_v<T, float> || !std::is_same_v<T, double>,
      "find_nearest_neighbours() must be called with float or double");
  constexpr bool is_double = std::is_same_v<T, double>;

  std::vector<T> dist(queries_cnt * data_cnt, -1.0);
  auto format = is_double ? GL_RG : GL_RED;
  glBindTexture(GL_TEXTURE_2D, dist_tex);
  glGetTexImage(GL_TEXTURE_2D, 0, format, GL_FLOAT, dist.data());
  handleGlError();
  glBindTexture(GL_TEXTURE_2D, 0);

  if constexpr (is_double)
    join_double(dist.data(), dist.size());

  nearest_neighbours_intern(dist.data(), neighbours, queries_cnt, data_cnt, k);
}

class Knn {
public:
  static Knn create(bool es) {
    auto ctx = GlContext::init(es);
    if (!ctx)
      throw std::runtime_error("Failed to create OpenGL context");
    auto &shader_src = es ? knn_glsl_es : knn_glsl;
    auto knn_shader = Shader::fromData(shader_src, GL_COMPUTE_SHADER);
    std::vector<Shader> knn_shaders;
    knn_shaders.push_back(std::move(knn_shader));
    auto knn_prog = Program::fromShaders(knn_shaders);

    auto knn_ssbo_shader = Shader::fromData(knn_ssbo, GL_COMPUTE_SHADER);
    std::vector<Shader> knn_ssbo_shaders;
    knn_ssbo_shaders.push_back(std::move(knn_ssbo_shader));
    auto knn_ssbo_prog = Program::fromShaders(knn_ssbo_shaders);

    return Knn(std::move(*ctx), std::move(knn_prog), std::move(knn_ssbo_prog),
               es);
  }

  Knn(const Knn &) = delete;
  Knn &operator=(const Knn &) = delete;

  Knn(Knn &&knn)
      : ctx_{std::move(knn.ctx_)}, knn_prog_{std::move(knn.knn_prog_)},
        knn_ssbo_prog_{std::move(knn.knn_ssbo_prog_)}, es_{std::move(knn.es_)} {
  }

  Knn &operator=(Knn &&knn) {
    if (&knn == this)
      return *this;
    ctx_ = std::move(knn.ctx_);
    knn_prog_ = std::move(knn.knn_prog_);
    knn_ssbo_prog_ = std::move(knn.knn_ssbo_prog_);
    es_ = std::move(knn.es_);
    return *this;
  }

  py::array_t<py::ssize_t>
  knn_with_ssbo(py::array_t<double, py::array::c_style> data,
                py::array_t<double, py::array::c_style> queries, int k) {
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

    auto program = knn_ssbo_prog_.program();
    glUseProgram(program);

    auto dim_loc = get_uniform_location(program, "dim");
    auto data_cnt_loc = get_uniform_location(program, "data_cnt");

    auto data_cnt = data_shape[0];
    auto queries_cnt = queries_shape[0];
    auto dim = data_shape[1];

    glUniform1i(dim_loc, dim);
    glUniform1i(data_cnt_loc, data_cnt);

    GLint dataBufLoc = 0;
    GLint queriesBufLoc = 1;
    GLint distBufferLoc = 2;

    auto data_ssbo = get_ssbo(data_ptr, data_size * sizeof(double), dataBufLoc);
    auto queries_ssbo =
        get_ssbo(queries_ptr, queries_size * sizeof(double), queriesBufLoc);
    auto dist_ssbo = get_ssbo(nullptr, data_cnt * queries_cnt * sizeof(double),
                              distBufferLoc);

    glDispatchCompute(queries_cnt, data_cnt, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, dist_ssbo);
    auto *dist = reinterpret_cast<double *>(glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, data_cnt * queries_cnt * sizeof(double),
        GL_MAP_READ_BIT));
    handleGlError();

    py::array_t<py::ssize_t> neighbours(
        {static_cast<py::ssize_t>(queries_cnt), static_cast<py::ssize_t>(k)});
    nearest_neighbours_intern(dist, neighbours, queries_cnt, data_cnt, k);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    std::array<GLuint, 3> bufs{data_ssbo, queries_ssbo, dist_ssbo};
    glDeleteBuffers(bufs.size(), bufs.data());
    glUseProgram(0);

    return neighbours;
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

    auto data_tex = vectors_to_texture(data_ptr, dim, data_cnt, data_loc, es_);
    auto query_tex =
        vectors_to_texture(queries_ptr, dim, queries_cnt, query_loc, es_);
    auto dist_tex = make_texture(data_cnt, queries_cnt, es_);
    auto dist_format = es_ ? GL_R32F : GL_RG32F;
    glBindImageTexture(dist_loc, dist_tex, 0, GL_FALSE, 0, GL_READ_WRITE,
                       dist_format);

    glDispatchCompute(queries_cnt, data_cnt, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    py::array_t<py::ssize_t> neighbours(
        {static_cast<py::ssize_t>(queries_cnt), static_cast<py::ssize_t>(k)});
    if (es_) {
      find_nearest_neighbours<float>(neighbours, queries_cnt, data_cnt, k,
                                     dist_tex);
    } else {
      find_nearest_neighbours<double>(neighbours, queries_cnt, data_cnt, k,
                                      dist_tex);
    }

    join_double(data_ptr, data_size);
    join_double(queries_ptr, queries_size);

    glUseProgram(0);
    std::array<GLuint, 3> textures{data_tex, query_tex, dist_tex};
    glDeleteTextures(textures.size(), textures.data());

    return neighbours;
  }

private:
  Knn(GlContext ctx, Program knn_prog, Program knn_ssbo_prog, bool es)
      : ctx_{std::move(ctx)}, knn_prog_{std::move(knn_prog)},
        knn_ssbo_prog_{std::move(knn_ssbo_prog)}, es_{es} {}

  GlContext ctx_;
  Program knn_prog_;
  Program knn_ssbo_prog_;
  bool es_;
};

PYBIND11_MODULE(knngl, m) {
  m.doc() = "KNN implementation using OpenGL";

  py::class_<Knn>(m, "Knn")
      .def(py::init(&Knn::create), py::kw_only(), py::arg("es"))
      .def("knn", &Knn::knn)
      .def("knn_with_ssbo", &Knn::knn_with_ssbo);
}
