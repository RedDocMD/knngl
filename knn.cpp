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

#define MEASURE_TIME 0

#if MEASURE_TIME
#include <chrono>
namespace chrono = std::chrono;
#endif

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

static GLuint get_ssbo(void *data, size_t size, GLint bind_point, GLint flags) {
  GLuint ssbo;
  glGenBuffers(1, &ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
  glBufferStorage(GL_SHADER_STORAGE_BUFFER, size, data, flags);
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
uniform int queries_off;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer dataBuffer {
	double buf[];
} data;

layout(std430, binding = 1) buffer queriesBuffer {
	double buf[];
} queries;

layout(std430, binding = 2) buffer outBuffer {
	float buf[];
} dist;

void main() {
	int query_idx = int(gl_GlobalInvocationID.x) + queries_off;
	int data_idx = int(gl_GlobalInvocationID.y);
	float sum = 0.0f;
	for (int i = 0; i < dim; i++) {
		float q = float(queries.buf[query_idx * dim + i]);
		float d = float(data.buf[data_idx * dim + i]);
		float diff = abs(q - d);
		sum += diff * diff;
	}
	float val = sqrt(sum);
	dist.buf[(query_idx - queries_off) * data_cnt + data_idx] = val;
}
)";

const std::string knn_sort = R"(
#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uniform int k;
uniform int data_cnt;

layout(std430, binding = 3) buffer outBuffer {
  int buf[];
} neigh;

layout(std430, binding = 2) buffer inBuffer {
  float buf[];
} dist;

void main() {
  int query_idx = int(gl_GlobalInvocationID.x);  
  float last_min = -1.0;
  for (int kval = 0; kval < k; kval++) {
    int min_idx = 0;
    float min_dist = 1.0e38;
    for (int i = 0; i < data_cnt; i++) {
      float d = dist.buf[query_idx * data_cnt + i];
      if (d < last_min)
	continue;
      if (d == last_min) {
	bool is_identical = false;
	for (int p = 0; p < kval; p++) {
	  if (neigh.buf[query_idx * k + p] == i) {
	    is_identical = true;
	    break;
	  }
	}
	if (is_identical)
	  continue;
      }
      if (d < min_dist) {
	min_dist = d;
	min_idx = i;
      }
    }
    neigh.buf[query_idx * k + kval] = min_idx;
    last_min = min_dist;
  }
}
)";

template <typename T>
static void
nearest_neighbours_intern(const T *dist, py::array_t<py::ssize_t> &neighbours,
                          py::ssize_t queries_cnt, py::ssize_t data_cnt, int k,
                          size_t queries_off = 0) {
  std::vector<ssize_t> indices(k);
  for (py::ssize_t q = 0; q < queries_cnt; q++) {
    T last_min = -1;
    for (int kval = 0; kval < k; kval++) {
      py::ssize_t min_idx = 0;
      auto min_dist = std::numeric_limits<T>().max();
      for (py::ssize_t i = 0; i < data_cnt; i++) {
        auto d = dist[q * data_cnt + i];
        if (d < last_min)
          continue;
        if (d == last_min) {
          bool is_identical = false;
          for (int p = 0; p < kval; p++) {
            if (indices[p] == i) {
              is_identical = true;
              break;
            }
          }
          if (is_identical)
            continue;
        }
        if (d < min_dist) {
          min_dist = d;
          min_idx = i;
        }
      }
      indices[kval] = min_idx;
      *neighbours.mutable_data(q + queries_off, kval) = min_idx;
      last_min = min_dist;
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

    auto knn_sort_shader = Shader::fromData(knn_sort, GL_COMPUTE_SHADER);
    std::vector<Shader> knn_sort_shaders;
    knn_sort_shaders.push_back(std::move(knn_sort_shader));
    auto knn_sort_prog = Program::fromShaders(knn_sort_shaders);

    return Knn(std::move(*ctx), std::move(knn_prog), std::move(knn_ssbo_prog),
               std::move(knn_sort_prog), es);
  }

  Knn(const Knn &) = delete;
  Knn &operator=(const Knn &) = delete;

  void destroy() { this->~Knn(); }

  Knn(Knn &&knn)
      : ctx_{std::move(knn.ctx_)}, knn_prog_{std::move(knn.knn_prog_)},
        knn_ssbo_prog_{std::move(knn.knn_ssbo_prog_)},
        knn_sort_prog_{std::move(knn.knn_sort_prog_)}, es_{std::move(knn.es_)} {
  }

  Knn &operator=(Knn &&knn) {
    if (&knn == this)
      return *this;
    ctx_ = std::move(knn.ctx_);
    knn_prog_ = std::move(knn.knn_prog_);
    knn_ssbo_prog_ = std::move(knn.knn_ssbo_prog_);
    knn_sort_prog_ = std::move(knn.knn_sort_prog_);
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

    auto data_cnt = data_shape[0];
    auto queries_cnt = queries_shape[0];
    auto dim = data_shape[1];

    constexpr GLint data_buf_loc = 0;
    constexpr GLint queries_buf_loc = 1;
    constexpr GLint dist_buf_loc = 2;
    constexpr GLint neigh_buf_loc = 3;

    GLuint max_ssbo_bytes = 0;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,
                  reinterpret_cast<GLint *>(&max_ssbo_bytes));

    if (data_size * sizeof(double) > max_ssbo_bytes)
      throw std::runtime_error("cannot fit dataset in GPU memory");
    auto data_ssbo =
        get_ssbo(data_ptr, data_size * sizeof(double), data_buf_loc, 0);

    if (queries_size * sizeof(double) > max_ssbo_bytes)
      throw std::runtime_error("cannot fit queries in GPU memory");
    auto queries_ssbo = get_ssbo(queries_ptr, queries_size * sizeof(double),
                                 queries_buf_loc, 0);

    constexpr float query_cutoff = 0.9f;
    size_t max_query_cnt =
        query_cutoff * max_ssbo_bytes / (data_cnt * sizeof(float));
    auto dist_size = max_query_cnt * data_cnt * sizeof(float);
    auto dist_ssbo =
        get_ssbo(nullptr, dist_size, dist_buf_loc,
                 GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
    auto neigh_ssbo =
        get_ssbo(nullptr, max_query_cnt * k * sizeof(int), neigh_buf_loc,
                 GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT);

    auto program = knn_ssbo_prog_.program();
    auto sort_program = knn_sort_prog_.program();

    auto dim_loc = get_uniform_location(program, "dim");
    auto data_cnt_loc = get_uniform_location(program, "data_cnt");
    auto queries_off_loc = get_uniform_location(program, "queries_off");
    auto k_loc = get_uniform_location(sort_program, "k");
    auto sort_data_cnt_loc = get_uniform_location(sort_program, "data_cnt");

    glUseProgram(program);
    glUniform1i(dim_loc, dim);
    glUniform1i(data_cnt_loc, data_cnt);

    glUseProgram(sort_program);
    glUniform1i(k_loc, k);
    glUniform1i(sort_data_cnt_loc, data_cnt);

    auto queries_left = static_cast<size_t>(queries_cnt);
    size_t queries_off = 0;
    py::array_t<py::ssize_t> neighbours(
        {static_cast<py::ssize_t>(queries_cnt), static_cast<py::ssize_t>(k)});

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, neigh_ssbo);
    auto *neigh_tmp = reinterpret_cast<int *>(glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, max_query_cnt * k * sizeof(int),
        GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT));

    while (queries_left > 0) {
      auto curr_query_cnt = std::min(max_query_cnt, queries_left);

      glUseProgram(program);
      glUniform1i(queries_off_loc, queries_off);
#if MEASURE_TIME
      auto begin_time = chrono::steady_clock::now();
#endif
      glDispatchCompute(queries_cnt, data_cnt, 1);
#if MEASURE_TIME
      glFinish();
      auto end_time = chrono::steady_clock::now();
      auto diff =
          chrono::duration_cast<chrono::milliseconds>(end_time - begin_time)
              .count();
      std::cout << "Time = " << diff << std::endl;
#endif

      glUseProgram(sort_program);
      glDispatchCompute(curr_query_cnt, 1, 1);
      glFinish();
      handleGlError();

      for (size_t q = queries_off; q < queries_off + curr_query_cnt; q++) {
        for (int kval = 0; kval < k; kval++) {
          *neighbours.mutable_data(q, kval) =
              neigh_tmp[(q - queries_off) * k + kval];
        }
      }

      queries_left -= curr_query_cnt;
      queries_off += curr_query_cnt;
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

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
  Knn(GlContext ctx, Program knn_prog, Program knn_ssbo_prog,
      Program knn_sort_prog, bool es)
      : ctx_{std::move(ctx)}, knn_prog_{std::move(knn_prog)},
        knn_ssbo_prog_{std::move(knn_ssbo_prog)},
        knn_sort_prog_{std::move(knn_sort_prog)}, es_{es} {}

  GlContext ctx_;
  Program knn_prog_;
  Program knn_ssbo_prog_;
  Program knn_sort_prog_;
  bool es_;
};

PYBIND11_MODULE(knngl, m) {
  m.doc() = "KNN implementation using OpenGL";

  py::class_<Knn>(m, "Knn")
      .def(py::init(&Knn::create), py::kw_only(), py::arg("es"))
      .def("knn", &Knn::knn)
      .def("knn_with_ssbo", &Knn::knn_with_ssbo)
      .def("destroy", &Knn::destroy);
}
