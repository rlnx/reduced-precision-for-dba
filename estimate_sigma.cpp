#include <cmath>
#include <tuple>
#include <memory>
#include <random>
#include <iomanip>
#include <fstream>
#include <string>
#include <iostream>
#include <atomic>
#include <thread>
#include <omp.h>

#include "floats.hpp"

using std::size_t;

class Shape {
public:
    Shape() : Shape(0, 0) {}

    Shape(const std::array<size_t, 2>& s)
        : s_(s) {}

    Shape(size_t row_count, size_t column_count)
        : s_({ row_count, column_count }) {}

    size_t operator[](size_t idx) const {
        return s_[idx];
    }

    size_t count() const {
        return s_[0] * s_[1];
    }

private:
    std::array<size_t, 2> s_;
};

inline bool operator==(const Shape& x, const Shape& y) {
    return x[0] == y[0] && x[1] == y[1];
}


inline bool operator!=(const Shape& x, const Shape& y) {
    return !(x == y);
}

template <typename T>
class Array {
public:
    Array() {}

    explicit Array(const std::shared_ptr<T> data, const Shape& shape)
        : data_(data),
          shape_(shape) {}

    T* data() const {
        return data_.get();
    }

    T* row_data(size_t row_idx) const {
        return &data()[row_idx * shape_[1]];
    }

    const Shape& shape() const {
        return shape_;
    }

    size_t row_count() const {
        return shape_[0];
    }

    size_t column_count() const {
        return shape_[1];
    }

    size_t count() const {
        return row_count() * column_count();
    }

private:
    std::shared_ptr<T> data_;
    Shape shape_;
};

template <typename T>
inline Array<T> make_empty(const Shape& shape) {
    T* ptr = new T[shape.count()];
    const auto data = std::shared_ptr<T>(ptr, std::default_delete<T[]>{});
    return Array<T>{data, shape};
}

template <typename T>
inline Array<T> make_full(const Shape& shape, T value) {
    auto a = make_empty<T>(shape);
    T* ptr = a.data();
    for (size_t i = 0; i < shape.count(); i++) {
        ptr[i] = value;
    }
    return a;
}

template <typename T>
inline Array<T> make_zeros(const Shape& shape) {
    return make_full<T>(shape, T(0));
}

template <typename T, typename Op>
inline Array<T> apply_binary_operation(const Array<T>& x, const Array<T>& y, Op&& op) {
    if (x.shape() != y.shape()) {
        throw std::invalid_argument{"Array shapes do not match"};
    }

    auto z = make_empty<T>(x.shape());

    const T* x_ptr = x.data();
    const T* y_ptr = y.data();
    T* z_ptr = z.data();

    for (size_t i = 0; i < x.count(); i++) {
        z_ptr[i] = op(x_ptr[i], y_ptr[i]);
    }

    return z;
}

template <typename T>
inline Array<T> operator+(const Array<T>& x, const Array<T>& y) {
    return apply_binary_operation<T>(x, y, std::plus<T>{});
}

template <typename T>
inline Array<T> operator-(const Array<T>& x, const Array<T>& y) {
    return apply_binary_operation<T>(x, y, std::minus<T>{});
}

template <typename T>
inline Array<T> operator*(const Array<T>& x, const Array<T>& y) {
    return apply_binary_operation<T>(x, y, std::multiplies<T>{});
}

template <typename T>
inline Array<T> operator/(const Array<T>& x, const Array<T>& y) {
    return apply_binary_operation<T>(x, y, std::divides<T>{});
}

template <typename T>
inline Array<T> uniform(std::mt19937& rng, T low, T high, const Shape& shape) {
    std::uniform_real_distribution<T> distr(low, high);
    auto a = make_empty<T>(shape);
    T* data = a.data();
    for (size_t i = 0; i < shape.count(); i++) {
        data[i] = distr(rng);
    }
    return a;
}

template <typename T>
inline Array<T> transpose(const Array<T>& x) {
    auto y = make_empty<T>({ x.column_count(), x.row_count() });
    T* x_ptr = x.data();
    T* y_ptr = y.data();

    for (size_t i = 0; i < x.row_count(); i++) {
        for (size_t j = 0; j < x.column_count(); j++) {
            y_ptr[j * x.row_count() + i] = x_ptr[i * x.column_count() + j];
        }
    }

    return y;
}

template <typename T>
inline void kahan_add(T& sum, T& c, T x) {
    const T y = x - c;
    const T t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

template <typename T>
inline T mean(const Array<T>& x) {
    const T* x_ptr = x.data();

    T sum = 0;
    T c = 0;

    for (size_t i = 0; i < x.count(); i++) {
        kahan_add(sum, c, x_ptr[i]);
    }

    return sum / x.count();
}

template <typename T>
inline T variance(const Array<T>& x) {
    const T m = mean<T>(x);
    const T* x_ptr = x.data();

    T sum = 0;
    T c = 0;

    for (size_t i = 0; i < x.count(); i++) {
        const T t = x_ptr[i] - m;
        kahan_add(sum, c, t * t);
    }

    return sum / (x.count() - 1);
}

template <typename T>
inline T standard_deviation(const Array<T>& x) {
    return std::sqrt(variance(x));
}

inline Array<float> generate_dataset(std::mt19937& rng,
                                     size_t row_count,
                                     size_t column_count,
                                     size_t gauss_terms) {
    auto locs = uniform<float>(rng, -10, 10, { column_count, gauss_terms });
    auto stds = uniform<float>(rng, 1, 10, { column_count, gauss_terms });
    auto Xt = make_zeros<float>({column_count, row_count});

    float* locs_ptr = locs.data();
    float* stds_ptr = stds.data();
    float* Xt_ptr = Xt.data();

    for (size_t i = 0; i < column_count; i++) {
        float* column_ptr = &Xt_ptr[i * row_count];
        for (size_t j = 0; j < gauss_terms; j++) {
            const float loc = locs_ptr[i * gauss_terms + j];
            const float std = stds_ptr[i * gauss_terms + j];
            std::normal_distribution<float> distr(loc, std);

            for (size_t k = 0; k < row_count; k++) {
                column_ptr[k] += distr(rng);
            }
        }

        float min_value =  std::numeric_limits<float>::infinity();
        float max_value = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < row_count; j++) {
            if (column_ptr[j] < min_value) {
                min_value = column_ptr[j];
            }
            if (column_ptr[j] > max_value) {
                max_value = column_ptr[j];
            }
        }

        float norm = max_value - min_value;
        for (size_t j = 0; j < row_count; j++) {
            column_ptr[j] = (column_ptr[j] - min_value) / norm;
        }
    }

    return transpose(Xt);
}

template <typename Distance>
Array<double> compute_distances(const Array<float>& X,
                                size_t distance_count,
                                Distance&& dist_func = Distance{}) {
    if (distance_count > X.row_count() * (X.row_count() - 1) / 2) {
        throw std::invalid_argument{"Too much distances requested"};
    }

    std::mt19937 rng(77777);
    std::uniform_int_distribution<size_t> distr(0, X.row_count() - 1);

    auto distances = make_empty<double>({ distance_count, 1 });
    double* d_ptr = distances.data();

    for (size_t d = 0; d < distance_count; d++) {
        size_t i = 0;
        size_t j = 0;

        while (i == j) {
            i = distr(rng);
            j = distr(rng);
        }

        float* x_i = X.row_data(i);
        float* x_j = X.row_data(j);

        d_ptr[d] = dist_func(x_i, x_j, X.column_count());
    }

    return distances;
}

Array<double> compute_distances_exact(const Array<float>& X, size_t distance_count) {
    return compute_distances(X, distance_count, [](float* x_1, float* x_2, size_t p) {
        double sum = 0;
        double c = 0;
        for (size_t i = 0; i < p; i++) {
            const double d = double(x_1[i]) - double(x_2[i]);
            kahan_add(sum, c, d * d);
        }
        return sum;
    });
}

template <typename F>
Array<double> compute_distances_approx(const Array<float>& X, size_t distance_count) {
    return compute_distances(X, distance_count, [](float* x_1, float* x_2, size_t p) {
        float sum_1 = 0;
        float sum_2 = 0;
        float sum_3 = 0;
        for (size_t i = 0; i < p; i++) {
            const float x_1_f16 = round_to<F>(x_1[i]);
            const float x_2_f16 = round_to<F>(x_2[i]);
            sum_1 = round_to<F>(sum_1 + x_1_f16 * x_1_f16);
            sum_2 = round_to<F>(sum_2 + x_1_f16 * x_2_f16);
            sum_3 = round_to<F>(sum_3 + x_2_f16 * x_2_f16);
        }
        return double(round_to<F>(round_to<F>(sum_1 - round_to<F>(2 * sum_2)) + sum_3));
    });
}

template <typename F>
Array<double> estimate_delta(const Array<float>& X) {
    Array<double> D_exact = compute_distances_exact(X, 5000);
    Array<double> D_approx = compute_distances_approx<F>(X, 5000);
    return D_approx - D_exact;
}

template <typename F>
Array<double> estimate_sigma_range(size_t min_feature_count,
                                   size_t max_feature_count,
                                   size_t step) {
    const size_t data_row_count = 500;
    const size_t gauss_term_count = 5;
    const size_t sigma_row_count = (max_feature_count - min_feature_count) / step +
                               int((max_feature_count - min_feature_count) % step > 0);
    auto sigmas = make_empty<double>({ sigma_row_count, 2 });

    std::mt19937 rng(77777);
    std::atomic<int> counter = 0;

    auto watcher = std::thread{[&]() {
        while (counter < sigma_row_count) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(100ms);
            std::cout << "\r";
            std::cout << counter << "/" << sigma_row_count;
            std::cout << std::flush;
        }

        std::cout << std::endl;
    }};

    #pragma omp parallel for private(rng) shared(counter) schedule(dynamic)
    for (size_t i = 0; i < sigma_row_count; i++) {
        const size_t feature_count = min_feature_count + i * step;

        // Each `rng` generates `data_row_count * max_feature_count * gauss_term_count`
        // random numbers in worst case, so use skip-ahead to guarantee uncorrelated sequences
        rng.discard(data_row_count * max_feature_count * gauss_term_count * i);

        auto X = generate_dataset(rng, data_row_count, feature_count, gauss_term_count);
        auto delta = estimate_delta<F>(X);
        sigmas.row_data(i)[0] = mean(delta);
        sigmas.row_data(i)[1] = standard_deviation(delta);

        counter++;
    }

    watcher.join();
    return sigmas;
}

template <typename F>
void write_sigma_range(const std::string& filename,
                       size_t max_feature_count,
                       size_t step) {
    auto sigmas = estimate_sigma_range<F>(2, max_feature_count, step);
    double* sigmas_ptr = sigmas.data();

    std::fstream file(filename, std::ios::trunc | std::ios::out);
    file << "features,mean,sigma" << std::endl;
    for (size_t i = 2; i < max_feature_count; i += step) {
        file << i << ',' << *(sigmas_ptr++) << ',' << *(sigmas_ptr++) << std::endl;
    }
}

int main(int argc, char const *argv[]) {
    omp_set_num_threads(28);

    // write_sigma_range<bf16_tag>("bf16_sigma.csv", 100, 1);
    write_sigma_range<f16_tag>("f16_sigma.csv", 1000, 1);
    // write_sigma_range<f32_tag>("f32_sigma.csv", 10000, 100);

    return 0;
}
