// NOLINT
#ifndef MLLM_GGUF_HPP
#define MLLM_GGUF_HPP
#include "ParamWriter.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
struct gguf_str {
    uint64_t n; // GGUFv2
    char *data;
};
#define ALIGNMENT_KEY "general.alignment"
#define GGUF_MAGIC "GGUF"
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGUF_VERSION 3
#define GGML_MAX_DIMS 4
enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 (5) support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    // k-quantizations
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
};
enum gguf_type {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT, // marks the end of the enum
};

struct gguf_context;

struct gguf_init_params {
    bool no_alloc;

    // if not NULL, create a ggml_context and allocate the tensor data in it
    struct ggml_context **ctx;
};
static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8] = sizeof(uint8_t),
    [GGUF_TYPE_INT8] = sizeof(int8_t),
    [GGUF_TYPE_UINT16] = sizeof(uint16_t),
    [GGUF_TYPE_INT16] = sizeof(int16_t),
    [GGUF_TYPE_UINT32] = sizeof(uint32_t),
    [GGUF_TYPE_INT32] = sizeof(int32_t),
    [GGUF_TYPE_FLOAT32] = sizeof(float),
    [GGUF_TYPE_BOOL] = sizeof(bool),
    [GGUF_TYPE_STRING] = sizeof(struct gguf_str),
    [GGUF_TYPE_UINT64] = sizeof(uint64_t),
    [GGUF_TYPE_INT64] = sizeof(int64_t),
    [GGUF_TYPE_FLOAT64] = sizeof(double),
    [GGUF_TYPE_ARRAY] = 0, // undefined

};
typedef struct {
    const char *type_name;
    int blck_size = 1;
    size_t type_size;
    bool is_quantized;
    bool is_available = true;
} ggml_type_traits_t;
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");
static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_I8] = {
        .type_name = "i8",
        .blck_size = 1,
        .type_size = sizeof(int8_t),
        .is_quantized = false,
    },
    [GGML_TYPE_I16] = {
        .type_name = "i16",
        .blck_size = 1,
        .type_size = sizeof(int16_t),
        .is_quantized = false,
    },
    [GGML_TYPE_I32] = {
        .type_name = "i32",
        .blck_size = 1,
        .type_size = sizeof(int32_t),
        .is_quantized = false,
    },
    [GGML_TYPE_F32] = {
        .type_name = "f32",
        .blck_size = 1,
        .type_size = sizeof(float),
        .is_quantized = false,

    },
    [GGML_TYPE_F16] = {
        .type_name = "f16",
        .blck_size = 1,
        .type_size = sizeof(uint16_t),
        .is_quantized = false,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name = "q4_0",
        .blck_size = QK4_0,
        .type_size = sizeof(block_q4_0),
        .is_quantized = true,
        // .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        // .from_float               = quantize_row_q4_0,
        // .from_float_reference     = (ggml_from_float_t) quantize_row_q4_0_reference,
        // .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        // .vec_dot_type             = GGML_TYPE_Q8_0,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name = "q4_1",
        .is_available = false,
        // .blck_size                = QK4_1,
        // .type_size                = sizeof(block_q4_1),
        // .is_quantized             = true,
        // .to_float                 = (ggml_to_float_t) dequantize_row_q4_1,
        // .from_float               = quantize_row_q4_1,
        // .from_float_reference     = (ggml_from_float_t) quantize_row_q4_1_reference,
        // .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        // .vec_dot_type             = GGML_TYPE_Q8_1,
    },
    [4] = {
        // GGML_TYPE_Q4_2
        .type_name = "DEPRECATED",
        .blck_size = 0,
        .type_size = 0,
        .is_quantized = false,
        .is_available = false,

        // .to_float                 = NULL,
        // .from_float               = NULL,
        // .from_float_reference     = NULL,
        // .vec_dot                  = NULL,
        // .vec_dot_type             = GGML_TYPE_COUNT,
    },
    [5] = {
        // GGML_TYPE_Q4_3
        .type_name = "DEPRECATED",
        .blck_size = 0,
        .type_size = 0,
        .is_quantized = false,
        .is_available = false,

        // .to_float                 = NULL,
        // .from_float               = NULL,
        // .from_float_reference     = NULL,
        // .vec_dot                  = NULL,
        // .vec_dot_type             = GGML_TYPE_COUNT,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name = "q5_0",
        // .blck_size                = QK5_0,
        // .type_size                = sizeof(block_q5_0),
        // .is_quantized             = true,
        .is_available = false,

        // .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        // .from_float               = quantize_row_q5_0,
        // .from_float_reference     = (ggml_from_float_t) quantize_row_q5_0_reference,
        // .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        // .vec_dot_type             = GGML_TYPE_Q8_0,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name = "q5_1",
        .is_available = false,

        // .blck_size                = QK5_1,
        // .type_size                = sizeof(block_q5_1),
        // .is_quantized             = true,
        // .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        // .from_float               = quantize_row_q5_1,
        // .from_float_reference     = (ggml_from_float_t) quantize_row_q5_1_reference,
        // .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        // .vec_dot_type             = GGML_TYPE_Q8_1,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name = "q8_0",
        .blck_size = QK8_0,
        .type_size = sizeof(block_q8_0),
        .is_quantized = true,
        // .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        // .from_float               = quantize_row_q8_0,
        // .from_float_reference     = (ggml_from_float_t) quantize_row_q8_0_reference,
        // .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        // .vec_dot_type             = GGML_TYPE_Q8_0,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name = "q8_1",
        .is_available = false,

        // .blck_size                = QK8_1,
        // .type_size                = sizeof(block_q8_1),
        // .is_quantized             = true,
        // .from_float               = quantize_row_q8_1,
        // .from_float_reference     = (ggml_from_float_t) quantize_row_q8_1_reference,
        // .vec_dot_type             = GGML_TYPE_Q8_1,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name = "q2_K",
        .is_available = false,

        // .blck_size                = QK_K,
        // .type_size                = sizeof(block_q2_K),
        // .is_quantized             = true,
        // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        // .from_float               = quantize_row_q2_K,
        // .from_float_reference     = (ggml_from_float_t) quantize_row_q2_K_reference,
        // .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        // .vec_dot_type             = GGML_TYPE_Q8_K,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name = "q3_K",
        .is_available = false,

        // .blck_size                = QK_K,
        // .type_size                = sizeof(block_q3_K),
        // .is_quantized             = true,
        // .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        // .from_float               = quantize_row_q3_K,
        // .from_float_reference     = (ggml_from_float_t) quantize_row_q3_K_reference,
        // .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        // .vec_dot_type             = GGML_TYPE_Q8_K,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name = "q4_K",
        .blck_size = QK_K,
        .type_size = sizeof(block_q4_K),
        .is_quantized = true,
        // .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        // .from_float               = quantize_row_q4_K,
        // .from_float_reference     = (ggml_from_float_t) quantize_row_q4_K_reference,
        // .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        // .vec_dot_type             = GGML_TYPE_Q8_K,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name = "q5_K",
        .is_available = false,

        // .blck_size                = QK_K,
        // .type_size                = sizeof(block_q5_K),
        // .is_quantized             = true,
        // .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        // .from_float               = quantize_row_q5_K,
        // .from_float_reference     = (ggml_from_float_t) quantize_row_q5_K_reference,
        // .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        // .vec_dot_type             = GGML_TYPE_Q8_K,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name = "q6_K",
        .is_available = false,

        // .blck_size                = QK_K,
        // .type_size                = sizeof(block_q6_K),
        // .is_quantized             = true,
        // .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        // .from_float               = quantize_row_q6_K,
        // .from_float_reference     = (ggml_from_float_t) quantize_row_q6_K_reference,
        // .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        // .vec_dot_type             = GGML_TYPE_Q8_K,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name = "q8_K",
        .blck_size = QK_K,
        .type_size = sizeof(block_q8_K),
        .is_quantized = true,
        // .from_float               = quantize_row_q8_K,
    }};
static const char *GGUF_TYPE_NAME[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8] = "u8",
    [GGUF_TYPE_INT8] = "i8",
    [GGUF_TYPE_UINT16] = "u16",
    [GGUF_TYPE_INT16] = "i16",
    [GGUF_TYPE_UINT32] = "u32",
    [GGUF_TYPE_INT32] = "i32",
    [GGUF_TYPE_FLOAT32] = "f32",
    [GGUF_TYPE_BOOL] = "bool",
    [GGUF_TYPE_STRING] = "str",
    [GGUF_TYPE_ARRAY] = "arr",
    [GGUF_TYPE_UINT64] = "u64",
    [GGUF_TYPE_INT64] = "i64",
    [GGUF_TYPE_FLOAT64] = "f64",
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

union gguf_value {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;

    struct gguf_str str;

    struct {
        enum gguf_type type;

        uint64_t n; // GGUFv2
        void *data;
    } arr;
};

struct gguf_kv {
    struct gguf_str key;

    enum gguf_type type;
    union gguf_value value;
    bool is_supported;
};

struct gguf_header {
    char magic[4];
    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv;      // GGUFv2
};

struct gguf_tensor_info {
    struct gguf_str name;

    uint32_t n_dims;
    uint64_t ne[GGML_MAX_DIMS];

    enum ggml_type type;

    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void *data;
    size_t size;
};

struct gguf_context {
    struct gguf_header header;

    struct gguf_kv *kv;
    struct gguf_tensor_info *infos;

    size_t alignment;
    size_t offset; // offset of `data` from beginning of file
    size_t size;   // size of `data` in bytes

    // uint8_t * padding;
    void *data;
};

static bool gguf_fread_el(FILE *file, void *dst, size_t size, size_t *offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

static bool gguf_fread_str(FILE *file, struct gguf_str *p, size_t *offset) {
    p->n = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && gguf_fread_el(file, &p->n, sizeof(p->n), offset);
    p->data = (char *)calloc(p->n + 1, 1);
    ok = ok && gguf_fread_el(file, p->data, p->n, offset);

    return ok;
}
static gguf_kv *get_kv(std::string key, std::unordered_map<std::string, gguf_kv> kv_map) {
    auto it = kv_map.find(key);
    if (it == kv_map.end()) {
        fprintf(stderr, "%s: key %s not found.\n", __func__, key.c_str());
        return nullptr;
    }
    return &it->second;
}
static size_t get_tensor_size(const struct gguf_tensor_info *info) {
    ggml_type_traits_t type = type_traits[info->type];
    if (!type.is_available) {
        fprintf(stderr, "%s: type %s is not available.\n", __func__, type.type_name);
        exit(-1);
    }
    size_t size = type.type_size * (info->ne[0] / type.blck_size);
    for (int i = 0; i < info->n_dims; ++i) {
        size *= info->ne[i];
    }
    return size;
}

static void from_gguf(std::string fname, ParamWriter *writer) {
    FILE *file = fopen(fname.c_str(), "rb");
    if (!file) {
        std::cout << "Failed to open file " << fname << "\n";
        exit(-1);
    }
    size_t offset = 0;

    char magic[4];
    {
        gguf_fread_el(file, &magic, sizeof(magic), &offset);

        for (uint32_t i = 0; i < sizeof(magic); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                fprintf(stderr, "%s: invalid magic characters %s.\n", __func__, magic);
                fclose(file);
                exit(-1);
            }
        }
    }
    bool ok = true;
    gguf_header header;
    {
        ok = ok && gguf_fread_el(file, &header.version, sizeof(header.version), &offset);
        ok = ok && gguf_fread_el(file, &header.n_tensors, sizeof(header.n_tensors), &offset);
        ok = ok && gguf_fread_el(file, &header.n_kv, sizeof(header.n_kv), &offset);
        if (!ok) {
            fprintf(stderr, "%s: failed to read header.\n", __func__);
            fclose(file);
            exit(-1);
        }
        if (header.version == 1) {
            fprintf(stderr, "%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
            fclose(file);
            exit(-1);
        }
        if (header.version > GGUF_VERSION) {
            fprintf(stderr, "%s: unsupported version %d.\n", __func__, header.version);
            fclose(file);
            exit(-1);
        }
    }
    std::unordered_map<std::string, gguf_kv> kv_map;
    kv_map.reserve(header.n_kv);
    {
        for (uint64_t i = 0; i < header.n_kv; i++) {
            gguf_kv kv;
            ok = ok && gguf_fread_str(file, &kv.key, &offset);
            ok = ok && gguf_fread_el(file, &kv.type, sizeof(kv.type), &offset);
            kv.is_supported = true;
            switch (kv.type) {
            case GGUF_TYPE_UINT8:
                ok = ok && gguf_fread_el(file, &kv.value.uint8, sizeof(kv.value.uint8), &offset);
                break;
            case GGUF_TYPE_INT8:
                ok = ok && gguf_fread_el(file, &kv.value.int8, sizeof(kv.value.int8), &offset);
                break;
            case GGUF_TYPE_UINT16:
                ok = ok && gguf_fread_el(file, &kv.value.uint16, sizeof(kv.value.uint16), &offset);
                break;
            case GGUF_TYPE_INT16:
                ok = ok && gguf_fread_el(file, &kv.value.int16, sizeof(kv.value.int16), &offset);
                break;
            case GGUF_TYPE_UINT32:
                ok = ok && gguf_fread_el(file, &kv.value.uint32, sizeof(kv.value.uint32), &offset);
                break;
            case GGUF_TYPE_INT32:
                ok = ok && gguf_fread_el(file, &kv.value.int32, sizeof(kv.value.int32), &offset);
                break;
            case GGUF_TYPE_FLOAT32:
                ok = ok && gguf_fread_el(file, &kv.value.float32, sizeof(kv.value.float32), &offset);
                break;
            case GGUF_TYPE_BOOL:
                ok = ok && gguf_fread_el(file, &kv.value.bool_, sizeof(kv.value.bool_), &offset);
                break;
            case GGUF_TYPE_STRING:
                ok = ok && gguf_fread_str(file, &kv.value.str, &offset);
                break;
            case GGUF_TYPE_UINT64:
                ok = ok && gguf_fread_el(file, &kv.value.uint64, sizeof(kv.value.uint64), &offset);
                break;
            case GGUF_TYPE_INT64:
                ok = ok && gguf_fread_el(file, &kv.value.int64, sizeof(kv.value.int64), &offset);
                break;
            case GGUF_TYPE_FLOAT64:
                ok = ok && gguf_fread_el(file, &kv.value.float64, sizeof(kv.value.float64), &offset);
                break;
            default:
                fprintf(stderr, "%s: invalid type %d.\n", __func__, kv.type);
                kv.is_supported = false;
            }
            if (!ok) {
                fprintf(stderr, "%s: failed to read kv.\n", __func__);
                fclose(file);
                exit(-1);
            }
            kv_map.insert({std::string(kv.key.data), kv});
        }
    }
    // padding
    {
        auto align = get_kv(ALIGNMENT_KEY, kv_map);
        if (align == nullptr) {
            fprintf(stderr, "%s: key %s not found.\n", __func__, ALIGNMENT_KEY);
            fclose(file);
            exit(-1);
        }

        std::unordered_map<std::string, gguf_tensor_info> info_map;
        info_map.reserve(header.n_tensors);
        std::vector<std::string> tensor_names;
        {
            for (uint64_t i = 0; i < header.n_tensors; i++) {
                gguf_tensor_info info;
                for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                    info.ne[j] = 1;
                }
                ok = ok && gguf_fread_str(file, &info.name, &offset);
                ok = ok && gguf_fread_el(file, &info.n_dims, sizeof(info.n_dims), &offset);
                // ok = ok && gguf_fread_el(file, &info.ne, sizeof(info.ne), &offset);
                for (int j = 0; j < info.n_dims; ++j) {
                    ok = ok && gguf_fread_el(file, &info.ne[j], sizeof(info.ne[j]), &offset);
                }
                ok = ok && gguf_fread_el(file, &info.type, sizeof(info.type), &offset);
                ok = ok && gguf_fread_el(file, &info.offset, sizeof(info.offset), &offset);
                if (!ok) {
                    fprintf(stderr, "%s: failed to read tensor info.\n", __func__);
                    fclose(file);
                    exit(-1);
                }
                info_map.insert({std::string(info.name.data), info});
                tensor_names.emplace_back(info.name.data);
                printf("name: %s, type: %s, offset: %lu, size: %lu\n", info.name.data, GGUF_TYPE_NAME[info.type], info.offset, get_tensor_size(&info));
            }
        }
        writer->paddingIndex(tensor_names);

        {
            auto alignment = align->value.uint32;
            const size_t offset_pad = offset % alignment;
            if (offset_pad != 0) {
                offset += alignment - offset_pad;
                fseek(file, offset, SEEK_SET);
            }
        }
        {
            for (uint64_t i = 0; i < header.n_tensors; i++) {
                gguf_tensor_info info = info_map[tensor_names[i]];
                size_t size = get_tensor_size(&info);
                void *data = malloc(size);
                ok = ok && gguf_fread_el(file, data, size, &offset);
                if (!ok) {
                    fprintf(stderr, "%s: failed to read tensor data.\n", __func__);
                    fclose(file);
                    exit(-1);
                }
                writer->writeParam(std::string(info.name.data), (DataType)info.type, data, size);
                printf("name: %s, type: %s, offset: %lu, size: %lu\n", info.name.data, GGUF_TYPE_NAME[info.type], info.offset, size);
                free(data);
            }
        }
        writer->writeIndex();
        fclose(file);
    }
}

#endif // MLLM_GGUF_HPP
