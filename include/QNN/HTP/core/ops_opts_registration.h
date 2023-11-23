//==============================================================================
//
// Copyright (c) 2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef OP_PACKAGE_OPS_OPTS_LIST_H
#define OP_PACKAGE_OPS_OPTS_LIST_H 1

// Note that Op files must include this file AFTER they've included either
// typical_op.h or variadic_op.h, since these headers both give definitions for DerivedType

#include "log.h"
#include "ops_opts_registration_defs.h"
#include "optimize_flags.h"
#include "optimize.h"
#include "op_register.h"
#include "op_register_ext.h"
#include <cstdint>
#include <cinttypes>
#include <string>
#include <string_view>

namespace hnnx {

/** @brief reg_op_node */
class reg_op_node {
    /** @brief parms parameters (cost func, flags, etc) for the Op */
    union {
        op_reg_parms op_parms;
        simop_reg_parms simple_op_parms;
    };
    /** @brief op_name */
    uint16_t op_name_offset;
    /** @brief type_tag */
    uint16_t type_suffix_offset;

    std::string_view const get_subview(std::string_view const strtab, std::string_view::size_type const start) const
    {
        return std::string_view{strtab.data() + start};
    }

  public:
    /** @brief reg_op_node @param a @param n @param t */
    constexpr reg_op_node(op_reg_parms p, uint16_t n, uint16_t t) noexcept
        : op_parms(p), op_name_offset(n), type_suffix_offset(t)
    {
    }

    constexpr reg_op_node(simop_reg_parms p, uint16_t n, uint16_t t) noexcept
        : simple_op_parms(p), op_name_offset(n), type_suffix_offset(t)
    {
    }

    /** @brief reg_op_node */
    constexpr reg_op_node() noexcept : reg_op_node(op_reg_parms{}, 0, 0) {}

    /** @brief process invoke the make_op_custom function */
    void core_process(std::string_view const op_name_strtab, std::string_view const type_suffix_strtab) const
    {
        std::string_view const op_name = get_subview(op_name_strtab, op_name_offset);
        std::string_view const type_tag = get_subview(type_suffix_strtab, type_suffix_offset);
        hnnx::make_op_custom(op_name, type_tag, op_parms);
    }

    /** @brief process append external oppkg ops into op vector for later use */
    void pkg_process(std::string_view const op_name_strtab, std::string_view const type_suffix_strtab) const
    {
        std::string_view const op_name = get_subview(op_name_strtab, op_name_offset);
        std::string_view const type_suffix = get_subview(type_suffix_strtab, type_suffix_offset);
        std::vector<std::unique_ptr<PackageOpStorageBase>> &ops = current_package_ops_storage_vec_func();
        ops.push_back(std::make_unique<PackageOpStorageBase>(
                op_name, type_suffix, simple_op_parms.sim_newop, *(simple_op_parms.tinf),
                simple_op_parms.deserializer_reg_func, simple_op_parms.deserialize_func, simple_op_parms.cost_f,
                simple_op_parms.flags));
    }
};

#ifdef PREPARE_DISABLED
/** @brief reg_optim_node This is stub class that does not
 *  register DEF_OPTs when prepare is disabled.
*/
class reg_optim_node {
  public:
    /** @brief No-op when prepare is disabled */
    void core_process(std::string_view const fname) const { (void)fname; }

    /** @brief No-op when prepare is disabled */
    void pkg_process(std::string_view const fname) const { (void)fname; }
};
#else
/** @brief reg_optim_node */
class reg_optim_node {
    /** @brief defopt */
    hnnx::get_entire_defopt_t defopt;
    /** @brief flags */
    OptimFlags::flags_t flags;
    /** @brief priority */
    uint16_t priority;
    /** @brief line */
    uint16_t line;

  public:
    /** @brief reg_optim_node @param p @param fl @param m @param c @param r @param f @param l */
    constexpr reg_optim_node(uint16_t const p, OptimFlags::flags_t const fl, hnnx::get_entire_defopt_t d,
                             uint16_t const l) noexcept
        : defopt(d), flags(fl), priority(p), line(l)
    {
    }

    /** @brief reg_optim_node */
    constexpr reg_optim_node() noexcept : reg_optim_node(0, 0U, nullptr, 0) {}

    /** @brief process invoke the add_package_opt function */
    void core_process(std::string_view const fname) const
    {
        hnnx::add_package_opt(current_package_opts_storage_vec_func(), priority, flags, defopt, fname.data(), line);
    }

    /** @brief process invoke the add_package_opt function for external oppkg */
    void pkg_process(std::string_view const fname) const
    {
        hnnx::add_package_opt(current_package_opts_storage_vec_func(), priority, flags, defopt, fname.data(), line);
    }
};
#endif

/** @brief sv_size_wrapper a wrapper template for string_view that carries the view size as
 *  a template parameter. This allows the size to be inferred by the built_array
 *  constructor template.
 */
template <std::string_view::size_type S> struct sv_size_wrapper {
    std::string_view v;
};

/** @brief built_array */
template <typename T, uint32_t N> class built_array {
    /** @brief arr */
    std::array<T, N> arr{};

  public:
    /** @brief size
     *  @return the array size
     */
    static constexpr uint32_t size() { return N; }
    /** @brief get_arr
     *  @return the array
     */
    constexpr const std::array<T, N> get_arr() const noexcept { return arr; }
    /** @brief built_array 
     *  @param old the previous array
     *  @param add the new element to append
     */
    constexpr built_array(built_array<T, N - 1> const &old, T add)
    {
        for (uint32_t i = 0U; i < N - 1U; i++) {
            arr[i] = old.get_arr()[i];
        }
        arr[N - 1U] = add;
    }
    /** @brief append
     *  @param add the new element to append
     *  @return the new array
     */
    constexpr built_array<T, N + 1> append(T add) const { return built_array<T, N + 1>(*this, add); }

    /** @brief append
     *  @param add the new element to append
     *  @return the new array
     */
    template <std::string_view::size_type I> constexpr built_array<T, N + I> append(sv_size_wrapper<I> add) const
    {
        return built_array<T, N + I>(*this, add);
    }

    /** @brief built_array 
     *  @param old the previous array
     *  @param add a view of the array of new elements to append
     */
    template <std::string_view::size_type I>
    constexpr built_array(built_array<T, N - I> const &old, sv_size_wrapper<I> add)
    {
        for (uint32_t i = 0U; i < (N - I); i++) {
            arr[i] = old.get_arr()[i];
        }
        for (uint32_t i = (N - I); i < N; i++) {
            arr[i] = add.v[i - (N - I)];
        }
    }
};

/** @brief built_array specialization for N = 0 */
template <typename T> class built_array<T, 0> {
  public:
    /** @brief built_array constructor */
    constexpr built_array() = default;
    /** @brief append
     *  @param add the new element to append
     *  @return the new array
     */
    constexpr built_array<T, 1> append(T add) const { return built_array<T, 1>(*this, add); }

    /** @brief append
     *  @param add the new element to append
     *  @return the new array
     */
    template <std::string_view::size_type I> constexpr built_array<T, I> append(sv_size_wrapper<I> add) const
    {
        return built_array<T, I>(*this, add);
    }

    /** @brief get_arr
     *  @return the array
     */
    constexpr static const std::array<T, 0> get_arr() noexcept { return std::array<T, 0>{}; }
};

/** @brief op_name_strtab_t empty struct to help specialize arr_container for the op_name string table */
struct op_name_strtab_t {
};
/** @brief type_suffix_strtab empty struct to help specialize arr_container for the type_suffix string table */
struct type_suffix_strtab_t {
};

template <typename> inline constexpr bool is_strtab = false;
template <> inline constexpr bool is_strtab<op_name_strtab_t> = true;
template <> inline constexpr bool is_strtab<type_suffix_strtab_t> = true;

/** @brief arr_container */
template <typename T, bool S = is_strtab<T>> struct arr_container {
    /** @brief chain link to the built_array contained in this structure */
    template <typename UNIQ_TY, uint32_t I> static constexpr built_array<T, I> chain = {};
};

/** @brief arr_container<T, true> */
template <typename T> struct arr_container<T, true> {
    /** @brief chain link to the built_array contained in this structure */
    template <typename UNIQ_TY, uint32_t I, uint32_t S>
    static constexpr built_array<std::string::value_type, S> chain = {};
};

/** @brief reg_op_table */
class reg_op_table {
    reg_op_node const *entries;
    uint32_t num_entries;
    std::string_view op_name_strtab;
    std::string_view type_suffix_strtab;

  public:
    constexpr reg_op_node const *get_entries() const noexcept { return entries; }
    constexpr uint32_t get_num_entries() const noexcept { return num_entries; }
    constexpr std::string_view const get_op_name_strtab() const noexcept { return op_name_strtab; }
    constexpr std::string_view const get_type_suffix_strtab() const noexcept { return type_suffix_strtab; }
    constexpr reg_op_table(reg_op_node const *p, uint32_t const n, std::string_view::value_type const *o,
                           std::string_view::size_type const o_size, std::string_view::value_type const *t,
                           std::string_view::size_type const t_size) noexcept
        : entries(p), num_entries(n), op_name_strtab{o, o_size}, type_suffix_strtab{t, t_size}
    {
    }
    constexpr reg_op_table() noexcept : reg_op_table(nullptr, 0U, "", 0U, "", 0U) {}
};

/** @brief reg_op_table_wrapper */
using reg_op_table_wrapper = reg_op_table const *(*)();

/** @brief reg_opt_table */
class reg_opt_table {
    reg_optim_node const *entries;
    uint32_t num_entries;
    std::string_view file_name;

  public:
    constexpr reg_optim_node const *get_entries() const noexcept { return entries; }
    constexpr uint32_t get_num_entries() const noexcept { return num_entries; }
    constexpr std::string_view const get_file_name() const noexcept { return file_name; }
    constexpr reg_opt_table(reg_optim_node const *p, uint32_t const n, std::string_view::value_type const *f) noexcept
        : entries(p), num_entries(n), file_name{f}
    {
    }
    constexpr reg_opt_table() noexcept : reg_opt_table(nullptr, 0U, "") {}
};

/** @brief reg_opt_table_wrapper */
using reg_opt_table_wrapper = reg_opt_table const *(*)();

/** @brief op_name_strtab_container */
using op_name_strtab_container = arr_container<op_name_strtab_t>;
/** @brief type_suffix_strtab_container */
using type_suffix_strtab_container = arr_container<type_suffix_strtab_t>;
/** @brief op_arr_container */
using op_arr_container = arr_container<reg_op_node>;
/** @brief opt_arr_container */
using opt_arr_container = arr_container<reg_optim_node>;
/** @brief op_table_arr_container */
using op_table_arr_container = arr_container<reg_op_table_wrapper>;
/** @brief opt_table_arr_container */
using opt_table_arr_container = arr_container<reg_opt_table_wrapper>;

/** @brief ba_str a built_array of char strings */
template <uint32_t I> using ba_str = built_array<std::string::value_type, I>;

/** @brief ba_op a built_array of reg_op_nodes */
template <uint32_t I> using ba_op = built_array<reg_op_node, I>;

/** @brief ba_opt a built_array of reg_optim_nodes */
template <uint32_t I> using ba_opt = built_array<reg_optim_node, I>;

/** @brief ba_op_table a built_array of reg_op_table_wrappers */
template <uint32_t I> using ba_op_table = built_array<reg_op_table_wrapper, I>;

/** @brief ba_opt_table a built_array of reg_opt_table_wrappers */
template <uint32_t I> using ba_opt_table = built_array<reg_opt_table_wrapper, I>;

/**
 * @brief
 * NodeCounter template converts the __COUNTER__'s
 * current value to counts for number of reg_op_nodes and 
 * reg_optim_nodes created so far. It is specialized upon every
 * REGISTER_OP/DEF_OPT by incrementing either "reg_op_count"
 * or "reg_opt_count", with member functions that can get the 
 * current counts.
 */
template <typename UNIQ_TY, int32_t I> class NodeCounter {
    /** @brief inc_op @return 0, or 1 if the op count is incremented */
    constexpr static int32_t inc_op() noexcept { return 0; }
    /** @brief inc_opt @return 0, or 1 if the opt count is incremented */
    constexpr static int32_t inc_opt() noexcept { return 0; }
    /** @brief inc_op_name_strtab_size @return 0, or some string size constant if the string table needs to grow */
    constexpr static uint64_t inc_op_name_strtab_size() noexcept { return 0; }
    /** @brief inc_type_suffix_strtab_size @return 0, or some string size constant if the string table needs to grow */
    constexpr static uint64_t inc_type_suffix_strtab_size() noexcept { return 0; }

  public:
    /** @brief reg_op_count @return The number of ops that have been registered so far */
    constexpr static int32_t reg_op_count() noexcept { return inc_op() + NodeCounter<UNIQ_TY, I - 1>::reg_op_count(); }
    /** @brief reg_op_count @return The number of opts that have been registered so far */
    constexpr static int32_t reg_opt_count() noexcept
    {
        return inc_opt() + NodeCounter<UNIQ_TY, I - 1>::reg_opt_count();
    }
    /** @brief op_name_strtab_size @return The string table size for the ops that have been registered so far */
    constexpr static uint64_t op_name_strtab_size() noexcept
    {
        return inc_op_name_strtab_size() + NodeCounter<UNIQ_TY, I - 1>::op_name_strtab_size();
    }
    /** @brief type_suffix_strtab_size @return The string table size for the ops that have been registered so far */
    constexpr static uint64_t type_suffix_strtab_size() noexcept
    {
        return inc_type_suffix_strtab_size() + NodeCounter<UNIQ_TY, I - 1>::type_suffix_strtab_size();
    }
};

/** @brief Shorthand for op_name_strtab_container::chain<...> */
template <typename U, size_t I> constexpr auto op_name_chain()
{
    return op_name_strtab_container::chain<U, NodeCounter<U, I>::reg_op_count(),
                                           NodeCounter<U, I>::op_name_strtab_size()>;
}

/** @brief Shorthand for type_suffix_strtab_container::chain<...> */
template <typename U, size_t I> constexpr auto type_tag_chain()
{
    return type_suffix_strtab_container::chain<U, NodeCounter<U, I>::reg_op_count(),
                                               NodeCounter<U, I>::type_suffix_strtab_size()>;
}

/**
 * @brief StrtabUpdate A class template for storing:
 * - The result of the existence check
 * - The offset the new string
 * when appending to the Op name and type suffix string tables.
 * 
 * @tparam U Unique type for identifying the translation unit containing this update
 * @tparam I Unique index number identifying the "ith" Op to be registered in this file
 */
template <typename U, uint32_t I> struct StrtabUpdate {
    /** @brief is_new_op_name whether the Op name to be appended is already present in the op_name_strtab */
    static bool const is_new_op_name;
    /** @brief op_name_offset short offset locating the Op name in the op_name_strtab */
    static uint16_t const op_name_offset;
    /** @brief is_new_type_suffix whether the type suffix to be appended is already present in the type_suffix_strtab */
    static bool const is_new_type_suffix;
    /** @brief type_suffix_offset short offset locating the type suffix in the type_suffix_strtab */
    static uint16_t const type_suffix_offset;
};

/**
 * @brief strtab_append Append string to table iff it is not already present.
 * @tparam U Unique type to ensure independence of specializations across translation units 
 * @tparam I REGISTER_OP index number
 * @tparam N Current table size
 */
template <typename U, uint32_t I, std::string_view::size_type M, bool A, uint32_t N>
constexpr auto strtab_append(ba_str<N> const &curr, std::string_view const add)
{
    if constexpr (A) {
        sv_size_wrapper<M> const w{add};
        return curr.append(w);
    } else {
        return curr;
    }
}

/** @brief make_string_view Convert an array of string data into a string_view, but
 *  substitute in an empty string if the array's .data() would be nullptr.
 */
template <std::string_view::size_type N>
constexpr std::string_view make_string_view(std::array<std::string::value_type, N> const &arr) noexcept
{
    return arr.size() != 0 ? std::string_view{arr.data(), arr.size()} : std::string_view{"", 1};
}

} // namespace hnnx

#endif // OP_PACKAGE_OPS_OPTS_LIST_H
