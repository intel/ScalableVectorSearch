/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

// header under test
#include "svs/lib/saveload.h"

// svs
#include "svs/lib/file.h"

// test helpers
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <filesystem>

// Test local objects.
namespace {

// Test object saving.
struct NotSaveable {};

struct Saveable {
  public:
    // Members
    int64_t val_;
    std::string file_;
    bool extra_arg_ = false;

  public:
    // Constructors
    Saveable(int64_t val, std::string file)
        : val_{val}
        , file_{std::move(file)} {}
    Saveable(int64_t val, std::string file, bool extra_arg)
        : val_{val}
        , file_{std::move(file)}
        , extra_arg_{extra_arg} {}

    // Static members
    static constexpr svs::lib::Version save_version = svs::lib::Version(0, 0, 1);

    // Methods
    svs::lib::SaveTable save(const svs::lib::SaveContext& ctx) const {
        auto my_path = ctx.get_directory() / file_;
        auto stream = svs::lib::open_write(my_path);
        svs::lib::write_binary(stream, val_);
        return svs::lib::SaveTable(
            save_version, {SVS_LIST_SAVE_(val), SVS_LIST_SAVE_(file)}
        );
    }

    static Saveable load(
        const toml::table& table,
        const svs::lib::LoadContext& ctx,
        const svs::lib::Version& version,
        bool extra_arg = false
    ) {
        CATCH_REQUIRE(version == save_version);
        auto val_from_table = SVS_LOAD_MEMBER_AT_(table, val);
        auto file = SVS_LOAD_MEMBER_AT_(table, file);

        // Make sure any side-effects were saved correctly.
        auto path = ctx.get_directory() / file;
        auto stream = svs::lib::open_read(path);
        auto val_from_file = svs::lib::read_binary<int64_t>(stream);
        CATCH_REQUIRE(val_from_table == val_from_file);
        return Saveable{val_from_table, file, extra_arg};
    }

    friend bool operator==(const Saveable&, const Saveable&) = default;
};

struct SaveableContextFree {
  public:
    // Members
    int32_t val_;
    bool extra_arg_ = false;

  public:
    SaveableContextFree(int32_t val, bool extra_arg)
        : val_{val}
        , extra_arg_{extra_arg} {}
    SaveableContextFree(int32_t val)
        : val_{val} {}

    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(svs::lib::Version(1, 2, 3), {SVS_LIST_SAVE_(val)});
    }

    static SaveableContextFree load(
        const toml::table& table, const svs::lib::Version& version, bool extra_arg = false
    ) {
        CATCH_REQUIRE(version == svs::lib::Version(1, 2, 3));
        return SaveableContextFree(SVS_LOAD_MEMBER_AT_(table, val), extra_arg);
    }

    friend bool
    operator==(const SaveableContextFree&, const SaveableContextFree&) = default;
};
static_assert(svs::lib::StaticLoadable<SaveableContextFree, const svs::lib::Version&>);
static_assert(svs::lib::
                  StaticLoadable<SaveableContextFree, const svs::lib::Version&, bool>);

struct SaveableHasBoth {
  public:
    mutable size_t context_free_calls_ = 0;
    mutable size_t contextual_calls_ = 0;
    bool constructed_context_free_ = false;
    bool constructed_with_context_ = false;
    bool extra_arg_ = false;

  public:
    SaveableHasBoth() = default;

    static constexpr svs::lib::Version save_version = svs::lib::Version(10, 20, 30);

    // Implement the context-free version and the contextual version.
    // Just return an empty table for both.
    svs::lib::SaveTable save() const {
        context_free_calls_ += 1;
        return svs::lib::SaveTable(save_version);
    }

    // Contextual version
    svs::lib::SaveTable save(const svs::lib::SaveContext& SVS_UNUSED(ctx)) const {
        contextual_calls_ += 1;
        return svs::lib::SaveTable(save_version);
    }

    // Context free load
    static SaveableHasBoth load(
        const toml::table& SVS_UNUSED(table),
        const svs::lib::Version& version,
        bool extra_arg = false
    ) {
        CATCH_REQUIRE(version == save_version);
        auto x = SaveableHasBoth();
        x.constructed_context_free_ = true;
        x.extra_arg_ = extra_arg;
        return x;
    }

    // Contextual load
    static SaveableHasBoth load(
        const toml::table& SVS_UNUSED(table),
        svs::lib::LoadContext& SVS_UNUSED(ctx),
        const svs::lib::Version& version
    ) {
        CATCH_REQUIRE(version == save_version);
        auto x = SaveableHasBoth();
        x.constructed_with_context_ = true;
        return x;
    }
};

struct Aggregate {
  public:
    Saveable a_;
    SaveableContextFree b_;
    SaveableHasBoth c_;

  public:
    Aggregate(Saveable a, SaveableContextFree b, SaveableHasBoth c)
        : a_{std::move(a)}
        , b_{std::move(b)}
        , c_{c} {}

    Aggregate(int64_t key1, std::string file, int32_t value)
        : a_{key1, std::move(file)}
        , b_{value}
        , c_{} {}

    svs::lib::SaveTable save(const svs::lib::SaveContext& ctx) const {
        auto table = svs::lib::SaveTable(
            svs::lib::Version(0, 0, 0),
            {
                SVS_LIST_SAVE_(a, ctx),
                SVS_LIST_SAVE_(b, ctx),
            }
        );
        // Test emplacement as well.
        SVS_INSERT_SAVE_(table, c, ctx);
        return table;
    }

    static Aggregate load(
        const toml::table& table,
        const svs::lib::LoadContext& ctx,
        const svs::lib::Version& version,
        bool extra_arg = false
    ) {
        CATCH_REQUIRE(version == svs::lib::Version(0, 0, 0));
        return Aggregate(
            SVS_LOAD_MEMBER_AT_(table, a, ctx),
            SVS_LOAD_MEMBER_AT_(table, b, ctx, extra_arg),
            SVS_LOAD_MEMBER_AT_(table, c, ctx)
        );
    }

    friend bool operator==(const Aggregate& x, const Aggregate& y) {
        return (x.a_ == y.a_) && (x.b_ == y.b_);
    }
};

struct BuiltIn {
  public:
    // Unsigned integers
    uint8_t u8_;
    uint16_t u16_;
    uint32_t u32_;
    uint64_t u64_;
    // Signed integers
    // Leave off underscores to test both macro forms.
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    // bool
    bool bool_;
    // string-likes
    std::string str_;
    std::filesystem::path path_;
    // vector
    std::vector<int> v_;

  public:
    BuiltIn(
        uint8_t u8,
        uint16_t u16,
        uint32_t u32,
        uint64_t u64,
        int8_t i8,
        int16_t i16,
        int32_t i32,
        int64_t i64,
        bool b,
        std::string str,
        std::filesystem::path path,
        std::vector<int> v
    )
        : u8_{u8}
        , u16_{u16}
        , u32_{u32}
        , u64_{u64}
        , i8{i8}
        , i16{i16}
        , i32{i32}
        , i64{i64}
        , bool_{b}
        , str_{std::move(str)}
        , path_{std::move(path)}
        , v_{std::move(v)} {}

    // The constructor here takes a boolean to select between two different to help with
    // testing.
    BuiltIn(bool path)
        : u8_(path ? 0 : 1)
        , u16_(path ? 2 : 3)
        , u32_(path ? 4 : 5)
        , u64_(path ? 6 : 7)
        , i8(path ? -1 : -2)
        , i16(path ? -3 : -4)
        , i32(path ? -5 : -6)
        , i64(path ? -7 : -8)
        , bool_{path}
        , str_{path ? "hello" : "world"}
        , path_{path ? "a/b/c" : "d/e/f"}
        , v_{path ? std::vector({1, 2, 3}) : std::vector({4, 5, 6})} {}

    // Comparison
    friend inline bool operator==(const BuiltIn&, const BuiltIn&) = default;

    // Saving and Loading
    svs::lib::SaveTable save() const {
        auto table = svs::lib::SaveTable(
            svs::lib::Version(0, 0, 0),
            {SVS_LIST_SAVE_(u8),
             SVS_LIST_SAVE_(u16),
             SVS_LIST_SAVE_(u32),
             SVS_LIST_SAVE_(u64),
             SVS_LIST_SAVE(i8),
             SVS_LIST_SAVE(i16),
             SVS_LIST_SAVE(i32),
             SVS_LIST_SAVE_(bool)}
        );
        SVS_INSERT_SAVE(table, i64);
        SVS_INSERT_SAVE_(table, str);
        SVS_INSERT_SAVE_(table, path);
        SVS_INSERT_SAVE_(table, v);
        return table;
    }

    static BuiltIn load(const toml::table& table, const svs::lib::Version& version) {
        CATCH_REQUIRE(version == svs::lib::Version(0, 0, 0));
        return BuiltIn{
            SVS_LOAD_MEMBER_AT_(table, u8),
            SVS_LOAD_MEMBER_AT_(table, u16),
            SVS_LOAD_MEMBER_AT_(table, u32),
            SVS_LOAD_MEMBER_AT_(table, u64),
            SVS_LOAD_MEMBER_AT(table, i8),
            SVS_LOAD_MEMBER_AT(table, i16),
            SVS_LOAD_MEMBER_AT(table, i32),
            SVS_LOAD_MEMBER_AT(table, i64),
            SVS_LOAD_MEMBER_AT_(table, bool),
            SVS_LOAD_MEMBER_AT_(table, str),
            SVS_LOAD_MEMBER_AT_(table, path),
            SVS_LOAD_MEMBER_AT_(table, v)};
    }
};

} // namespace

CATCH_TEST_CASE("Save/Load", "[lib][saveload]") {
    // Setup the temporary directory.
    auto setup = []() {
        svs_test::prepare_temp_directory();
        return svs_test::temp_directory();
    };

    CATCH_SECTION("Testing File Creation.") {
        auto temp_dir = setup();
        auto config_path = temp_dir / svs::lib::config_file_name;

        auto x = Saveable{10, "my_file.bin"};
        CATCH_REQUIRE(!std::filesystem::exists(config_path));
        svs::lib::save_to_disk(x, temp_dir);
        CATCH_REQUIRE(std::filesystem::exists(config_path));

        // Read the generated config file.
        auto table = toml::parse_file(config_path.c_str());
        auto subtable = svs::toml_helper::get_as<toml::table>(table, "object");
        CATCH_REQUIRE(svs::lib::load_at<int64_t>(subtable, "val") == 10);
        auto expected = std::string("my_file.bin");
        CATCH_REQUIRE(svs::lib::load_at<std::string>(subtable, "file") == expected);

        // Loading
        auto y = svs::lib::load_from_disk<Saveable>(temp_dir);
        CATCH_REQUIRE(y == x);

        // Make sure we can load from the full config file path as well.
        auto z = svs::lib::load_from_disk<Saveable>(config_path);
        CATCH_REQUIRE(z == x);

        // Do this process again, but now with the automatic saving path.
        CATCH_REQUIRE(svs::lib::test_self_save_load(x, temp_dir / "another_level"));
    }

    CATCH_SECTION("Saving and Loading. Contextual.") {
        auto temp_dir = setup();
        auto x = Saveable(123, "hello");
        CATCH_REQUIRE(x.extra_arg_ == false);

        // Test the equality operator.
        auto y = x;
        CATCH_REQUIRE(x == y);
        y.val_ += 1;
        CATCH_REQUIRE(x != y);
        y = x;
        CATCH_REQUIRE(x == y);
        y.extra_arg_ = true;
        CATCH_REQUIRE(x != y);

        // Test save and reload.
        svs::lib::save_to_disk(x, temp_dir);
        auto z = svs::lib::load_from_disk<Saveable>(temp_dir);
        CATCH_REQUIRE(z == x);

        // Test argument forwarding.
        z = svs::lib::load_from_disk<Saveable>(temp_dir, true);
        CATCH_REQUIRE(z != x);
        CATCH_REQUIRE(z.val_ == x.val_);
        CATCH_REQUIRE(z.extra_arg_ == true);
    }

    CATCH_SECTION("Saving and Loading. Context - Free") {
        auto temp_dir = setup();
        auto x = SaveableContextFree(1234);
        svs::lib::save_to_disk(x, temp_dir);

        // Saving and reloading.
        auto y = svs::lib::load_from_disk<SaveableContextFree>(temp_dir);
        CATCH_REQUIRE(x == y);

        // Test argument forwarding.
        y = svs::lib::load_from_disk<SaveableContextFree>(temp_dir, true);
        CATCH_REQUIRE(x != y);
        CATCH_REQUIRE(x.val_ == y.val_);
        CATCH_REQUIRE(y.extra_arg_);

        // Test now that we can round-trip through a TOML table correctly.
        auto table = svs::lib::save_to_table(x);
        y = svs::lib::load<SaveableContextFree>(table);
        CATCH_REQUIRE(x == y);

        // Argument forwarding
        y = svs::lib::load<SaveableContextFree>(table, true);
        CATCH_REQUIRE(x != y);
        CATCH_REQUIRE(x.val_ == y.val_);
        CATCH_REQUIRE(y.extra_arg_);
    }

    CATCH_SECTION("Saving and Loading. Style priority.") {
        auto temp_dir = setup();
        auto x = SaveableHasBoth();

        // Make sure that both calls to "save" have the correct side-effects.
        CATCH_REQUIRE(x.context_free_calls_ == 0);
        auto _ = x.save();
        CATCH_REQUIRE(x.context_free_calls_ == 1);

        CATCH_REQUIRE(x.contextual_calls_ == 0);
        _ = x.save(svs::lib::SaveContext(temp_dir));
        CATCH_REQUIRE(x.contextual_calls_ == 1);

        // Reset and now go through the full loading procedure.
        x = SaveableHasBoth();
        svs::lib::save_to_disk(x, temp_dir);
        CATCH_REQUIRE(x.context_free_calls_ == 1);
        CATCH_REQUIRE(x.contextual_calls_ == 0);

        // Loading should go through the context-free path.
        auto y = svs::lib::load_from_disk<SaveableHasBoth>(temp_dir);
        CATCH_REQUIRE(y.constructed_context_free_ == true);
        CATCH_REQUIRE(y.extra_arg_ == false);

        // Argument forwarding.
        y = svs::lib::load_from_disk<SaveableHasBoth>(temp_dir, true);
        CATCH_REQUIRE(y.constructed_context_free_ == true);
        CATCH_REQUIRE(y.extra_arg_ == true);

        // Serialization directly through a table.
        auto table = svs::lib::save_to_table(x);
        y = svs::lib::load<SaveableHasBoth>(table);
        CATCH_REQUIRE(y.constructed_context_free_ == true);
        CATCH_REQUIRE(y.extra_arg_ == false);

        y = svs::lib::load<SaveableHasBoth>(table, true);
        CATCH_REQUIRE(y.constructed_context_free_ == true);
        CATCH_REQUIRE(y.extra_arg_ == true);

        // Make sure we *can* load through the contextual path if we try hard enough.
        auto load_context = svs::lib::LoadContext(temp_dir, svs::lib::Version(0, 0, 0));
        y = SaveableHasBoth::load(table, load_context, SaveableHasBoth::save_version);
        CATCH_REQUIRE(y.constructed_context_free_ == false);
        CATCH_REQUIRE(y.constructed_with_context_ == true);
    }

    CATCH_SECTION("Saving And Loading Aggregates") {
        auto temp_dir = setup();
        auto x = Aggregate(10, "hello_world.bin", 32);
        svs::lib::save_to_disk(x, temp_dir);
        auto y = svs::lib::load_from_disk<Aggregate>(temp_dir);
        CATCH_REQUIRE(x == y);
        // Make sure the "SaveableHasBoth" member was saved correctly.
        CATCH_REQUIRE(x.c_.context_free_calls_ == 1);
        CATCH_REQUIRE(x.c_.contextual_calls_ == 0);
        CATCH_REQUIRE(y.c_.constructed_context_free_ == true);
        CATCH_REQUIRE(y.c_.constructed_with_context_ == false);
    }

    CATCH_SECTION("Save and Load Override. Context Free.") {
        size_t a = 10;
        size_t b = 20;
        auto version = svs::lib::Version(3, 6, 9);

        auto temp_dir = setup();
        svs::lib::save_to_disk(
            svs::lib::SaveOverride([&]() {
                return svs::lib::SaveTable(version, {SVS_LIST_SAVE(a), SVS_LIST_SAVE(b)});
            }),
            temp_dir
        );

        auto [a2, b2] = svs::lib::load_from_disk(
            svs::lib::LoadOverride([&](const toml::table& table,
                                       const svs::lib::Version& reloaded_version) {
                CATCH_REQUIRE(reloaded_version == version);
                auto a2 = svs::lib::load_at<size_t>(table, "a");
                auto b2 = svs::lib::load_at<size_t>(table, "b");
                return std::make_pair(a2, b2);
            }),
            temp_dir
        );
        CATCH_REQUIRE(a2 == a);
        CATCH_REQUIRE(b2 == b);
    }

    CATCH_SECTION("Save and Load Override. Contextual.") {
        size_t a = 10;
        size_t b = 20;
        auto version = svs::lib::Version(3, 6, 9);

        auto temp_dir = setup();
        svs::lib::save_to_disk(
            svs::lib::SaveOverride([&](const svs::lib::SaveContext& ctx) {
                auto file = ctx.generate_name("file", "bin");
                auto ostream = svs::lib::open_write(file);
                svs::lib::write_binary(ostream, a);
                return svs::lib::SaveTable(
                    version, {SVS_LIST_SAVE(b), {"file", svs::lib::save(file.filename())}}
                );
            }),
            temp_dir
        );

        auto [a2, b2] = svs::lib::load_from_disk(
            svs::lib::LoadOverride([&](const toml::table& table,
                                       const svs::lib::LoadContext& ctx,
                                       const svs::lib::Version& reloaded_version) {
                CATCH_REQUIRE(reloaded_version == version);

                const auto& filename =
                    svs::lib::load_at<std::filesystem::path>(table, "file");
                auto istream = svs::lib::open_read(ctx.get_directory() / filename);
                auto a2 = svs::lib::read_binary<size_t>(istream);
                // Test the non-underscore-appending macro.
                auto b2 = SVS_LOAD_MEMBER_AT(table, b);
                return std::make_pair(a2, b2);
            }),
            temp_dir
        );
        CATCH_REQUIRE(a2 == a);
        CATCH_REQUIRE(b2 == b);
    }

    CATCH_SECTION("Built-in Types") {
        auto test_true = BuiltIn{true};
        auto test_false = BuiltIn{false};
        CATCH_REQUIRE(test_true != test_false);
        CATCH_REQUIRE(test_true == svs::lib::load<BuiltIn>(svs::lib::save(test_true)));
        CATCH_REQUIRE(test_false == svs::lib::load<BuiltIn>(svs::lib::save(test_false)));

        auto temp_dir = setup();
        svs::lib::save_to_disk(test_true, temp_dir);
        CATCH_REQUIRE(test_true == svs::lib::load_from_disk<BuiltIn>(temp_dir));
    }

    CATCH_SECTION("Vector - context free") {
        auto tempdir = setup();
        auto v = std::vector<SaveableContextFree>(
            {SaveableContextFree(10, false), SaveableContextFree(20, false)}
        );

        auto tmp = svs::lib::save(v);
        using T = std::vector<SaveableContextFree>;
        auto allocator = typename T::allocator_type();

        // Save and load through a table.
        auto u = svs::lib::load<std::vector<SaveableContextFree>>(tmp);
        CATCH_REQUIRE(u == v);

        u = svs::lib::load<std::vector<SaveableContextFree>>(tmp, allocator);
        CATCH_REQUIRE(u == v);

        // Load through a table, giving an extra argument.
        u = svs::lib::load<std::vector<SaveableContextFree>>(tmp, true);
        CATCH_REQUIRE(u.at(0).extra_arg_ == true);
        CATCH_REQUIRE(u.at(1).extra_arg_ == true);

        u = svs::lib::load<std::vector<SaveableContextFree>>(tmp, allocator, true);
        CATCH_REQUIRE(u.at(0).extra_arg_ == true);
        CATCH_REQUIRE(u.at(1).extra_arg_ == true);

        // Save and load through disk.
        svs::lib::save_to_disk(v, tempdir);
        u = svs::lib::load_from_disk<std::vector<SaveableContextFree>>(tempdir);
        CATCH_REQUIRE(v == u);

        u = svs::lib::load_from_disk<std::vector<SaveableContextFree>>(tempdir, allocator);
        CATCH_REQUIRE(v == u);

        u = svs::lib::load_from_disk<std::vector<SaveableContextFree>>(tempdir, true);
        CATCH_REQUIRE(u.at(0).extra_arg_ == true);
        CATCH_REQUIRE(u.at(1).extra_arg_ == true);

        u = svs::lib::load_from_disk<std::vector<SaveableContextFree>>(
            tempdir, allocator, true
        );
        CATCH_REQUIRE(u.at(0).extra_arg_ == true);
        CATCH_REQUIRE(u.at(1).extra_arg_ == true);
    }

    CATCH_SECTION("Vector - contextual") {
        auto tempdir = setup();
        auto v = std::vector<Aggregate>(
            {Aggregate(10, "helloworld", -10), Aggregate(20, "foobar", 12)}
        );

        using T = std::vector<Aggregate>;
        auto allocator = std::allocator<Aggregate>();

        svs::lib::save_to_disk(v, tempdir);
        auto u = svs::lib::load_from_disk<T>(tempdir);
        CATCH_REQUIRE(u == v);

        u = svs::lib::load_from_disk<T>(tempdir, true);
        CATCH_REQUIRE(u != v);
        CATCH_REQUIRE(u.at(0).b_.extra_arg_ == true);
        CATCH_REQUIRE(u.at(1).b_.extra_arg_ == true);

        u = svs::lib::load_from_disk<T>(tempdir, allocator);
        CATCH_REQUIRE(u == v);

        u = svs::lib::load_from_disk<T>(tempdir, allocator, true);
        CATCH_REQUIRE(u != v);
        CATCH_REQUIRE(u.at(0).b_.extra_arg_ == true);
        CATCH_REQUIRE(u.at(1).b_.extra_arg_ == true);
    }

    CATCH_SECTION("Full Unsigned") {
        auto m = std::numeric_limits<uint64_t>::max();
        for (size_t x : std::array<size_t, 3>{0, m - 1, m}) {
            auto tmp = svs::lib::save(svs::lib::FullUnsigned(x));
            uint64_t u = svs::lib::load<svs::lib::FullUnsigned>(tmp);
            CATCH_REQUIRE(u == x);
        }
    }
}
