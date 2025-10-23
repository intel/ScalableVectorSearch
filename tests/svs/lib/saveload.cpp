/*
 * Copyright 2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// header under test
#include "svs/lib/saveload.h"

// svs
#include "svs/lib/file.h"

// test helpers
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

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
    static constexpr std::string_view serialization_schema = "svstest_saveable";

    // Methods
    svs::lib::SaveTable save(const svs::lib::SaveContext& ctx) const {
        auto my_path = ctx.get_directory() / file_;
        auto stream = svs::lib::open_write(my_path);
        svs::lib::write_binary(stream, val_);
        return svs::lib::SaveTable(
            serialization_schema, save_version, {SVS_LIST_SAVE_(val), SVS_LIST_SAVE_(file)}
        );
    }

    static Saveable load(const svs::lib::LoadTable& table, bool extra_arg = false) {
        CATCH_REQUIRE(table.version() == save_version);
        CATCH_REQUIRE(table.schema() == serialization_schema);
        auto val_from_table = SVS_LOAD_MEMBER_AT_(table, val);
        auto file = SVS_LOAD_MEMBER_AT_(table, file);

        // Make sure any side-effects were saved correctly.
        auto path = table.resolve(file);
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
    bool old_version_ = false;
    bool old_schema_ = false;

  public:
    SaveableContextFree(int32_t val, bool extra_arg)
        : val_{val}
        , extra_arg_{extra_arg} {}
    SaveableContextFree(int32_t val)
        : val_{val} {}

    // Provide the expected overloads for default compatibility checking but *also* define
    // a manual compatibility check.
    static constexpr std::string_view serialization_schema =
        "svstest_saveable_context_free";
    static constexpr std::string_view backup_schema = "svstest_backup";
    static constexpr svs::lib::Version save_version{1, 2, 3};

    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            "svstest_saveable_context_free",
            svs::lib::Version(1, 2, 3),
            {SVS_LIST_SAVE_(val)}
        );
    }

    static bool
    check_load_compatibility(std::string_view schema, svs::lib::Version version) {
        bool schema_matches = schema == serialization_schema || schema == backup_schema;
        return schema_matches && version <= save_version;
    }

    static SaveableContextFree
    load(const svs::lib::ContextFreeLoadTable& table, bool extra_arg = false) {
        auto val = SaveableContextFree(SVS_LOAD_MEMBER_AT_(table, val), extra_arg);
        if (table.version() < save_version) {
            val.old_version_ = true;
        }
        if (table.schema() == backup_schema) {
            val.old_schema_ = true;
        }
        return val;
    }

    static svs::lib::TryLoadResult<SaveableContextFree> try_load(
        const svs::lib::ContextFreeLoadTable& table,
        bool extra_arg = false,
        bool auto_fail = false
    ) {
        if (auto_fail) {
            return svs::lib::Unexpected(svs::lib::TryLoadFailureReason::Other);
        }
        return load(table, extra_arg);
    }

    friend bool
    operator==(const SaveableContextFree&, const SaveableContextFree&) = default;
};

void change_reserved_field(
    toml::table& table, std::string_view key, std::string_view value
) {
    CATCH_REQUIRE(table.contains(key));
    CATCH_REQUIRE(table.erase(key) == 1);
    table.insert(key, value);
}

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
    static constexpr std::string_view serialization_schema = "svstest_saveable_has_both";

    // Implement the context-free version and the contextual version.
    // Just return an empty table for both.
    svs::lib::SaveTable save() const {
        context_free_calls_ += 1;
        return svs::lib::SaveTable(serialization_schema, save_version);
    }

    // Contextual version
    svs::lib::SaveTable save(const svs::lib::SaveContext& SVS_UNUSED(ctx)) const {
        contextual_calls_ += 1;
        return svs::lib::SaveTable(serialization_schema, save_version);
    }

    // Context free load
    static SaveableHasBoth
    load(const svs::lib::ContextFreeLoadTable& table, bool extra_arg = false) {
        CATCH_REQUIRE(table.version() == save_version);
        CATCH_REQUIRE(table.schema() == serialization_schema);
        auto x = SaveableHasBoth();
        x.constructed_context_free_ = true;
        x.extra_arg_ = extra_arg;
        return x;
    }

    // Contextual load
    static SaveableHasBoth load(const svs::lib::LoadTable& table) {
        CATCH_REQUIRE(table.version() == save_version);
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
            "svstest_aggregate",
            svs::lib::Version(0, 0, 0),
            {SVS_LIST_SAVE_(a, ctx), SVS_LIST_SAVE_(b, ctx)}
        );
        // Test emplacement as well.
        SVS_INSERT_SAVE_(table, c, ctx);
        return table;
    }

    static bool
    check_load_compatibility(std::string_view schema, svs::lib::Version version) {
        return schema == "svstest_aggregate" && version == svs::lib::Version(0, 0, 0);
    }

    static Aggregate load(const svs::lib::LoadTable& table, bool extra_arg = false) {
        CATCH_REQUIRE(table.version() == svs::lib::Version(0, 0, 0));
        CATCH_REQUIRE(table.schema() == "svstest_aggregate");

        return Aggregate(
            SVS_LOAD_MEMBER_AT_(table, a),
            SVS_LOAD_MEMBER_AT_(table, b, extra_arg),
            SVS_LOAD_MEMBER_AT_(table, c)
        );
    }

    static svs::lib::TryLoadResult<Aggregate> try_load(
        const svs::lib::LoadTable& table, bool extra_arg = false, bool auto_fail_b = false
    ) {
        CATCH_REQUIRE(table.version() == svs::lib::Version(0, 0, 0));
        CATCH_REQUIRE(table.schema() == "svstest_aggregate");

        // Load a sub-member that can fail.
        auto ex =
            svs::lib::try_load_at<SaveableContextFree>(table, "b", extra_arg, auto_fail_b);
        if (!ex) {
            return svs::lib::Unexpected(ex.error());
        }
        return Aggregate(
            SVS_LOAD_MEMBER_AT_(table, a),
            std::move(ex).value(),
            SVS_LOAD_MEMBER_AT_(table, c)
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

    static constexpr std::string_view serialization_schema = "svstest_buildin";
    static constexpr svs::lib::Version save_version = svs::lib::Version(0, 0, 0);

    // Saving and Loading
    svs::lib::SaveTable save() const {
        auto table = svs::lib::SaveTable(
            serialization_schema,
            save_version,
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

    static BuiltIn load(const svs::lib::ContextFreeLoadTable& table) {
        CATCH_REQUIRE(table.version() == svs::lib::Version(0, 0, 0));
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
            SVS_LOAD_MEMBER_AT_(table, v)
        };
    }

    static svs::lib::TryLoadResult<BuiltIn>
    try_load(const svs::lib::ContextFreeLoadTable& table, bool auto_fail = false) {
        // Schema check performed by the saving and loading infrastructure.
        if (auto_fail) {
            return svs::lib::Unexpected(svs::lib::TryLoadFailureReason::Other);
        }
        return load(table);
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
        auto table =
            svs::lib::ContextFreeSerializedObject(toml::parse_file(config_path.c_str()));
        auto object = table.object().template cast<toml::table>();

        // auto subtable =
        //     svs::lib::LoadTable(svs::toml_helper::get_as<toml::table>(table, "object"));
        CATCH_REQUIRE(svs::lib::load_at<int64_t>(object, "val") == 10);
        auto expected = std::string("my_file.bin");
        CATCH_REQUIRE(svs::lib::load_at<std::string>(object, "file") == expected);

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

        // Make sure we get an error if we reload with incorrect values.
        {
            svs::lib::save_to_disk(x, temp_dir);
            auto file = temp_dir / svs::lib::config_file_name;
            auto t = toml::parse_file(file.c_str());
            change_reserved_field(*t["object"].as_table(), "__version__", "v500.500.500");
            {
                auto io = svs::lib::open_write(file);
                io << t << '\n';
            }

            CATCH_REQUIRE_THROWS_MATCHES(
                svs::lib::load_from_disk<Saveable>(temp_dir),
                svs::ANNException,
                svs_test::ExceptionMatcher(
                    Catch::Matchers::ContainsSubstring(
                        "Trying to deserialize incompatible object"
                    ) &&
                    Catch::Matchers::ContainsSubstring("v500.500.500")

                )
            );
        }

        {
            svs::lib::save_to_disk(x, temp_dir);
            auto file = temp_dir / svs::lib::config_file_name;
            auto t = toml::parse_file(file.c_str());
            change_reserved_field(*t["object"].as_table(), "__schema__", "bad_schema");
            {
                auto io = svs::lib::open_write(file);
                io << t << '\n';
            }

            CATCH_REQUIRE_THROWS_MATCHES(
                svs::lib::load_from_disk<Saveable>(temp_dir),
                svs::ANNException,
                svs_test::ExceptionMatcher(
                    Catch::Matchers::ContainsSubstring(
                        "Trying to deserialize incompatible object"
                    ) &&
                    Catch::Matchers::ContainsSubstring("bad_schema")

                )
            );
        }
    }

    CATCH_SECTION("Saving and Loading. Context - Free") {
        auto temp_dir = setup();
        auto x = SaveableContextFree(1234);
        svs::lib::save_to_disk(x, temp_dir);

        // Saving and reloading.
        auto y = svs::lib::load_from_disk<SaveableContextFree>(temp_dir);
        CATCH_REQUIRE(x == y);

        // go directly through files.
        auto temp_file = temp_dir / "my_file.toml";
        svs::lib::save_to_file(x, temp_file);
        y = svs::lib::load_from_file<SaveableContextFree>(temp_file);
        CATCH_REQUIRE(x == y);

        // Test argument forwarding.
        y = svs::lib::load_from_disk<SaveableContextFree>(temp_dir, true);
        CATCH_REQUIRE(x != y);
        CATCH_REQUIRE(x.val_ == y.val_);
        CATCH_REQUIRE(y.extra_arg_);

        // Argument forwarding through files.
        y = svs::lib::load_from_file<SaveableContextFree>(temp_file, true);
        CATCH_REQUIRE(x != y);
        CATCH_REQUIRE(x.val_ == y.val_);
        CATCH_REQUIRE(y.extra_arg_);

        // Test now that we can round-trip through a TOML table correctly.
        auto table = svs::lib::save_to_table(x);
        y = svs::lib::load<SaveableContextFree>(svs::lib::node_view(table));
        CATCH_REQUIRE(x == y);

        // Argument forwarding
        y = svs::lib::load<SaveableContextFree>(svs::lib::node_view(table), true);
        CATCH_REQUIRE(x != y);
        CATCH_REQUIRE(x.val_ == y.val_);
        CATCH_REQUIRE(y.extra_arg_);

        // Test compatibility checks.
        {
            auto t = svs::lib::save_to_table(x);
            change_reserved_field(t, "__version__", "v0.0.0");

            auto z = svs::lib::load<SaveableContextFree>(svs::lib::node_view(t));
            CATCH_REQUIRE(z.old_version_);
            CATCH_REQUIRE(!z.old_schema_);
        }

        {
            auto t = svs::lib::save_to_table(x);
            change_reserved_field(t, "__schema__", SaveableContextFree::backup_schema);
            auto z = svs::lib::load<SaveableContextFree>(svs::lib::node_view(t));
            CATCH_REQUIRE(!z.old_version_);
            CATCH_REQUIRE(z.old_schema_);
        }

        // Make sure we get an error if we reload with incorrect values.
        {
            auto t = svs::lib::save_to_table(x);
            change_reserved_field(t, "__version__", "v500.500.500");
            CATCH_REQUIRE_THROWS_MATCHES(
                svs::lib::load<SaveableContextFree>(svs::lib::node_view(t)),
                svs::ANNException,
                svs_test::ExceptionMatcher(
                    Catch::Matchers::ContainsSubstring(
                        "Trying to deserialize incompatible object"
                    ) &&
                    Catch::Matchers::ContainsSubstring("v500.500.500")

                )
            );
        }

        {
            auto t = svs::lib::save_to_table(x);
            change_reserved_field(t, "__schema__", "bad_schema");
            CATCH_REQUIRE_THROWS_MATCHES(
                svs::lib::load<SaveableContextFree>(svs::lib::node_view(t)),
                svs::ANNException,
                svs_test::ExceptionMatcher(
                    Catch::Matchers::ContainsSubstring(
                        "Trying to deserialize incompatible object"
                    ) &&
                    Catch::Matchers::ContainsSubstring("bad_schema")

                )
            );
        }

        // try-load
        {
            auto z = svs::lib::try_load_from_disk<SaveableContextFree>(temp_dir);
            CATCH_REQUIRE(z);
            CATCH_REQUIRE(z.value() == x);

            // argument forwarding.
            z = svs::lib::try_load_from_disk<SaveableContextFree>(temp_dir, true);
            CATCH_REQUIRE(z);
            CATCH_REQUIRE(z.value() != x);
            CATCH_REQUIRE(z.value().val_ == x.val_);
            CATCH_REQUIRE(z.value().extra_arg_);

            // auto-failure.
            z = svs::lib::try_load_from_disk<SaveableContextFree>(temp_dir, true, true);
            CATCH_REQUIRE(!z);

            // Change serialized schema - ensure that the compatibility check fails.
            auto src = temp_dir / svs::lib::config_file_name;
            auto dst = temp_dir / "modified_config.toml";
            svs_test::mutate_table(
                src, dst, {{{"object", "__schema__"}, "invalid_schema"}}
            );

            // Make sure loading directly from the source file works.
            z = svs::lib::try_load_from_disk<SaveableContextFree>(src);
            CATCH_REQUIRE(z);
            CATCH_REQUIRE(z.value() == x);

            // Loading from the modified file should fail due to invalid schema.
            z = svs::lib::try_load_from_disk<SaveableContextFree>(dst);
            CATCH_REQUIRE(!z);
            CATCH_REQUIRE(z.error() == svs::lib::TryLoadFailureReason::InvalidSchema);

            // now - mutate the version instead of the schema.
            svs_test::mutate_table(src, dst, {{{"object", "__version__"}, "v20.1.2"}});
            z = svs::lib::try_load_from_disk<SaveableContextFree>(dst);
            CATCH_REQUIRE(!z);
            CATCH_REQUIRE(z.error() == svs::lib::TryLoadFailureReason::InvalidSchema);

            // Modify the underlying value just to double check.
            svs_test::mutate_table(src, dst, {{{"object", "val"}, 20}});
            z = svs::lib::try_load_from_disk<SaveableContextFree>(dst);
            CATCH_REQUIRE(z);
            CATCH_REQUIRE(z.value().val_ == 20);
        }
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

        // Loading should go through the contextual path.
        auto y = svs::lib::load_from_disk<SaveableHasBoth>(temp_dir);
        CATCH_REQUIRE(y.constructed_context_free_ == false);
        CATCH_REQUIRE(y.extra_arg_ == false);

        // Argument forwarding - overload should pick the method that takes an additional
        // argument.
        y = svs::lib::load_from_disk<SaveableHasBoth>(temp_dir, true);
        CATCH_REQUIRE(y.constructed_context_free_ == true);
        CATCH_REQUIRE(y.extra_arg_ == true);

        // Serialization directly through a table - should take the context-free path.
        auto table = svs::lib::save_to_table(x);
        y = svs::lib::load<SaveableHasBoth>(svs::lib::node_view(table));
        CATCH_REQUIRE(y.constructed_context_free_ == true);
        CATCH_REQUIRE(y.extra_arg_ == false);

        y = svs::lib::load<SaveableHasBoth>(svs::lib::node_view(table), true);
        CATCH_REQUIRE(y.constructed_context_free_ == true);
        CATCH_REQUIRE(y.extra_arg_ == true);

        // Make sure we *can* load through the contextual path if we try hard enough.
        auto load_context = svs::lib::LoadContext(temp_dir, svs::lib::Version(0, 0, 0));
        y = SaveableHasBoth::load(svs::lib::node_view(table, load_context));
        CATCH_REQUIRE(y.constructed_context_free_ == false);
        CATCH_REQUIRE(y.constructed_with_context_ == true);
    }

    CATCH_SECTION("Saving And Loading Aggregates") {
        auto temp_dir = setup();
        auto x = Aggregate(10, "hello_world.bin", 32);
        svs::lib::save_to_disk(x, temp_dir);
        CATCH_SECTION("Standard Loading") {
            auto y = svs::lib::load_from_disk<Aggregate>(temp_dir);
            CATCH_REQUIRE(x == y);
            // Make sure the "SaveableHasBoth" member was saved correctly.
            CATCH_REQUIRE(x.c_.context_free_calls_ == 1);
            CATCH_REQUIRE(x.c_.contextual_calls_ == 0);
            CATCH_REQUIRE(y.c_.constructed_context_free_ == false);
            CATCH_REQUIRE(y.c_.constructed_with_context_ == true);
        }
        CATCH_SECTION("Try Loading") {
            auto y = svs::lib::try_load_from_disk<Aggregate>(temp_dir);
            CATCH_REQUIRE(y);
            CATCH_REQUIRE(x == y.value());

            auto src = temp_dir / svs::lib::config_file_name;
            auto dst = temp_dir / "modified.toml";

            // Modify the schema of a deep object - should result in a failed try-load.
            svs_test::mutate_table(
                src, dst, {{{"object", "b", "__schema__"}, "invalid_schema"}}
            );
            y = svs::lib::try_load_from_disk<Aggregate>(dst);
            CATCH_REQUIRE(!y);
            CATCH_REQUIRE(y.error() == svs::lib::TryLoadFailureReason::InvalidSchema);

            // Error via argument forwarding.
            y = svs::lib::try_load_from_disk<Aggregate>(temp_dir, true, true);
            CATCH_REQUIRE(!y);
            CATCH_REQUIRE(y.error() == svs::lib::TryLoadFailureReason::Other);
        }
    }

    CATCH_SECTION("Built-in Types") {
        auto test_true = BuiltIn{true};
        auto test_false = BuiltIn{false};
        CATCH_REQUIRE(test_true != test_false);
        CATCH_REQUIRE(
            test_true ==
            svs::lib::load<BuiltIn>(svs::lib::node_view(svs::lib::save(test_true)))
        );
        CATCH_REQUIRE(
            test_false ==
            svs::lib::load<BuiltIn>(svs::lib::node_view(svs::lib::save(test_false)))
        );

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
        auto u = svs::lib::load<std::vector<SaveableContextFree>>(svs::lib::node_view(tmp));
        CATCH_REQUIRE(u == v);

        u = svs::lib::load<std::vector<SaveableContextFree>>(
            svs::lib::node_view(tmp), allocator
        );
        CATCH_REQUIRE(u == v);

        // Load through a table, giving an extra argument.
        u = svs::lib::load<std::vector<SaveableContextFree>>(
            svs::lib::node_view(tmp), true
        );
        CATCH_REQUIRE(u.at(0).extra_arg_ == true);
        CATCH_REQUIRE(u.at(1).extra_arg_ == true);

        u = svs::lib::load<std::vector<SaveableContextFree>>(
            svs::lib::node_view(tmp), allocator, true
        );
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

        // Save and load through file.
        auto tempfile = tempdir / "temp_file.toml";
        svs::lib::save_to_file(v, tempfile);
        u = svs::lib::load_from_file<std::vector<SaveableContextFree>>(tempfile);
        CATCH_REQUIRE(v == u);

        u = svs::lib::load_from_file<std::vector<SaveableContextFree>>(tempfile, allocator);
        CATCH_REQUIRE(v == u);

        u = svs::lib::load_from_file<std::vector<SaveableContextFree>>(tempfile, true);
        CATCH_REQUIRE(u.at(0).extra_arg_ == true);
        CATCH_REQUIRE(u.at(1).extra_arg_ == true);

        u = svs::lib::load_from_file<std::vector<SaveableContextFree>>(
            tempfile, allocator, true
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
            uint64_t u = svs::lib::load<svs::lib::FullUnsigned>(svs::lib::node_view(tmp));
            CATCH_REQUIRE(u == x);
        }
    }

    CATCH_SECTION("Percent") {
        auto temp_dir = setup();
        auto temp_file = temp_dir / "temp.toml";
        auto x = svs::lib::Percent(0.125);
        svs::lib::save_to_file(x, temp_file);
        auto y = svs::lib::load_from_file<svs::lib::Percent>(temp_file);
        CATCH_REQUIRE(x == y);
    }

    CATCH_SECTION("Binary Blob") {
        auto temp_dir = setup();
        auto v = std::vector<int>{1, 2, 3, 4, 5};
        svs::lib::save_to_disk(svs::lib::BinaryBlobSaver(v), temp_dir);

        // Reload.
        std::vector<int> u =
            svs::lib::load_from_disk<svs::lib::BinaryBlobLoader<int>>(temp_dir);
        CATCH_REQUIRE(u == v);

        // Reload with a different allocator.
        std::vector<int, svs::lib::Allocator<int>> w = svs::lib::load_from_disk<
            svs::lib::BinaryBlobLoader<int, svs::lib::Allocator<int>>>(
            temp_dir, svs::lib::Allocator<int>()
        );

        CATCH_REQUIRE(w.size() == v.size());
        CATCH_REQUIRE(std::equal(w.begin(), w.end(), v.begin()));
    }
}
