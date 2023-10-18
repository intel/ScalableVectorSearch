#include "svs/lib/saveload.h"

// svs
#include "svs/lib/readwrite.h"
#include "svs/third-party/toml.h" // included in saveload.h

// svsmain
#include "svsmain.h"

// third-party
#include "fmt/core.h"

// stl
#include <filesystem>
#include <numeric>
#include <vector>

namespace {

//! [context-free]
class ContextFreeSaveable {
    // Members
  private:
    int64_t a_;
    int64_t b_;

  public:
    ContextFreeSaveable(int64_t a, int64_t b)
        : a_{a}
        , b_{b} {}

    friend bool
    operator==(const ContextFreeSaveable&, const ContextFreeSaveable&) = default;

    // The version number used for saving and loading.
    // This can be used to detect and reload older versions of the data structure in a
    // backwards compatible way.
    static constexpr svs::lib::Version save_version = svs::lib::Version{0, 0, 1};

    // Save the object.
    svs::lib::SaveTable save() const;
    // Load the object.
    static ContextFreeSaveable load(const toml::table&, const svs::lib::Version&);
};
//! [context-free]

//! [context-free-saving]
// The `svs::lib::SaveTable` is a pair consisting of a `toml::table` and a version
// number.
svs::lib::SaveTable ContextFreeSaveable::save() const {
    return svs::lib::SaveTable(
        save_version, {{"a", svs::lib::save(a_)}, SVS_LIST_SAVE_(b)}
    );
}
//! [context-free-saving]

//! [context-free-loading]
// Loading takes the items produced by `save()` and should yield an instance of the
// associated class.
ContextFreeSaveable
ContextFreeSaveable::load(const toml::table& table, const svs::lib::Version& version) {
    // Perform a version check.
    // This class is only compatible with one version.
    if (version != save_version) {
        throw std::runtime_error("Version Mismatch!");
    }

    // Retrieve the saved values from the table.
    return ContextFreeSaveable(
        svs::lib::load_at<int64_t>(table, "a"), SVS_LOAD_MEMBER_AT_(table, b)
    );
}
//! [context-free-loading]

//! [contextual-loading]
class Saveable {
  private:
    // We have a member that is also a saveable object.
    ContextFreeSaveable member_;
    // The `data_` member may be arbitrarily long and is thus not necessarily suitable
    // for storage in a `toml::table`.
    std::vector<float> data_;

  public:
    Saveable(ContextFreeSaveable member, std::vector<float> data)
        : member_{member}
        , data_{std::move(data)} {}

    friend bool operator==(const Saveable&, const Saveable&) = default;

    static constexpr svs::lib::Version save_version = svs::lib::Version{0, 0, 1};

    // Contextual saving.
    svs::lib::SaveTable save(const svs::lib::SaveContext& ctx) const;
    // Contextual loading.
    static Saveable load(
        const toml::table& table,
        const svs::lib::LoadContext& ctx,
        const svs::lib::Version& version
    );
};
//! [contextual-loading]

//! [contextual-saving-impl]
svs::lib::SaveTable Saveable::save(const svs::lib::SaveContext& ctx) const {
    // Generate a unique name for the file where we will save the associated binary
    // data.
    //
    // This filename will be unique in the directory generated for saving this object.
    auto fullpath = ctx.generate_name("data", "bin");

    {
        // Open the file and store the contents of the vector into that file in a dense
        // binary form.
        auto ostream = svs::lib::open_write(fullpath);
        svs::lib::write_binary(ostream, data_);
    }

    // Generate a table to save the object.
    auto table = svs::lib::SaveTable(save_version);

    // Use `save` to save the sub-object into a sub-table.
    // Even though `ContextFreeSaveable` is context free, we can still pass the
    // context variable if desired.
    //
    // The library infrastructure will call the correct member function.
    SVS_INSERT_SAVE_(table, member, ctx);

    // Also store the size of the vector we're going to save.
    // Since integers of type `size_t` are not natively saveable in a
    // `toml::table`, we use the overload set `svs::prepare` to safely convert it.
    table.insert("data_size", svs::lib::save(data_.size()));

    // Store only the relative portion of the path to make the saved object
    // relocatable.
    //
    // Again, we need to use `svs::prepare` to convert `std::filesystem::path`
    // to a string-like type for the `toml::table`.
    table.insert("data_file", svs::lib::save(fullpath.filename()));
    return table;
}
//! [contextual-saving-impl]

//! [contextual-loading-impl]
Saveable Saveable::load(
    const toml::table& table,
    const svs::lib::LoadContext& ctx,
    const svs::lib::Version& version
) {
    // Check that the version matches.
    if (version != save_version) {
        throw std::runtime_error("Version Mismatch!");
    }

    // Obtain the file path and the size of the stored vector.
    auto relative_path = svs::lib::load_at<std::string>(table, "data_file");
    auto full_path = ctx.get_directory() / relative_path;
    auto data_size = svs::lib::load_at<size_t>(table, "data_size");

    // Allocate a sufficiently sized vector and
    auto data = std::vector<float>(data_size);
    {
        auto istream = svs::lib::open_read(full_path);
        svs::lib::read_binary(istream, data);
    }

    // Finish constructing the object by recursively loading the `member_` subobject.
    return Saveable(SVS_LOAD_MEMBER_AT_(table, member), std::move(data));
}
//! [contextual-loading-impl]

void demonstrate_context_free(const std::filesystem::path& dir) {
    //! [saving-and-reloading-context-free]
    // Construct an object, save it to disk, and reload it.
    auto context_free = ContextFreeSaveable(10, 20);
    auto context_free_reloaded =
        svs::lib::load<ContextFreeSaveable>(svs::lib::save(context_free));

    // Check that saving and reloading was successful
    if (context_free != context_free_reloaded) {
        throw ANNEXCEPTION("Context free reloading in-memory failed!");
    }

    // We also get saving and reloading from disk for free.
    svs::lib::save_to_disk(context_free, dir);
    context_free_reloaded = svs::lib::load_from_disk<ContextFreeSaveable>(dir);

    if (context_free != context_free_reloaded) {
        throw ANNEXCEPTION("Context free reloading to-disk failed!");
    }
    //! [saving-and-reloading-context-free]
}

void demonstrate_context_free_to_table() {
    //! [saving-to-table]
    // Construct an object, save it to a table and reload.
    auto context_free = ContextFreeSaveable(10, 20);
    auto table = svs::lib::save_to_table(context_free);
    auto context_free_reloaded = svs::lib::load<ContextFreeSaveable>(table);

    if (context_free != context_free_reloaded) {
        throw ANNEXCEPTION("Context free reloading failed!");
    }
    //! [saving-to-table]
}

void demonstrate_context_required(const std::filesystem::path& dir) {
    //! [saving-and-reloading-contextual]
    // Initialize the data vector.
    auto data = std::vector<float>(100);
    std::iota(data.begin(), data.end(), 10);

    // Construct, save, and reload.
    auto context_required = Saveable(ContextFreeSaveable(20, 30), std::move(data));
    svs::lib::save_to_disk(context_required, dir);
    auto context_required_reloaded = svs::lib::load_from_disk<Saveable>(dir);

    // Ensure that the reloaded object equals the original.
    if (context_required != context_required_reloaded) {
        throw ANNEXCEPTION("Context required reloading failed!");
    }
    //! [saving-and-reloading-contextual]
}

void demonstrate_vector(const std::filesystem::path& dir) {
    //! [saving-and-reloading-vector]
    auto data = std::vector<Saveable>(
        {Saveable(ContextFreeSaveable(10, 20), {1, 2, 3}),
         Saveable(ContextFreeSaveable(30, 40), {4, 5, 6})}
    );

    svs::lib::save_to_disk(data, dir);
    auto reloaded = svs::lib::load_from_disk<std::vector<Saveable>>(dir);
    if (reloaded != data) {
        throw ANNEXCEPTION("Reloading vector failed!");
    }
    //! [saving-and-reloading-vector]
}

} // namespace

bool directory_is_not_empty(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return false;
    }

    auto itr = std::filesystem::directory_iterator(path);
    return begin(itr) != end(itr);
}

int svs_main(std::vector<std::string> args) {
    // Default arguments.
    auto dir = std::filesystem::path("temp");
    auto nargs = args.size();

    if (nargs == 2) {
        dir = args.at(1);
    } else if (nargs > 2) {
        throw ANNEXCEPTION("Unexpected number of arguments");
    }

    if (directory_is_not_empty(dir)) {
        throw ANNEXCEPTION("Directory ", dir, " is not empty!");
    }

    demonstrate_context_free(dir);
    demonstrate_context_free_to_table();

    // Cleanup temporary directory
    std::filesystem::remove_all(dir);
    demonstrate_context_required(dir);

    // Cleanup temporary_directory
    std::filesystem::remove_all(dir);
    demonstrate_vector(dir);

    return 0;
}

SVS_DEFINE_MAIN();
