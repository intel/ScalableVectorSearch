file(REMOVE_RECURSE
  "libsvs_x86_objects.a"
  "libsvs_x86_objects.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/svs_x86_objects.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
