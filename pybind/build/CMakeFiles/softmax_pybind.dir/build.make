# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kedi/Desktop/a3/pybind

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kedi/Desktop/a3/pybind/build

# Include any dependencies generated for this target.
include CMakeFiles/softmax_pybind.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/softmax_pybind.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/softmax_pybind.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/softmax_pybind.dir/flags.make

CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.o: CMakeFiles/softmax_pybind.dir/flags.make
CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.o: ../pybind_kernel_cuda.cpp
CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.o: CMakeFiles/softmax_pybind.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kedi/Desktop/a3/pybind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.o -MF CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.o.d -o CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.o -c /home/kedi/Desktop/a3/pybind/pybind_kernel_cuda.cpp

CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kedi/Desktop/a3/pybind/pybind_kernel_cuda.cpp > CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.i

CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kedi/Desktop/a3/pybind/pybind_kernel_cuda.cpp -o CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.s

# Object files for target softmax_pybind
softmax_pybind_OBJECTS = \
"CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.o"

# External object files for target softmax_pybind
softmax_pybind_EXTERNAL_OBJECTS =

softmax_module.cpython-310-x86_64-linux-gnu.so: CMakeFiles/softmax_pybind.dir/pybind_kernel_cuda.cpp.o
softmax_module.cpython-310-x86_64-linux-gnu.so: CMakeFiles/softmax_pybind.dir/build.make
softmax_module.cpython-310-x86_64-linux-gnu.so: CMakeFiles/softmax_pybind.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kedi/Desktop/a3/pybind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module softmax_module.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/softmax_pybind.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/softmax_pybind.dir/build: softmax_module.cpython-310-x86_64-linux-gnu.so
.PHONY : CMakeFiles/softmax_pybind.dir/build

CMakeFiles/softmax_pybind.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/softmax_pybind.dir/cmake_clean.cmake
.PHONY : CMakeFiles/softmax_pybind.dir/clean

CMakeFiles/softmax_pybind.dir/depend:
	cd /home/kedi/Desktop/a3/pybind/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kedi/Desktop/a3/pybind /home/kedi/Desktop/a3/pybind /home/kedi/Desktop/a3/pybind/build /home/kedi/Desktop/a3/pybind/build /home/kedi/Desktop/a3/pybind/build/CMakeFiles/softmax_pybind.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/softmax_pybind.dir/depend
