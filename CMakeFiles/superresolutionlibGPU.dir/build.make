# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /usr/prakt/p116/GPULabProj

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /usr/prakt/p116/GPULabProj

# Include any dependencies generated for this target.
include CMakeFiles/superresolutionlibGPU.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/superresolutionlibGPU.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/superresolutionlibGPU.dir/flags.make

CMakeFiles/superresolutionlibGPU.dir/src/superresolution/./superresolutionlibGPU_generated_superresolution.cu.o: CMakeFiles/superresolutionlibGPU.dir/src/superresolution/superresolutionlibGPU_generated_superresolution.cu.o.depend
CMakeFiles/superresolutionlibGPU.dir/src/superresolution/./superresolutionlibGPU_generated_superresolution.cu.o: CMakeFiles/superresolutionlibGPU.dir/src/superresolution/superresolutionlibGPU_generated_superresolution.cu.o.cmake
CMakeFiles/superresolutionlibGPU.dir/src/superresolution/./superresolutionlibGPU_generated_superresolution.cu.o: src/superresolution/superresolution.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /usr/prakt/p116/GPULabProj/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object CMakeFiles/superresolutionlibGPU.dir/src/superresolution/./superresolutionlibGPU_generated_superresolution.cu.o"
	cd /usr/prakt/p116/GPULabProj/CMakeFiles/superresolutionlibGPU.dir/src/superresolution && /usr/bin/cmake -E make_directory /usr/prakt/p116/GPULabProj/CMakeFiles/superresolutionlibGPU.dir/src/superresolution/.
	cd /usr/prakt/p116/GPULabProj/CMakeFiles/superresolutionlibGPU.dir/src/superresolution && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/usr/prakt/p116/GPULabProj/CMakeFiles/superresolutionlibGPU.dir/src/superresolution/./superresolutionlibGPU_generated_superresolution.cu.o -D generated_cubin_file:STRING=/usr/prakt/p116/GPULabProj/CMakeFiles/superresolutionlibGPU.dir/src/superresolution/./superresolutionlibGPU_generated_superresolution.cu.o.cubin.txt -P /usr/prakt/p116/GPULabProj/CMakeFiles/superresolutionlibGPU.dir/src/superresolution/superresolutionlibGPU_generated_superresolution.cu.o.cmake

# Object files for target superresolutionlibGPU
superresolutionlibGPU_OBJECTS =

# External object files for target superresolutionlibGPU
superresolutionlibGPU_EXTERNAL_OBJECTS = \
"/usr/prakt/p116/GPULabProj/CMakeFiles/superresolutionlibGPU.dir/src/superresolution/./superresolutionlibGPU_generated_superresolution.cu.o"

lib/libsuperresolutionlibGPU.so: /work/sdks/cudaversions/cuda50/cuda/lib64/libcudart.so
lib/libsuperresolutionlibGPU.so: /usr/lib/libcuda.so
lib/libsuperresolutionlibGPU.so: CMakeFiles/superresolutionlibGPU.dir/build.make
lib/libsuperresolutionlibGPU.so: CMakeFiles/superresolutionlibGPU.dir/src/superresolution/./superresolutionlibGPU_generated_superresolution.cu.o
lib/libsuperresolutionlibGPU.so: CMakeFiles/superresolutionlibGPU.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library lib/libsuperresolutionlibGPU.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/superresolutionlibGPU.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/superresolutionlibGPU.dir/build: lib/libsuperresolutionlibGPU.so
.PHONY : CMakeFiles/superresolutionlibGPU.dir/build

CMakeFiles/superresolutionlibGPU.dir/requires:
.PHONY : CMakeFiles/superresolutionlibGPU.dir/requires

CMakeFiles/superresolutionlibGPU.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/superresolutionlibGPU.dir/cmake_clean.cmake
.PHONY : CMakeFiles/superresolutionlibGPU.dir/clean

CMakeFiles/superresolutionlibGPU.dir/depend: CMakeFiles/superresolutionlibGPU.dir/src/superresolution/./superresolutionlibGPU_generated_superresolution.cu.o
	cd /usr/prakt/p116/GPULabProj && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /usr/prakt/p116/GPULabProj /usr/prakt/p116/GPULabProj /usr/prakt/p116/GPULabProj /usr/prakt/p116/GPULabProj /usr/prakt/p116/GPULabProj/CMakeFiles/superresolutionlibGPU.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/superresolutionlibGPU.dir/depend

