# CMakeList.txt : CMake project for cdftest, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.11) # FetchContent is new in version 3.11.

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("cdftest")

# Add source to this project's executable.
add_executable (cdftest "cdftest.cpp" "cdftest.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET cdftest PROPERTY CXX_STANDARD 20)
endif()

include(FetchContent)

#FetchContent_Declare(
#	glm
#	GIT_REPOSITORY	https://github.com/g-truc/glm.git
#	GIT_TAG 	bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
#)
#FetchContent_MakeAvailable(glm)
#target_link_libraries(cdftest PRIVATE glm::glm)

if(MSVC)
  # make msvc produce an assembly listing
  target_compile_options(cdftest PUBLIC "/FAs")
else()
  # make gcc compile for current architecture
  target_compile_options(cdftest PUBLIC "-march=native")
#  target_compile_options(cdftest PUBLIC "-g")
  target_compile_options(cdftest PUBLIC "-O3")
endif()
