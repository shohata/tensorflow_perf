# Locates the tensorflow library and include dirs.

include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(TensorFlow_INCLUDE_DIR
	NAMES
	tensorflow/core
	tensorflow/cc
	third_party
	PATHS
	/usr/include
	/usr/local/include)

find_library(TensorFlowCC_LIBRARY
    NAMES
    tensorflow_cc
	tensorflow_framework
	PATHS
	/usr/lib
	/usr/local/lib)

find_library(TensorFlowFW_LIBRARY
    NAMES
	tensorflow_framework
	PATHS
	/usr/lib
	/usr/local/lib)

# set Tensorflow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlowCC_LIBRARY TensorFlowFW_LIBRARY)

# set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND)
	set(TensorFlow_LIBRARIES ${TensorFlowCC_LIBRARY} ${TensorFlowFW_LIBRARY})
	set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
endif()

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlowCC_LIBRARY TensorFlowFW_LIBRARY)
