cmake_minimum_required(VERSION 3.31.1)
project( yet_anpther_app )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package( Torch REQUIRED )
find_package( OpenCV 4.10.0 REQUIRED )
find_package( onnxruntime REQUIRED )
include_directories(  
  "/usr/local/include/onnxruntime"  
  ${OpenCV_INCLUDE_DIRS}
)
link_directories( ${OpenCV_LIB_DIR} )
add_executable( yet_anpther_app src/main.cpp )
target_link_libraries(
  yet_anpther_app
  "/usr/local/lib/libonnxruntime.so"
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
)
set_property(TARGET yet_anpther_app PROPERTY CXX_STANDARD 20)


# set(ONNXRUNTIME_LIBRARIES 
#     "onnxruntime.lib" 
# )


# target_link_directories(myproject PUBLIC "${ONNXRUNTIME_ROOTDIR}/lib")

# target_link_libraries(myproject PUBLIC 
#     ${ONNXRUNTIME_LIBRARIES} 
# )