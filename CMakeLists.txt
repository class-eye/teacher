project(teacher)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include(mini-caffe/mini-caffe.cmake)
include(teacher/teacher.cmake)
file(GLOB SRC main/*.h* main/*.c*)
add_executable(main ${SRC})
target_link_libraries(main caffe ${OpenCV_LIBS} ${TEACHER_LIBRARY})
