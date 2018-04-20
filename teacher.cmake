include_directories(${CMAKE_CURRENT_LIST_DIR})
set(SRC ${CMAKE_CURRENT_LIST_DIR}/teacher.cpp
		${CMAKE_CURRENT_LIST_DIR}/pose.cpp 
		${CMAKE_CURRENT_LIST_DIR}/Timer.cpp 
		${CMAKE_CURRENT_LIST_DIR}/rfcn.cpp  
		${CMAKE_CURRENT_LIST_DIR}/jfda.cpp 
		${CMAKE_CURRENT_LIST_DIR}/teacher/teacher.hpp
		${CMAKE_CURRENT_LIST_DIR}/teacher/pose.hpp
		${CMAKE_CURRENT_LIST_DIR}/teacher/Timer.hpp
		${CMAKE_CURRENT_LIST_DIR}/teacher/rfcn.hpp
		${CMAKE_CURRENT_LIST_DIR}/teacher/jfda.hpp )
set(TEACHER_INCLUDE ${CMAKE_CURRENT_LIST_DIR})
set(TEACHER_LIBRARY teacher)

add_library(teacher STATIC ${SRC})
target_link_libraries(teacher ${OpenCV_LIBS})