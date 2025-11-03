set(CMAKE_CXX_STANDARD 14)

add_executable(main main.cpp)
target_link_libraries(main ${PROJECT_NAME})

find_library(AIO_LIB aio)
if (AIO_LIB)
    target_link_libraries(main PRIVATE ${AIO_LIB})
else()
    message(FATAL_ERROR "libaio not found")
endif()