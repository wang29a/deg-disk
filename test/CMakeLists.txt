set(CMAKE_CXX_STANDARD 17)

# add_executable(main main.cpp)
# target_link_libraries(main ${PROJECT_NAME})

add_executable(disk_build disk_build.cpp)
target_link_libraries(disk_build ${PROJECT_NAME})

add_executable(disk_search disk_search.cpp)
target_link_libraries(disk_search ${PROJECT_NAME})

find_library(AIO_LIB aio)
if (AIO_LIB)
    target_link_libraries(disk_search PRIVATE ${AIO_LIB})
    target_link_libraries(disk_build PRIVATE ${AIO_LIB})
else()
    message(FATAL_ERROR "libaio not found")
endif()