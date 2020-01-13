set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(GMM GMM.cpp)
target_link_libraries(GMM ${OpenCV_LIBS})
target_compile_options(GMM PUBLIC
    -std=c++11
    -Werror
    # -fsanitize=address
    # -fno-omit-frame-pointer
)