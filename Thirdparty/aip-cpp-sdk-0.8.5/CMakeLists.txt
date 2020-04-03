find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})
link_libraries(${JSONCPP_LIBRARIES})


find_package(CURL REQUIRED)


#find_package(crypto REQUIRED)

find_package(OpenSSL REQUIRED)
