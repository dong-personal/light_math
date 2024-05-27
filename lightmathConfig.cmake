
# 创建 INTERFACE 库
add_library(lightmath INTERFACE)

# 设置这个库的包含目录
set_target_properties(lightmath PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR})
