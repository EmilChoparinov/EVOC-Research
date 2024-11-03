add_executable(multineat_xor
        main.cpp
        )

target_link_libraries(multineat_xor
        MultiNEAT
        $<$<BOOL:${WIN32}>:ws2_32>
        Boost::unit_test_framework
        )

add_test(xor multineat_xor)
