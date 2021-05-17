# 用于简化测试流程的脚本
# builds
cd ../src
python3 setup.py build_ext -t ../tests -b ../tests
cd ../tests
make

# pytest-3  # 测试正确性
# 测试性能
# python3 test_performance.py
# echo
# ./cholmod_c_test ./test_data/ted_B.mtx
# ./cholmod_c_test ./test_data/s3rmt3m3.mtx
# ./cholmod_c_test ./test_data/thermomech_dM.mtx
# ./cholmod_c_test ./test_data/parabolic_fem.mtx
make clean