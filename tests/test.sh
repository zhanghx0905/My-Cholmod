# 用于简化测试流程的脚本
# builds
cd ../src
python3 setup.py build_ext -t ../tests -b ../tests
cd ../tests
make

python3 test_correstness.py
# python3 test_performance.py # 测试性能
make clean