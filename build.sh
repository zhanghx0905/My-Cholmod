# 用于简化测试流程的脚本
# builds
cd src
python3 setup.py build_ext -b ../tests -t ../tests
cd ../tests
make
# "test correctness"
pytest-3
# "test performance"
python3 test_performance.py
echo
./cholmod_c_test ./test_data/ted_B.mtx
./cholmod_c_test ./test_data/s3rmt3m3.mtx

make clean