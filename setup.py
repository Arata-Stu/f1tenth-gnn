from setuptools import setup, Extension
import numpy

module = Extension(
    # 拡張モジュール名は import 時の名前に合わせます
    'lidar_graph',
    sources=['src/data/graph/lidar_graph.c'],
    include_dirs=[numpy.get_include()],  # ここで NumPy ヘッダの場所を指定
)

setup(
    name='lidar_graph',
    version='1.0',
    description='Lidar Graph C extension',
    ext_modules=[module],
)
