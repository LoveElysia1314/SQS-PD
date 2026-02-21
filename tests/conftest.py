"""
pytest配置和fixtures
====================
提供测试基础设施和常用测试数据
"""

import pytest
import sys
import os
import types
import math
from pathlib import Path
from fractions import Fraction
from pymatgen.core import Structure, Lattice

# 统一添加 src 根路径，确保测试使用包名 `sqs_pd` 导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# 检测运行模式
LEARNING_MODE = os.environ.get("LEARNING_MODE", "0") == "1"


@pytest.fixture
def fcc_lattice():
    """FCC晶格 3.0Å"""
    return Lattice.cubic(3.0)


@pytest.fixture
def binary_structure(fcc_lattice):
    """二元合金: 0.5Ni + 0.5Cu"""
    species = [{"Ni": 0.5, "Cu": 0.5}]
    coords = [[0, 0, 0]]
    return Structure(fcc_lattice, species, coords)


@pytest.fixture
def demo_pd_cif():
    """PD示例文件路径"""
    return Path(__file__).parent.parent / "data" / "input" / "demo_pd.cif"


@pytest.fixture
def demo_sd_cif():
    """SD示例文件路径"""
    return Path(__file__).parent.parent / "data" / "input" / "demo_sd.cif"


def print_section(title: str):
    """打印章节标题"""
    if LEARNING_MODE:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")


def print_concept(content: str):
    """打印概念说明"""
    if LEARNING_MODE:
        print(f"\n💡 {content}")


def print_code_example(code: str):
    """打印代码示例"""
    if LEARNING_MODE:
        print(f"\n📝 代码示例:")
        for line in code.strip().split("\n"):
            print(f"   {line}")
