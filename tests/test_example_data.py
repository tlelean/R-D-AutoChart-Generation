import os
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "Rev02"))
from main import generate_program_pdf

EXAMPLE_ROOT = Path("Example Data/Hydraulic")


@pytest.mark.parametrize(
    "program_dir",
    [
        "Atmospheric Breakouts",
        "Dynamic Cycles Petrobras",
        "Dynamic Cycles PR2",
        "Holds",
        "Open-Close",
    ],
)
def test_generate_pdf(program_dir, tmp_path):
    data_dir = EXAMPLE_ROOT / program_dir
    primary = data_dir / "primary_data.csv"
    details = data_dir / "test_details.csv"
    output_path = tmp_path
    result = generate_program_pdf(str(primary), str(details), str(output_path), False)
    assert result is not None
    assert os.path.exists(result)
