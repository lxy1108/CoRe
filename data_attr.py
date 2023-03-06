from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

DATA_FOLDER = "./dataset"


@dataclass
class DatasetAttr:
    path: Path
    freq: str
    dimension: int
    instance_split: Optional[List]


dataset_dict = {"electricity": DatasetAttr(Path(DATA_FOLDER, "electricity/electricity.csv"), "h", 1, [257, 289, 321]),
                "etth": DatasetAttr(Path(DATA_FOLDER, "Etth.csv"), "h", 6, None),
                "etth1": DatasetAttr(Path(DATA_FOLDER, "ETT-small/ETTh1.csv"), "m", 7, None),
                "etth2": DatasetAttr(Path(DATA_FOLDER, "ETT-small/ETTh2.csv"), "m", 7, None),
                "ettm": DatasetAttr(Path(DATA_FOLDER, "ETTm.csv"), "h", 1, None),
                "ettm1": DatasetAttr(Path(DATA_FOLDER, "ETT-small/ETTm1.csv"), "m", 7, None),
                "ettm2": DatasetAttr(Path(DATA_FOLDER, "ETT-small/ETTm2.csv"), "m", 7, None),
                "exchange": DatasetAttr(Path(DATA_FOLDER, "exchange_rate/exchange_rate.csv"), "d", 1, [6, 7, 8]),
                "illness": DatasetAttr(Path(DATA_FOLDER, "illness/national_illness.csv"), "d", 7, None),
                "traffic": DatasetAttr(Path(DATA_FOLDER, "traffic/traffic.csv"), "h", 1, [690, 776, 862]),
                "weather": DatasetAttr(Path(DATA_FOLDER, "weather/weather.csv"), "h", 21, None)}