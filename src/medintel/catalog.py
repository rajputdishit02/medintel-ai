"""Proposed components; this catalogue does not imply data have been acquired."""

from dataclasses import dataclass

CYCLE = "August 2021-August 2023"
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"


@dataclass(frozen=True)
class Component:
    code: str
    description: str

    @property
    def documentation_url(self) -> str:
        return f"{BASE_URL}/{self.code}.htm"

    @property
    def data_url(self) -> str:
        return f"{BASE_URL}/{self.code}.xpt"


COMPONENTS = (
    Component("DEMO_L", "Demographics and sample weights"),
    Component("BPXO_L", "Oscillometric blood pressure"),
    Component("BMX_L", "Body measures"),
    Component("TCHOL_L", "Total cholesterol"),
    Component("HDL_L", "HDL cholesterol"),
    Component("GHB_L", "Glycohemoglobin"),
    Component("BPQ_L", "Blood pressure and cholesterol questionnaire"),
    Component("MCQ_L", "Medical conditions and reported disease history"),
    Component("SMQ_L", "Cigarette use"),
)
