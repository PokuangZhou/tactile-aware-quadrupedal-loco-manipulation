from dataclasses import dataclass
from cyclonedds.idl import IdlStruct

@dataclass
class ArmString_(IdlStruct, typename="unitree_arm::msg::dds_::ArmString_"):
    data_: str