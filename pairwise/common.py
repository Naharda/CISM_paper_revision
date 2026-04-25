import enum


class Columns(str, enum.Enum):
    PATIENT_CLASS = "patient_class"
    PATIENT_CLASS_ID = "patient_class_id"
    PAIRWISE_COUNT = "pairwise_count"
