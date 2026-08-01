CLASS_NAMES = (
    "Grade1",
    "Grade2",
    "Grade3",
    "Grade4",
)

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: name
    for name, index in CLASS_TO_INDEX.items()
}

SCALES = ("5x", "10x", "20x")

SUPPORTED_DATASETS = (
    "tcga_prad",
    "gleason19",
    "diagset",
)
