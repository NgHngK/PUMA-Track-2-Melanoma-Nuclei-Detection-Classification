"""Stage 1 output geometry and nucleus classes."""

roi_size = 1024
output_stride = 2
ROI_SIZE = roi_size
STAGE1_OUTPUT_STRIDE = output_stride
NUCLEUS_CLASSES = (
    "nuclei_tumor",
    "nuclei_lymphocyte",
    "nuclei_plasma_cell",
    "nuclei_histiocyte",
    "nuclei_melanophage",
    "nuclei_neutrophil",
    "nuclei_stroma",
    "nuclei_endothelium",
    "nuclei_epithelium",
    "nuclei_apoptosis",
)
CLASS_TO_ID = {name: i for i, name in enumerate(NUCLEUS_CLASSES)}
