from trunk_width_estimation import PackagePaths
from trunk_width_estimation import TrunkAnalyzerAnalyzer
import logging

logging.basicConfig(level=logging.INFO)

trunk_analyzer = TrunkAnalyzerAnalyzer(PackagePaths('width_estimation_config_apple.yaml'), create_vis_flag=False)

hardware_notes = "GPU: Nvidia RTX 3070 Ti, CPU: Intel i7-11700K, RAM: 32GB 3200MHz"
trunk_analyzer.do_performance_analysis(gt_datasets=True, save_segmented_images=False, save_model=False, hardware_notes=hardware_notes, only_nth_images=1)

