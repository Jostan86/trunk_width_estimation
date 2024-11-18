from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QCheckBox,
QScrollArea, QGridLayout, QListWidget, QListWidgetItem, QPushButton, QLineEdit, QSpinBox, QComboBox, QSlider, QTreeWidget, QTreeWidgetItem,
QFormLayout, QGroupBox, QDialog, QFileDialog, QTextEdit, QProgressBar, QMessageBox)
from superqt import QLabeledDoubleRangeSlider, QLabeledRangeSlider
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
import cv2
import numpy as np
from trunk_width_estimation import PackagePaths, TrunkAnalyzerAnalyzer
from trunk_width_estimation.width_estimation import ProcessVisualization, TrunkAnalyzerData, TrunkAnalyzerAbstractOperation
from trunk_width_estimation.parameters import ParametersWidthEstimation
import os
import pandas as pd
import json
from typing import List, Tuple, Dict, Union
import re
from PyQt6 import QtGui
from PyQt6.QtWidgets import QDialog, QVBoxLayout
import logging
import matplotlib.pyplot as plt

def determine_visualization_widgets_needed(visualization_widgets_list, num_vis_needed, prev_num_vis):
    """A somewhat overcomplicated function to handle the showing and hiding of the visualization widgets based on the number currently needed"""

    if num_vis_needed == prev_num_vis:
        return visualization_widgets_list, 0
    
    total_num_vis = len(visualization_widgets_list)
    
    vis_to_unhide = total_num_vis - prev_num_vis if num_vis_needed > total_num_vis else max(0, num_vis_needed - prev_num_vis)
    vis_to_add = max(0, num_vis_needed - total_num_vis) if num_vis_needed > total_num_vis else 0
    vis_to_hide = max(0, prev_num_vis - num_vis_needed) if num_vis_needed < prev_num_vis else 0
    
    # Unhide, add, and hide the visualizations as needed
    for i in range(vis_to_unhide):
        visualization_widgets_list[prev_num_vis + i].show()

    for i in range(vis_to_hide):
        visualization_widgets_list[prev_num_vis - i - 1].hide()
    
    return visualization_widgets_list, vis_to_add

def set_widget_font(widget: QWidget, font_scale=1.2, bold=False):
    """Sets the font of a widget to be larger and optionally bold"""
    font = widget.font()
    current_font_size = font.pointSize()
    font.setPointSize(int(current_font_size * font_scale))
    if bold:
        font.setBold(True)
    widget.setFont(font)

class VisualizationSettings:
    def __init__(self):
        self.include_explanations = True
        self.vis_image_scale = 0.7
        self.refine_image_scale = 0.75
        self.image_width = 640
        self.image_height = 480
    
class MainWindow(QMainWindow):
    set_vis_settings_signal = pyqtSignal(VisualizationSettings)

    def __init__(self, analysis_name: str) -> None:
        super().__init__()

        self.image_paths: np.ndarray = None

        self.prev_num_vis = 0

        self.current_image_index = 0

        self.vis_settings = VisualizationSettings()

        self.vis_to_exclude_right_side = []

        self.package_paths = PackagePaths()
        self.package_paths.set_current_analysis_data(analysis_name, set_config_to_current=True)
        
        # Initialize the UI elements with no dependency on the trunk analyzer or the data
        self.initialize_ui_main()             

        self.trunk_analyzer = TrunkAnalyzerAnalyzer(self.package_paths, create_vis_flag=True)

        # Initialize the UI elements that depend on the trunk analyzer
        self.image_widgets: List[VisualizationImageWidget] = []
            
        self.initialize_operation_refinement_widgets()
        self.set_refinement_widget()

        self.scale_sliders_widget.on_scale_apply()
        
        self.update_operations_combo_box()

        # Trigger the first image resegmentation
        self.on_trigger_image_resegmentation()

        self.set_window_size()


    def initialize_ui_main(self):
        """Initializes the UI elements with no dependency on the trunk analyzer or the data"""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowMaximizeButtonHint)

        self.resize(1, 1200)

        main_layout = QHBoxLayout(central_widget)

        self.left_layout = QVBoxLayout()

        self.browse_buttons = BrowseImageWidget(self)
        self.browse_buttons.next_image_signal.connect(self.go_to_next_image)
        self.browse_buttons.previous_image_signal.connect(self.go_to_previous_image)
        self.browse_buttons.set_image_number_signal.connect(self.go_to_image)
        self.browse_buttons.more_info_signal.connect(self.show_image_info)
        self.left_layout.addWidget(self.browse_buttons)
        self.left_layout.addSpacing(10)

        width_data_layout = QHBoxLayout()
        self.width_estimation_label = QLabel("Width Estimation: ")
        label_width = self.width_estimation_label.fontMetrics().boundingRect(self.width_estimation_label.text()).width()
        self.width_estimation_label.setFixedWidth(label_width + 50)
        width_data_layout.addWidget(self.width_estimation_label)
        self.ground_truth_label = QLabel("Ground Truth Width: ")
        width_data_layout.addWidget(self.ground_truth_label)
        width_data_layout.addStretch()
        self.left_layout.addLayout(width_data_layout)

        self.left_layout.addSpacing(10)

        self.image_data_label = QLabel("Image Name:")

        self.left_tabs = QTabWidget(parent=self)
        self.left_layout.addWidget(self.left_tabs)
        
        self.initialize_ui_operation_refinement_tab()
        self.initialize_ui_operation_selection_tab()
        self.initialize_ui_analysis_tab()
        self.initialize_ui_settings_tab()
        self.initialize_ui_right_images()

        main_layout.addLayout(self.left_layout)
        main_layout.addWidget(self.right_scroll_area)        

    def initialize_ui_operation_selection_tab(self):
        """Initializes the UI elements for the browse tab"""

        self.operation_selection_tab = QWidget(parent=self.left_tabs)
        self.operation_selection_tab_layout = QVBoxLayout()
        self.operation_selection_tab.setLayout(self.operation_selection_tab_layout)
        self.left_tabs.addTab(self.operation_selection_tab, "Operation Selection")

        self.operation_selection_label = QLabel("Select Which Operations and Visualizations to Include:")
        set_widget_font(self.operation_selection_label, font_scale=1.2, bold=True)
        self.operation_selection_tab_layout.addSpacing(10)
        self.operation_selection_tab_layout.addWidget(self.operation_selection_label)

        self.operation_selection_explanation_label = QLabel("The top level selection is for whether to include/exclude the operation itself in the " \
                                                            "width estimation algorithm (note that many are required). The sub-selection is whether to include visualizations " \
                                                            "on the right side.")
        self.operation_selection_explanation_label.setWordWrap(True)
        self.operation_selection_tab_layout.addWidget(self.operation_selection_explanation_label)

        self.operation_vis_checkbox_widget = CheckableTreeWidget(self)
        self.operation_vis_checkbox_widget.operation_vis_change_signal.connect(self.on_operation_change)
        operation_selection_color_palette = self.operation_vis_checkbox_widget.palette()
        operation_selection_color_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(255, 255, 255))
        self.operation_vis_checkbox_widget.setPalette(operation_selection_color_palette)
        self.operation_selection_tab_layout.addWidget(self.operation_vis_checkbox_widget)        

        self.operation_selection_tab_layout.addStretch()

    def initialize_ui_operation_refinement_tab(self):
        """Initializes the UI elements for the operation refinement tab"""

        self.operation_refinement_tab = QWidget(parent=self.left_tabs)
        self.operation_refinement_layout = QVBoxLayout()
        self.operation_refinement_tab.setLayout(self.operation_refinement_layout)
        self.left_tabs.addTab(self.operation_refinement_tab, "Adjust Parameters")

        self.operation_refinement_selection_layout = QHBoxLayout()
        self.operation_refinement_selection_label = QLabel("Choose Operation to Adjust:")
        self.operation_selection_combo_box = QComboBox()
        self.operation_selection_combo_box.currentIndexChanged.connect(self.set_refinement_widget)
        # self.operation_refinement_layout.addWidget(self.operation_selection_combo_box)
        self.operation_refinement_selection_layout.addWidget(self.operation_refinement_selection_label)
        self.operation_refinement_selection_layout.addWidget(self.operation_selection_combo_box)
        self.operation_refinement_selection_layout.addStretch()

        # Set the font for the combo box and label
        set_widget_font(self.operation_selection_combo_box, font_scale=1.2)
        set_widget_font(self.operation_refinement_selection_label, font_scale=1.2, bold=True)


        self.operation_refinement_layout.addSpacing(10)
        self.operation_refinement_layout.addLayout(self.operation_refinement_selection_layout)
        self.operation_refinement_layout.addSpacing(10)
    
    def initialize_ui_analysis_tab(self):
        """Initializes the UI elements for the analysis tab"""

        self.analysis_tab = QWidget(parent=self.left_tabs)
        self.analysis_layout = QVBoxLayout()
        self.analysis_tab.setLayout(self.analysis_layout)
        self.left_tabs.addTab(self.analysis_tab, "Analyze and Filter")

        self.analysis_widget = AnalysisWidget(self, self.package_paths, self.set_image_paths)
        self.analysis_layout.addWidget(self.analysis_widget)

        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.clicked.connect(self.run_analysis)
        self.analysis_layout.addWidget(self.run_analysis_button)
    
    def initialize_ui_settings_tab(self):
        """Initializes the UI elements for the settings tab"""

        self.settings_tab = QWidget(parent=self.left_tabs)
        self.settings_layout = QVBoxLayout()
        self.settings_tab.setLayout(self.settings_layout)
        self.left_tabs.addTab(self.settings_tab, "App Settings")

        self.scale_sliders_widget = ScaleSlidersWidget(self)
        self.scale_sliders_widget.apply_scale_signal.connect(self.set_vis_settings)
        self.include_explanations_checkbox = QCheckBox("Include Explanations")
        self.include_explanations_checkbox.setChecked(True)
        self.include_explanations_checkbox.stateChanged.connect(self.set_vis_settings)

        self.load_new_dataset_button = QPushButton("Load New Analysis Results Dataset")
        self.load_new_dataset_button.clicked.connect(self.on_load_new_dataset_button)


        self.settings_layout.addWidget(self.scale_sliders_widget)
        self.settings_layout.addWidget(self.include_explanations_checkbox)
        self.settings_layout.addWidget(self.load_new_dataset_button)
        self.settings_layout.addStretch()

    def initialize_ui_right_images(self):
        """Initializes the UI elements for the right side images, which step through the width estimation operation"""

        self.right_scroll_area = QScrollArea()
        self.right_side_widget = QWidget()
        self.right_side_layout = QVBoxLayout(self.right_side_widget)
        self.right_scroll_area.setWidget(self.right_side_widget)
        self.right_scroll_area.setWidgetResizable(True)

        self.right_side_label = QLabel("Width Estimation Steps")
        self.right_side_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_layout = QGridLayout()
        self.right_side_layout.addLayout(self.image_layout)

    def initialize_operation_refinement_widgets(self):
        """Initializes the operation refinement widgets"""

        self.refinement_widgets: List[OperationRefinementBaseWidget] = []

        self.width_estimation_refinement_widget = WidthEstimationRefinementWidget(self, self.trunk_analyzer.width_estimation)
        self.refinement_widgets.append(self.width_estimation_refinement_widget)

        self.ob_type_filter_refinement_widget = ObjectTypeFilterRefinementWidget(self, self.trunk_analyzer.object_type_filter)
        self.refinement_widgets.append(self.ob_type_filter_refinement_widget)

        self.width_correction_refinement_widget = WidthCorrectionRefinementWidget(self, self.trunk_analyzer.width_correction)
        self.refinement_widgets.append(self.width_correction_refinement_widget)

        self.segmenter_refinement_widget = SegmenterRefinementWidget(self, self.trunk_analyzer.trunk_segmenter)
        self.refinement_widgets.append(self.segmenter_refinement_widget)

        self.depth_filter_refinement_widget = DepthFilterRefinementWidget(self, self.trunk_analyzer.filter_depth)
        self.refinement_widgets.append(self.depth_filter_refinement_widget)

        self.position_filter_refinement_widget = PositionFilterRefinementWidget(self, self.trunk_analyzer.filter_position)
        self.refinement_widgets.append(self.position_filter_refinement_widget)

        self.nms_filter_refinement_widget = NMSFilterRefinementWidget(self, self.trunk_analyzer.filter_nms)
        self.refinement_widgets.append(self.nms_filter_refinement_widget)

        self.depth_calc_filter_refinement_widget = DepthCalcFilterRefinementWidget(self, self.trunk_analyzer.depth_calculation)
        self.refinement_widgets.append(self.depth_calc_filter_refinement_widget)

        self.edge_filter_refinement_widget = EdgeFilterRefinementWidget(self, self.trunk_analyzer.filter_edge)
        self.refinement_widgets.append(self.edge_filter_refinement_widget)

        for refinement_widget in self.refinement_widgets:
            self.set_vis_settings_signal.connect(refinement_widget.set_vis_settings)
            refinement_widget.image_reseg_signal.connect(self.on_trigger_image_resegmentation)
            refinement_widget.set_param_funcs(self.trunk_analyzer.set_parameters, self.trunk_analyzer.get_parameters)

            self.operation_refinement_layout.addWidget(refinement_widget)
            refinement_widget.hide()

        # A blank widget to show when no operation is selected
        self.blank_refinement_widget = QWidget()
        blank_refinement_layout = QVBoxLayout()
        blank_refinement_layout.addStretch()
        self.blank_refinement_widget.setLayout(blank_refinement_layout)
        self.operation_refinement_layout.addWidget(self.blank_refinement_widget)
        self.blank_refinement_widget.hide()   

    def set_image_paths(self, image_paths: np.ndarray, image_index: int = 1):
        self.image_paths = image_paths
        self.browse_buttons.on_image_list_change(len(image_paths))
        if image_index == -1:
            return
        else:
            self.go_to_image(image_index)
    
    def add_right_side_image_widget(self):
        """Adds a new image widget to the right side"""

        image_widget = VisualizationImageWidget(self)

        image_widget.set_spaceholder_image()
        self.set_vis_settings_signal.connect(image_widget.set_vis_settings)
        self.image_layout.addWidget(image_widget, len(self.image_widgets) // 2, len(self.image_widgets) % 2)
        self.image_widgets.append(image_widget)
    
    def check_num_images_right_side(self, num_vis_needed):
        """Sets the number of images to show on the right side"""
        
        self.image_widgets, vis_to_add = determine_visualization_widgets_needed(self.image_widgets, num_vis_needed, self.prev_num_vis)
        
        for i in range(vis_to_add):
            self.add_right_side_image_widget()

        # If any were added, set the visualization settings to the new widgets
        if vis_to_add > 0:
            self.set_vis_settings_signal.emit(self.vis_settings)
        
        self.prev_num_vis = num_vis_needed
            
    
    @pyqtSlot()
    def on_trigger_image_resegmentation(self, update_operation_vis_tree=False):
        """Triggers the resegmentation of the current image based on the selected operations. Currently called when the operation selection 
        changes or a new image is selected.""" 

        if self.image_paths is None:
            return
        
        image_path = self.image_paths[self.current_image_index]
        depth_path = image_path.replace("rgb", "depth")

        image_name = os.path.basename(image_path)

        image_rgb = cv2.imread(image_path)
        image_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        if self.vis_settings.image_height != image_rgb.shape[0] or self.vis_settings.image_width != image_rgb.shape[1]:
            self.vis_settings.image_height = image_rgb.shape[0]
            self.vis_settings.image_width = image_rgb.shape[1]
            self.set_vis_settings()

        trunk_data = TrunkAnalyzerData.from_images(image_rgb, image_depth)
        trunk_data = self.trunk_analyzer.get_width_estimation(trunk_data)
        
        if not self.operation_vis_checkbox_widget.tree_created:
            self.operation_vis_checkbox_widget.create_tree(self.trunk_analyzer)
        elif update_operation_vis_tree:
            self.operation_vis_checkbox_widget.update_tree(self.trunk_analyzer)

        self.browse_buttons.set_value(self.current_image_index + 1)
        self.update_width_labels(trunk_data)
        self.update_visualizations()
    
    def update_visualizations(self):
        # Get the visualizations from the trunk analyzer and set them to the image widgets

        visualizations = self.trunk_analyzer.get_visualizations()

        visualizations_use = []
        for visualization in visualizations:
            if visualization is None:
                continue
            if visualization.name in self.vis_to_exclude_right_side:
                continue
            visualizations_use.append(visualization)
        
        self.check_num_images_right_side(len(visualizations_use))

        for i, visualization in enumerate(visualizations_use):
            self.image_widgets[i].set_image_from_visualization(visualization)
        
        for operation_widget in self.refinement_widgets:
            operation_widget.trigger_update()

    @pyqtSlot(list)
    def on_operation_change(self, to_change_list):
        """Called when the operation selection changes. Updates the right side content and triggers the image resegmentation"""
        parameters = self.trunk_analyzer.get_parameters()

        for to_change in to_change_list:
            if to_change['is_operation']:
                parameters.set_operation_usage(to_change['operation_name'], to_change['include_operation'])
                self.trunk_analyzer.set_parameters(parameters)
                self.update_operations_combo_box()
                self.on_trigger_image_resegmentation(update_operation_vis_tree=True)
            else:
                self.set_visualization_usage(to_change['vis_name'], to_change['include_vis'])
                self.on_trigger_image_resegmentation(update_operation_vis_tree=False)
    
    # def set_operation_usage(self, operation_name: str, use_operation: bool):
    #     """Activates/deactivates an operation in the list"""
    #     # for operation in self.trunk_analyzer.operations:
    #         # if operation.display_name == operation_name:
    #             # operation.employ_operation = use_operation
    #             # break
    #     if operation_name == "Width Correction":
            
    #         parameters.include_width_correction = use_operation
    #         self.trunk_analyzer.set_parameters(parameters)
        
    
    def set_visualization_usage(self, vis_name: str, include_vis: bool):
        """Activates/deactivates a visualization in the list"""
        if vis_name in self.vis_to_exclude_right_side:
            if include_vis:
                self.vis_to_exclude_right_side.remove(vis_name)
        else:
            if not include_vis:
                self.vis_to_exclude_right_side.append(vis_name)

    @pyqtSlot()
    def go_to_next_image(self):
        """Called when the next image button is clicked. Increments the current image index and triggers the image resegmentation"""

        self.current_image_index += 1
        self.go_to_image()
    
    @pyqtSlot()
    def go_to_previous_image(self):
        """Called when the previous image button is clicked. Decrements the current image index and triggers the image resegmentation"""

        self.current_image_index -= 1
        self.go_to_image()
    
    @pyqtSlot(int)
    def go_to_image(self, image_number=None):
        """Loads the current image index image and triggers the image resegmentation"""

        if image_number is not None:
            self.current_image_index = image_number - 1
        if self.current_image_index < 0:
            self.current_image_index = 0
        if self.current_image_index >= len(self.image_paths):
            self.current_image_index = len(self.image_paths) - 1
        self.on_trigger_image_resegmentation()
    
    def update_operations_combo_box(self):
        """Updates the operation selection combo box based on the selected operation"""

        self.operation_selection_combo_box.clear()
        for operation in self.trunk_analyzer.operations:
            if operation.employ_operation:
                self.operation_selection_combo_box.addItem(operation.display_name)

    def set_refinement_widget(self):
        """Sets the refinement widget to show in the browse tab based on the selected operation"""

        selected_operation_name = self.operation_selection_combo_box.currentText()

        set_an_operation = False
        for refinement_widget in self.refinement_widgets:
            if refinement_widget.name == selected_operation_name:
                refinement_widget.show()
                refinement_widget.trigger_update()                
                set_an_operation = True
            else:
                refinement_widget.hide()
        
        if not set_an_operation:
            self.blank_refinement_widget.show()
        else:
            self.blank_refinement_widget.hide()
            
    def set_window_size(self):
        """Sets the image scale for the visualization and refinement images and updates the app width"""

        scroll_area_width = int(self.vis_settings.image_width * self.vis_settings.vis_image_scale * 2 + 92)
        self.right_scroll_area.setFixedWidth(scroll_area_width)
        left_tabs_width = int(self.vis_settings.image_width * self.vis_settings.refine_image_scale + 118)
        self.setFixedWidth(scroll_area_width + left_tabs_width)
    #    
    @pyqtSlot()
    def set_vis_settings(self):
        """Sets the visualization settings based on the settings tab"""

        include_explanations = self.include_explanations_checkbox.isChecked()
        if include_explanations != self.vis_settings.include_explanations:
            self.vis_settings.include_explanations = include_explanations
            # self.update_operation_widgets()
            # self.on_trigger_image_resegmentation()
            self.update_visualizations()

        vis_image_scale = self.scale_sliders_widget.vis_images_scale_slider.value() / 100
        refine_image_scale = self.scale_sliders_widget.refine_images_scale_slider.value() / 100

        if vis_image_scale != self.vis_settings.vis_image_scale or refine_image_scale != self.vis_settings.refine_image_scale:
            self.vis_settings.vis_image_scale = vis_image_scale
            self.vis_settings.refine_image_scale = refine_image_scale
            self.set_window_size()

        self.set_vis_settings_signal.emit(self.vis_settings)
    
    @pyqtSlot()
    def show_image_info(self):
        """Shows the image info dialog"""
        image_path = self.image_paths[self.current_image_index]
        image_info_dialog = ImageInfoDialog(image_path, self.get_gt_data(image_path))

        image_info_dialog.exec()
    
    def get_gt_data(self, image_path):
        gt_data_path = image_path.replace("rgb", "data").replace(".png", ".json")
        if not os.path.exists(gt_data_path):
            return None
        return json.load(open(gt_data_path, "r"))

    def update_width_labels(self, trunk_data: TrunkAnalyzerData):
        """Updates the width labels based on the trunk data"""
        image_path = self.image_paths[self.current_image_index]
        gt_data = self.get_gt_data(image_path)
        if gt_data is None:        
            self.ground_truth_label.setText("Ground Truth Width: N/A")
        else:
            self.ground_truth_label.setText(f"Ground Truth Width: {gt_data['ground_truth_width']*100:.2f} cm")

        width_str = "Width Estimation: "
        if trunk_data.num_instances is not None:
            for width in trunk_data.object_widths:
                width_str += f"{width*100:.2f} cm, "
            width_str = width_str[:-2]
        else:
            width_str += "N/A"
        self.width_estimation_label.setText(width_str)

        gt_text_width = self.ground_truth_label.fontMetrics().boundingRect(self.ground_truth_label.text()).width()
        self.ground_truth_label.setFixedWidth(gt_text_width)

        width_text_width = self.width_estimation_label.fontMetrics().boundingRect(self.width_estimation_label.text()).width()
        self.width_estimation_label.setFixedWidth(width_text_width + 50)
    
    def on_load_new_dataset_button(self):
        self.load_new_dataset()
    
    def load_new_dataset(self, analysis_name: str = None):

        if analysis_name is None:
            analysis_path = QFileDialog.getExistingDirectory(self, "Choose a Results Directory", self.package_paths.analysis_results_dir)
            if analysis_path == "":
                return
            if os.path.dirname(analysis_path) != self.package_paths.analysis_results_dir:
                # message box
                message_box = QMessageBox()
                message_box.setIcon(QMessageBox.Icon.Warning)
                message_box.setText("The selected directory is not in the analysis results directory.")
                message_box.setInformativeText("Please select a directory within the analysis results directory.")
                message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                message_box.exec()
                return
            analysis_name = os.path.basename(analysis_path)

        self.package_paths.set_current_analysis_data(analysis_name, set_config_to_current=True)

        # The same package_paths object is referenced in the trunk analyzer, so the trunk analyzer will use the new config file to update the parameters
        self.trunk_analyzer.reload_parameters()

        for refinement_widget in self.refinement_widgets:
            refinement_widget.on_load_new_dataset()

        analysis_widget_index = self.analysis_layout.indexOf(self.analysis_widget)
        self.analysis_layout.removeWidget(self.analysis_widget)
        self.analysis_widget.deleteLater()
        self.analysis_widget = AnalysisWidget(self, self.package_paths, self.set_image_paths)
        self.analysis_layout.insertWidget(analysis_widget_index, self.analysis_widget)

        self.on_trigger_image_resegmentation()
    
    def run_analysis(self):
        prev_analysis_name = os.path.basename(self.package_paths.current_analysis_results_dir)
        analysis_dialog = RunAnalysisDialog(self, self.package_paths, self.trunk_analyzer)
        load_data = analysis_dialog.exec()
        if load_data:
            analysis_name = os.path.basename(self.package_paths.current_analysis_results_dir)
            self.load_new_dataset(analysis_name)
        else:
            self.package_paths.set_current_analysis_data(prev_analysis_name, set_config_to_current=True)
            self.on_trigger_image_resegmentation()
        
    
class ImageInfoDialog(QDialog):
    def __init__(self, image_path: str, gt_data: dict) -> None:
        super().__init__()

        self.setWindowTitle("Image Info")

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        main_layout.addWidget(QLabel(f"Image Name: {image_path.split('/')[-1]}"))
        main_layout.addWidget(QLabel(f"Image Path: {image_path}"))
        main_layout.addWidget(QLabel(""))
        main_layout.addWidget(QLabel(f"Ground Truth Data:"))
        if gt_data is None:
            main_layout.addWidget(QLabel("No ground truth data found"))
            return
        for key, value in gt_data.items():
            if key == "measured_width":
                continue
            main_layout.addWidget(QLabel(f"\t{key}: {value}"))
       

class VisualizationImageWidget(QWidget):
    trigger_image_resegmentation = pyqtSignal()

    def __init__(self, parent: QWidget, for_operation_refinement=False):
        super().__init__(parent)

        self.vis_settings = VisualizationSettings()

        self.for_operation_refinement = for_operation_refinement 
        self.image_scale = self.vis_settings.refine_image_scale if self.for_operation_refinement else self.vis_settings.vis_image_scale 

        self.initialize_ui()

        self.image = None

        self.metric_labels: List[QLabel] = []
    
    def initialize_ui(self):
        """Initializes the UI elements"""

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.image_name_label = QLabel()
        self.image_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_widget_font(self.image_name_label, font_scale=1.25, bold=True)
        self.main_layout.addWidget(self.image_name_label)

        self.image_label = QLabel()
        self.main_layout.addWidget(self.image_label)

        self.explanation_label = QLabel()
        self.explanation_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.explanation_label.setWordWrap(True)
        self.main_layout.addWidget(self.explanation_label)
        
        self.set_spaceholder_image()

    def set_spaceholder_image(self):
        """Sets a spaceholder image that is whitish"""

        image = np.ones((self.vis_settings.image_height, self.vis_settings.image_width, 3), dtype=np.uint8) * 240
        self.set_image_from_numpy(image)

        self.image_name_label.setText("")
        self.explanation_label.setText("")
    
    def set_image_from_visualization(self, visualization: ProcessVisualization=None):
        """Sets the image from a visualization object. Also loads the metrics if the widget is for operation refinement"""

        # remove the stretch at the end if there is one
        if self.main_layout.itemAt(self.main_layout.count() - 1).widget() is None:
            self.main_layout.removeItem(self.main_layout.itemAt(self.main_layout.count() - 1))

        if visualization is None:
            self.set_spaceholder_image()
        else:
            self.set_image_from_numpy(visualization.get_image())

        self.image_name_label.setText(visualization.name)

        if self.vis_settings.include_explanations:
            self.explanation_label.show()
            self.explanation_label.setText(visualization.explanation)
        else:
            self.explanation_label.hide()
        
        # Load the metrics if the widget is for operation refinement
        if self.for_operation_refinement:
            # Remove the previous metrics
            for metric_label in self.metric_labels:
                metric_label.setParent(None)
            self.metric_labels = []
        
            metrics = visualization.metrics
            indented = False
            for metric_name, metric_value in metrics:
                if metric_name == "":
                    continue

                if metric_name.startswith("Mask"):
                    indented = True
                
                if indented and not metric_name.startswith("Mask"):
                    metric_name = "    " + metric_name

                metric_label = QLabel(f"{metric_name} {metric_value}")
                self.main_layout.addWidget(metric_label)
                self.metric_labels.append(metric_label)

        self.main_layout.addStretch()

    def set_image_from_numpy(self, image: np.ndarray):
        """Sets the image from a numpy array"""

        # Make a copy of the image to hold onto for resizing
        self.image = image.copy()
        
        # Resize the image if the scale is not 1.0
        if not np.isclose(self.image_scale, 1.0):
            image = cv2.resize(image, (0, 0), fx=self.image_scale, fy=self.image_scale)        

        # Convert the image to RGB and set it to the image label
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_qt = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image_qt)
        self.image_label.setPixmap(pixmap)
    
    @pyqtSlot(VisualizationSettings)
    def set_vis_settings(self, vis_settings: VisualizationSettings):
        """Sets the visualization settings"""

        self.vis_settings = vis_settings
        self.image_scale = self.vis_settings.refine_image_scale if self.for_operation_refinement else self.vis_settings.vis_image_scale

        if self.image is not None:
            self.set_image_from_numpy(self.image)
        
class ProcessStepListWidget(QListWidget):
    trigger_operations_change = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
            
    def create_operations_list(self, trunk_analyzer: TrunkAnalyzerAnalyzer):
        """Creates the list of the operation steps in the trunk analyzer"""

        self.item_dict = {}
        for operation in trunk_analyzer.operations:
            item = QListWidgetItem(operation.display_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.addItem(item)
            self.item_dict[operation.display_name] = item
        
        self.trunk_analyzer = trunk_analyzer
        self.itemChanged.connect(self.on_list_item_changed)

    def on_list_item_changed(self, item: QListWidgetItem):
        """Called when an item in the list is changed. Updates which operation stpes the trunk analyzer uses"""

        for i in range(self.count()):
            operation_name = self.item(i).text()
            include_operation = self.item(i).checkState() == Qt.CheckState.Checked
            self.set_operations(operation_name, include_operation)
    
        self.trigger_operations_change.emit()
    
    def set_operations(self, operation_name: str, employ_operation: bool):
        """Activates/deactivates an operation in the list"""
        for operation in self.trunk_analyzer.operations:
            if operation.display_name == operation_name:
                operation.employ_operation = employ_operation
                break



class CheckableTreeWidget(QWidget):
    operation_vis_change_signal = pyqtSignal(list)

    def __init__(self, parent: QWidget):
        super().__init__(parent)

        self.tree_created = False

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)  # Hide the header if not needed
        self.tree.itemChanged.connect(self.on_tree_item_changed)
        
        # Layout setup
        layout = QVBoxLayout(self)
        layout.addWidget(self.tree)
        self.setLayout(layout)

    def create_tree(self, trunk_analyzer: TrunkAnalyzerAnalyzer):
        """Creates the tree of operation steps and visualizations in the trunk analyzer"""
        self.tree.blockSignals(True)

        for operation in trunk_analyzer.operations:
            # Modify the display name if the operation is required
            display_name = operation.display_name
            if operation.is_required:
                display_name += " (required)"
            
            # Create the operation item
            operation_item = QTreeWidgetItem([display_name])
            # operation_item.setFlags(operation_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            
            # # If the operation is required, make the checkbox non-toggleable
            # if operation.is_required:
            #     operation_item.setFlags(operation_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)  # Remove checkable flag
            
            if operation.is_required:
                # operation_item.setFlags(operation_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                operation_item.setFlags(operation_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)


            self.tree.addTopLevelItem(operation_item)

            if operation.employ_operation:
                operation_item.setCheckState(0, Qt.CheckState.Checked)
            else:
                operation_item.setCheckState(0, Qt.CheckState.Unchecked)
                continue

            # Add visualizations as children
            for visualization in operation.visualizations:
                visualization_name = visualization.name
                if visualization.extra_detail_only:
                    continue
                vis_item = QTreeWidgetItem([visualization_name])
                # vis_item.setFlags(vis_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                vis_item.setCheckState(0, Qt.CheckState.Checked)
                operation_item.addChild(vis_item)

        self.tree.expandAll()

        self.tree_created = True
        self.tree.blockSignals(False)
    
    def update_tree(self, trunk_analyzer: TrunkAnalyzerAnalyzer):
        """Updates the tree of operation steps and visualizations in the trunk analyzer"""
        self.tree.blockSignals(True)

        for operation in trunk_analyzer.operations:
            operation_name = operation.display_name
            operation_item = self.tree.findItems(operation_name, Qt.MatchFlag.MatchExactly)
            if len(operation_item) == 0:
                continue
            operation_item = operation_item[0]

            if operation.employ_operation:
                operation_item.setCheckState(0, Qt.CheckState.Checked)

                child_names = [operation_item.child(i).text(0) for i in range(operation_item.childCount())]

                for visualization in operation.visualizations:
                    if visualization.extra_detail_only:
                        continue
                    
                    visualization_name = visualization.name
                    
                    if visualization_name not in child_names:
                        # Add the visualization if it doesn't exist
                        vis_item = QTreeWidgetItem([visualization_name])
                        vis_item.setCheckState(0, Qt.CheckState.Checked)
                        operation_item.addChild(vis_item)
                        
                
            else:
                operation_item.setCheckState(0, Qt.CheckState.Unchecked)
                for i in range(operation_item.childCount()):
                    operation_item.child(i).setCheckState(0, Qt.CheckState.Unchecked)
        
        self.tree.blockSignals(False)


    def on_tree_item_changed(self, item: QTreeWidgetItem, column: int):
        """Called when an item in the tree is changed. Updates which operations the trunk analyzer uses"""
        self.tree.blockSignals(True)
        
        to_change = []
        
        self.add_item_data(item, to_change)

        if item.parent() is None:
            parent_check_state = item.checkState(column)
            for i in range(item.childCount()):
                item.child(i).setCheckState(column, parent_check_state)
                self.add_item_data(item.child(i), to_change)

        self.tree.blockSignals(False)

        self.operation_vis_change_signal.emit(to_change)

    
    def add_item_data(self, item: QTreeWidgetItem, to_change_list: list, column=0):
        """Adds the item data to the to_change_list"""
        if item.parent() is None:
            to_change_list.append({'operation_name': item.text(column), 'is_operation': True, 'is_visualization': False, 'include_operation': item.checkState(column) == Qt.CheckState.Checked})
        else:
            to_change_list.append({'operation_name': item.parent().text(column), 'is_operation': False, 'is_visualization': True, 'vis_name': item.text(column), 'include_vis': item.checkState(column) == Qt.CheckState.Checked})

        
class BrowseImageWidget(QWidget):
    next_image_signal = pyqtSignal()
    previous_image_signal = pyqtSignal()
    set_image_number_signal = pyqtSignal(int)
    more_info_signal = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.setLayout(layout)

        button_layout = QHBoxLayout()

        self.previous_button = QPushButton("Previous Image")
        self.previous_button.clicked.connect(self.on_previous_image)
        button_layout.addWidget(self.previous_button)
        
        self.next_button = QPushButton("Next Image")
        self.next_button.clicked.connect(self.on_next_image)
        button_layout.addWidget(self.next_button)

        layout.addLayout(button_layout)

        go_to_image_layout = QHBoxLayout()
        self.go_to_image_label = QLabel("Image Number:")
        label_width = self.go_to_image_label.fontMetrics().boundingRect(self.go_to_image_label.text()).width()
        self.go_to_image_label.setFixedWidth(label_width)

        self.image_number_spin_box = QSpinBox()
        self.image_number_spin_box.setRange(1, 10000)
        self.image_number_spin_box.setValue(1)
        self.image_number_spin_box.valueChanged.connect(self.set_image_number_signal)
        self.image_number_spin_box.setFixedWidth(100)

        self.total_images_label = QLabel("of 0")
        self.total_images_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.total_images_label.setFixedWidth(70)

        self.image_info_button = QPushButton("Image Info")
        self.image_info_button.clicked.connect(self.more_info_signal)

        go_to_image_layout.addWidget(self.go_to_image_label)
        go_to_image_layout.addWidget(self.image_number_spin_box)
        go_to_image_layout.addWidget(self.total_images_label)
        go_to_image_layout.addWidget(self.image_info_button)

        layout.addLayout(go_to_image_layout)

    def on_next_image(self):
        self.next_image_signal.emit()
    
    def on_previous_image(self):
        self.previous_image_signal.emit()
    
    # def on_go_to_image(self):
    #     self.set_image_number_signal.emit((self.image_number_spin_box.value()))
    
    def on_image_list_change(self, num_images):
        self.total_images_label.setText(f"of {num_images}")
        self.image_number_spin_box.setRange(1, num_images)

    def set_value(self, value):
        self.image_number_spin_box.blockSignals(True)
        self.image_number_spin_box.setValue(value)
        self.image_number_spin_box.blockSignals(False)


class ScaleSlidersWidget(QWidget):
    # apply_scale_signal = pyqtSignal(float, float)
    apply_scale_signal = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.vis_images_scale_label = QLabel("Right Image Scale:")
        vis_text_width = self.vis_images_scale_label.fontMetrics().boundingRect(self.vis_images_scale_label.text()).width()
        self.vis_images_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.vis_images_scale_slider.setRange(50, 100)
        self.vis_images_scale_slider.setValue(70)
        self.vis_images_scale_slider.setFixedWidth(200)
        self.vis_images_slider_value_label = QLabel("70%")
        self.vis_images_slider_value_label.setFixedWidth(50)
        vis_images_scale_layout = QHBoxLayout()
        vis_images_scale_layout.addWidget(self.vis_images_scale_label)
        vis_images_scale_layout.addWidget(self.vis_images_scale_slider)
        vis_images_scale_layout.addWidget(self.vis_images_slider_value_label)
        vis_images_scale_layout.addStretch()
        layout.addLayout(vis_images_scale_layout)

        self.refine_images_scale_label = QLabel("Left Image Scale:")
        refine_text_width = self.refine_images_scale_label.fontMetrics().boundingRect(self.refine_images_scale_label.text()).width()
        self.refine_images_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.refine_images_scale_slider.setRange(75, 100)
        self.refine_images_scale_slider.setValue(75)
        self.refine_images_scale_slider.setFixedWidth(200)
        self.refine_images_slider_value_label = QLabel("70%")
        self.refine_images_slider_value_label.setFixedWidth(50)
        refine_images_scale_layout = QHBoxLayout()
        refine_images_scale_layout.addWidget(self.refine_images_scale_label)
        refine_images_scale_layout.addWidget(self.refine_images_scale_slider)
        refine_images_scale_layout.addWidget(self.refine_images_slider_value_label)
        refine_images_scale_layout.addStretch()
        layout.addLayout(refine_images_scale_layout)

        max_width = max(vis_text_width, refine_text_width)
        self.vis_images_scale_label.setFixedWidth(max_width+5)
        self.refine_images_scale_label.setFixedWidth(max_width+5)
        
        # self.apply_scale_button.clicked.connect(self.on_scale_apply)
        self.vis_images_scale_slider.valueChanged.connect(self.on_scale_apply)  
        self.refine_images_scale_slider.valueChanged.connect(self.on_scale_apply)

    def on_scale_apply(self):
        """Sets the image scales to the nearest 5% and emits the signal"""

        self.vis_images_scale_slider.setValue((self.vis_images_scale_slider.value() + 2) // 5 * 5)
        self.refine_images_scale_slider.setValue((self.refine_images_scale_slider.value() + 2) // 5 * 5)

        self.vis_images_slider_value_label.setText(f"{self.vis_images_scale_slider.value()}%")
        self.refine_images_slider_value_label.setText(f"{self.refine_images_scale_slider.value()}%")
        # vis_image_scale = self.vis_images_scale_slider.value() / 100
        # refine_image_scale = self.refine_images_scale_slider.value() / 100
        # self.apply_scale_signal.emit(vis_image_scale, refine_image_scale)
        self.apply_scale_signal.emit()


class OperationRefinementBaseWidget(QWidget):
    # image_scale_signal = pyqtSignal(float, float)
    # image_size_signal = pyqtSignal(int, int)
    # include_explanations_signal = pyqtSignal(bool)
    image_reseg_signal = pyqtSignal()
    set_vis_settings_signal = pyqtSignal(VisualizationSettings)

    def __init__(self, parent: QWidget, operation: TrunkAnalyzerAbstractOperation) -> None:
        super().__init__(parent)

        self.vis_settings = VisualizationSettings()

        self.operation = operation
        self.prev_num_vis = 0
        self.auto_apply_change = True

        self.visualization_widgets: List[VisualizationImageWidget] = []

        self.default_settings = {}
        self.starting_settings = {}

        self.initialize_ui()

        self.on_load_new_dataset()        

        # self.save_settings(self.default_settings)
    
    def set_param_funcs(self, set_params_func: callable, get_params_func: callable):
        self.set_params_func = set_params_func
        self.get_params_func = get_params_func

    def initialize_ui(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.operation_name_label = QLabel(self.operation.display_name)
        self.operation_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_widget_font(self.operation_name_label, font_scale=1.2)
        self.main_layout.addWidget(self.operation_name_label)

        self.operation_parameters_layout = QVBoxLayout()
        self.main_layout.addLayout(self.operation_parameters_layout)

        self.initialize_settings_ui()

        self.apply_button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.on_apply)
        self.apply_auto_checkbox = QCheckBox("Auto Apply On Change")
        self.apply_auto_checkbox.setChecked(True)
        self.apply_auto_checkbox.stateChanged.connect(self.on_auto_apply_state_changed)
        self.on_auto_apply_state_changed(True)
        self.apply_button_layout.addWidget(self.apply_button)
        self.apply_button_layout.addWidget(self.apply_auto_checkbox)
        
        self.main_layout.addLayout(self.apply_button_layout)

        self.reset_settings_layout = QHBoxLayout()
        self.reset_settings_button = QPushButton("Reset Settings")
        self.reset_settings_button.clicked.connect(self.on_reset_settings)
        self.reset_settings_button.setToolTip("Resets the settings to what they were when the operation refinement was opened")
        self.reset_settings_layout.addWidget(self.reset_settings_button)
        self.reset_to_default_button = QPushButton("Load Default Settings")
        self.reset_to_default_button.setToolTip("Resets the settings to the default settings")
        self.reset_to_default_button.clicked.connect(self.on_reset_to_default)
        self.reset_settings_layout.addWidget(self.reset_to_default_button)
        self.main_layout.addLayout(self.reset_settings_layout)

        self.visualization_scroll_area = QScrollArea()
        self.visualization_content_widget = QWidget()
        self.visualization_content_layout = QVBoxLayout(self.visualization_content_widget)
        self.visualization_scroll_area.setWidget(self.visualization_content_widget)
        self.visualization_scroll_area.setWidgetResizable(True)

        self.main_layout.addWidget(self.visualization_scroll_area)

    def initialize_settings_ui(self):
        pass

    @property
    def name(self):
        return self.operation.display_name
    
    def on_auto_apply_state_changed(self, state):
        self.auto_apply_change = state
        self.apply_button.setEnabled(not state)

    def trigger_update(self):
        """Triggers the update of the visualization images if the operation is selected"""

        if self.isHidden():
            return
        if self.operation.visualizations is None:
            return

        self.setup_visualization_layout()
        for i, visualization in enumerate(self.operation.visualizations):
            self.visualization_widgets[i].set_image_from_visualization(visualization)

    def setup_visualization_layout(self):
        """Sets up the visualization layout based on the number of visualizations needed, which can change based on the operation
        and number of masks in a particular image"""

        num_vis_needed = self.operation.num_visualizations + self.operation.num_extra_detail_visualizations
        
        self.visualization_widgets, vis_to_add = determine_visualization_widgets_needed(self.visualization_widgets, num_vis_needed, self.prev_num_vis)
        
        for i in range(vis_to_add):
            self.add_visualization_widget()

        # If any were added, set the image size and scale
        if vis_to_add > 0:
            self.set_vis_settings_signal.emit(self.vis_settings)
        
        self.prev_num_vis = num_vis_needed

    def add_visualization_widget(self):
        """Adds a visualization widget to the layout at the end"""
        visualization_widget = VisualizationImageWidget(self, for_operation_refinement=True)
        self.set_vis_settings_signal.connect(visualization_widget.set_vis_settings)

        visualization_widget.set_spaceholder_image()
        self.visualization_content_layout.addWidget(visualization_widget)
        self.visualization_widgets.append(visualization_widget)

    @pyqtSlot(VisualizationSettings)
    def set_vis_settings(self, vis_settings: VisualizationSettings):
        self.vis_settings = vis_settings
        self.set_vis_settings_signal.emit(vis_settings)

    def on_reset_settings(self):
        self.block_all_signals(True)
        self.load_settings(self.starting_settings)
        self.block_all_signals(False)
        self.on_apply()
    
    def on_reset_to_default(self):
        self.block_all_signals(True)
        self.load_settings(self.default_settings)
        self.block_all_signals(False)
        self.on_apply()
    
    def on_load_new_dataset(self):
        """Called when the dataset is changed. Updates the values to those currently in the trunk analyzer"""
        self.block_all_signals(True)
        self.load_operation_settings()
        self.default_settings = {}
        self.starting_settings = {}
        self.save_settings(self.default_settings)
        self.save_settings(self.starting_settings)
        self.block_all_signals(False)

    def on_value_changed(self):
        if self.auto_apply_change:
            self.on_apply()
    
    def show(self):
        self.save_settings(self.starting_settings)
        super().show()      

    def block_all_signals(self, block):
        self.blockSignals(block)
        for child_widget in self.findChildren(QWidget):
            child_widget.blockSignals(block)


    def on_apply(self):
        pass

    def save_settings(self, settings: dict):
        pass

    def load_settings(self, settings: dict):
        pass

    def load_operation_settings(self):
        pass

class DepthCalcFilterRefinementWidget(OperationRefinementBaseWidget):
    def __init__(self, parent: QWidget, operation: TrunkAnalyzerAbstractOperation):
        super().__init__(parent, operation)

    def initialize_settings_ui(self):
        self.settings_layout = QFormLayout()

        self.top_ignore_spin_box = QDoubleSpinBox(parent=self)
        self.top_ignore_spin_box.setRange(0, 1)
        self.top_ignore_spin_box.setToolTip("Proportion of the top of the image to ignore mask points in when calculating the depth.")
        self.top_ignore_spin_box.setSingleStep(0.02)
        self.top_ignore_spin_box.valueChanged.connect(self.top_ignore_spin_box_changed)
        self.settings_layout.addRow("Top Ignore:", self.top_ignore_spin_box)

        self.bottom_ignore_spin_box = QDoubleSpinBox(parent=self)
        self.bottom_ignore_spin_box.setRange(0, 1)
        self.bottom_ignore_spin_box.setToolTip("Proportion of the bottom of the image to ignore mask points in when calculating the depth.")
        self.bottom_ignore_spin_box.setSingleStep(0.02)
        self.bottom_ignore_spin_box.valueChanged.connect(self.bottom_ignore_spin_box_changed)
        self.settings_layout.addRow("Bottom Ignore:", self.bottom_ignore_spin_box)

        self.min_num_points_spin_box = QSpinBox(parent=self)
        self.min_num_points_spin_box.setRange(100, 1000000)
        self.min_num_points_spin_box.setSingleStep(100)
        self.min_num_points_spin_box.setToolTip("Minimum number of valid depth points needed to keep the mask.")
        self.min_num_points_spin_box.valueChanged.connect(self.min_num_points_spin_box_changed)
        self.settings_layout.addRow("Min Num Points:", self.min_num_points_spin_box)

        self.depth_percentile_spin_box = QSpinBox(parent=self)
        self.depth_percentile_spin_box.setRange(1, 99)
        self.depth_percentile_spin_box.setToolTip("Percentile of the depth values to use for the depth estimate.")
        self.depth_percentile_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Depth Percentile:", self.depth_percentile_spin_box)

        self.operation_parameters_layout.addLayout(self.settings_layout)


    def top_ignore_spin_box_changed(self, top_ignore):
        bottom_ignore = self.bottom_ignore_spin_box.value()
        if top_ignore + bottom_ignore >= .99:
            self.bottom_ignore_spin_box.setValue(0.99 - top_ignore)
        
        self.on_value_changed()
    
    def bottom_ignore_spin_box_changed(self, bottom_ignore):
        top_ignore = self.top_ignore_spin_box.value()
        if top_ignore + bottom_ignore >= 0.99:
            self.top_ignore_spin_box.setValue(0.99 - bottom_ignore)
        
        self.on_value_changed()
    
    def min_num_points_spin_box_changed(self, min_num_points):
        if min_num_points > self.vis_settings.image_height * self.vis_settings.image_width:
            self.min_num_points_spin_box.setValue(self.vis_settings.image_height * self.vis_settings.image_width)
        
        self.on_value_changed()
    
    def on_apply(self):
        top_ignore = self.top_ignore_spin_box.value()
        bottom_ignore = self.bottom_ignore_spin_box.value()
        min_num_points = self.min_num_points_spin_box.value()
        depth_percentile = self.depth_percentile_spin_box.value()
    
        params: ParametersWidthEstimation = self.get_params_func()
        params.depth_calc_top_ignore = top_ignore
        params.depth_calc_bottom_ignore = bottom_ignore
        params.depth_calc_min_num_points = min_num_points
        params.depth_calc_percentile = depth_percentile
        self.set_params_func(params)

        self.image_reseg_signal.emit()
    
    def save_settings(self, settings: dict):
        settings["top_ignore"] = self.top_ignore_spin_box.value()
        settings["bottom_ignore"] = self.bottom_ignore_spin_box.value()
        settings["min_num_points"] = self.min_num_points_spin_box.value()
        settings["depth_percentile"] = self.depth_percentile_spin_box.value()
    
    def load_settings(self, settings: dict):
        self.top_ignore_spin_box.setValue(settings["top_ignore"])
        self.bottom_ignore_spin_box.setValue(settings["bottom_ignore"])
        self.min_num_points_spin_box.setValue(settings["min_num_points"])
        self.depth_percentile_spin_box.setValue(int(settings["depth_percentile"]))
    
    def load_operation_settings(self):
        self.top_ignore_spin_box.setValue(self.operation.parameters.depth_calc_top_ignore)
        self.bottom_ignore_spin_box.setValue(self.operation.parameters.depth_calc_bottom_ignore)
        self.min_num_points_spin_box.setValue(self.operation.parameters.depth_calc_min_num_points)
        self.depth_percentile_spin_box.setValue(int(self.operation.parameters.depth_calc_percentile))

class EdgeFilterRefinementWidget(OperationRefinementBaseWidget):
    def __init__(self, parent: QWidget, operation: TrunkAnalyzerAbstractOperation) -> None:
        super().__init__(parent, operation)

    def initialize_settings_ui(self):
        self.settings_layout = QFormLayout()

        self.edge_threshold_spin_box = QDoubleSpinBox()
        self.edge_threshold_spin_box.setRange(0.01, 0.49)
        self.edge_threshold_spin_box.setSingleStep(0.02)
        self.edge_threshold_spin_box.setToolTip("Proportion of the image width that is considered the edge.")
        self.edge_threshold_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Edge Threshold:", self.edge_threshold_spin_box)

        self.edge_size_threshold_spin_box = QDoubleSpinBox()
        self.edge_size_threshold_spin_box.setRange(0.02, 1)
        self.edge_size_threshold_spin_box.setSingleStep(0.02)
        self.edge_size_threshold_spin_box.setToolTip("Proportion of the mask that must be in the edge 'zone' for the mask to be removed.")
        self.edge_size_threshold_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Edge Size Threshold:", self.edge_size_threshold_spin_box)

        self.operation_parameters_layout.addLayout(self.settings_layout)
    
    def on_apply(self):
        edge_threshold = self.edge_threshold_spin_box.value()
        edge_size_threshold = self.edge_size_threshold_spin_box.value()

        params: ParametersWidthEstimation = self.get_params_func()
        params.filter_edge_edge_threshold = edge_threshold
        params.filter_edge_size_threshold = edge_size_threshold
        self.set_params_func(params)

        self.image_reseg_signal.emit()
    
    def save_settings(self, settings: dict):
        settings["edge_threshold"] = self.edge_threshold_spin_box.value()
        settings["edge_size_threshold"] = self.edge_size_threshold_spin_box.value()

    def load_settings(self, settings: dict):
        self.edge_threshold_spin_box.setValue(settings["edge_threshold"])
        self.edge_size_threshold_spin_box.setValue(settings["edge_size_threshold"])
    
    def load_operation_settings(self):
        self.edge_threshold_spin_box.setValue(self.operation.parameters.filter_edge_edge_threshold)
        self.edge_size_threshold_spin_box.setValue(self.operation.parameters.filter_edge_size_threshold)


class PositionFilterRefinementWidget(OperationRefinementBaseWidget):
    def __init__(self, parent: QWidget, operation: TrunkAnalyzerAbstractOperation) -> None:
        super().__init__(parent, operation)
    
    def initialize_settings_ui(self):
        self.settings_layout = QFormLayout()

        self.bottom_position_spin_box = QDoubleSpinBox()
        self.bottom_position_spin_box.setRange(0.01, 0.99)
        self.bottom_position_spin_box.setSingleStep(0.02)
        self.bottom_position_spin_box.setToolTip("If the bottom of the mask is above this threshold, the mask is considered to be too high." +
                                              "The value is the percentage of the image height up from the bottom.")
        self.bottom_position_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Bottom Position Threshold:", self.bottom_position_spin_box)

        self.top_position_spin_box = QDoubleSpinBox()
        self.top_position_spin_box.setRange(0.01, 0.99)
        self.top_position_spin_box.setSingleStep(0.02)
        self.top_position_spin_box.setToolTip("If the top of the mask is below this threshold, the mask is considered to be too low." +
                                              "The value is the percentage of the image height down from the top.")
        self.top_position_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Top Position Threshold:", self.top_position_spin_box)

        self.operation_parameters_layout.addLayout(self.settings_layout)
    
    def on_apply(self):
        bottom_position = self.bottom_position_spin_box.value()
        top_position = self.top_position_spin_box.value()

        params: ParametersWidthEstimation = self.get_params_func()
        params.filter_position_bottom_threshold = bottom_position
        params.filter_position_top_threshold = top_position
        self.set_params_func(params)

        self.image_reseg_signal.emit()
    
    def save_settings(self, settings: dict):
        settings["bottom_position"] = self.bottom_position_spin_box.value()
        settings["top_position"] = self.top_position_spin_box.value()
    
    def load_settings(self, settings: dict):
        self.bottom_position_spin_box.setValue(settings["bottom_position"])
        self.top_position_spin_box.setValue(settings["top_position"])
    
    def load_operation_settings(self):
        self.bottom_position_spin_box.setValue(self.operation.parameters.filter_position_bottom_threshold)
        self.top_position_spin_box.setValue(self.operation.parameters.filter_position_top_threshold)

class DepthFilterRefinementWidget(OperationRefinementBaseWidget):
    def __init__(self, parent: QWidget, operation: TrunkAnalyzerAbstractOperation) -> None:
        super().__init__(parent, operation)
    
    def initialize_settings_ui(self):
        self.settings_layout = QFormLayout()

        self.depth_threshold_spin_box = QDoubleSpinBox()
        self.depth_threshold_spin_box.setRange(0.01, 20.0)
        self.depth_threshold_spin_box.setSingleStep(0.1)
        self.depth_threshold_spin_box.setToolTip("The maximum distance allowed for trees, if farther than this the segmentation is removed.")
        self.depth_threshold_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Depth Threshold (m):", self.depth_threshold_spin_box)

        self.operation_parameters_layout.addLayout(self.settings_layout)
    
    def on_apply(self):
        depth_threshold = self.depth_threshold_spin_box.value()

        params: ParametersWidthEstimation = self.get_params_func()
        params.filter_depth_max_depth = depth_threshold
        self.set_params_func(params)

        self.image_reseg_signal.emit()
    
    def save_settings(self, settings: dict):
        settings["depth_threshold"] = self.depth_threshold_spin_box.value()

    def load_settings(self, settings: dict):
        self.depth_threshold_spin_box.setValue(settings["depth_threshold"])
    
    def load_operation_settings(self):
        self.depth_threshold_spin_box.setValue(self.operation.parameters.filter_depth_max_depth)

class NMSFilterRefinementWidget(OperationRefinementBaseWidget):
    def __init__(self, parent: QWidget, operation: TrunkAnalyzerAbstractOperation) -> None:
        super().__init__(parent, operation)
    
    def initialize_settings_ui(self):
        self.settings_layout = QFormLayout()

        self.nms_threshold_spin_box = QDoubleSpinBox()
        self.nms_threshold_spin_box.setRange(0.01, 0.49)
        self.nms_threshold_spin_box.setSingleStep(0.02)
        self.nms_threshold_spin_box.setToolTip("The threshold for the non-maximum suppression algorithm for differing classes" +
                                               "(Yolo does it for the same class)")
        self.nms_threshold_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("NMS Threshold:", self.nms_threshold_spin_box)

        self.operation_parameters_layout.addLayout(self.settings_layout)
    
    def on_apply(self):
        nms_threshold = self.nms_threshold_spin_box.value()

        params: ParametersWidthEstimation = self.get_params_func()
        params.filter_nms_overlap_threshold = nms_threshold
        self.set_params_func(params)

        self.image_reseg_signal.emit()
    
    def save_settings(self, settings: dict):
        settings["nms_threshold"] = self.nms_threshold_spin_box.value()

    def load_settings(self, settings: dict):
        self.nms_threshold_spin_box.setValue(settings["nms_threshold"])
    
    def load_operation_settings(self):
        self.nms_threshold_spin_box.setValue(self.operation.parameters.filter_nms_overlap_threshold)

class WidthEstimationRefinementWidget(OperationRefinementBaseWidget):
    def __init__(self, parent: QWidget, operation: TrunkAnalyzerAbstractOperation) -> None:
        super().__init__(parent, operation)
    
    def initialize_settings_ui(self):
        self.settings_layout = QFormLayout()

        self.segment_length_spin_box = QSpinBox()
        self.segment_length_spin_box.setRange(5, 100)
        self.segment_length_spin_box.setSingleStep(5)
        self.segment_length_spin_box.setToolTip("The length of the segments to use when estimating the width.")
        self.segment_length_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Segment Length:", self.segment_length_spin_box)

        self.pixel_width_percentile_spin_box = QSpinBox()
        self.pixel_width_percentile_spin_box.setRange(1, 99)
        self.pixel_width_percentile_spin_box.setToolTip("The percentile of the pixel width values to use for the width estimate.")
        self.pixel_width_percentile_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Pixel Width Percentile:", self.pixel_width_percentile_spin_box)

        self.operation_parameters_layout.addLayout(self.settings_layout)

    def on_apply(self):
        segment_length = self.segment_length_spin_box.value()
        pixel_width_percentile = self.pixel_width_percentile_spin_box.value()

        params: ParametersWidthEstimation = self.get_params_func()
        params.pixel_width_segment_length = segment_length
        params.pixel_width_percentile = pixel_width_percentile
        self.set_params_func(params)

        self.image_reseg_signal.emit()
    
    def save_settings(self, settings: dict):
        settings["segment_length"] = self.segment_length_spin_box.value()
        settings["pixel_width_percentile"] = self.pixel_width_percentile_spin_box.value()
    
    def load_settings(self, settings: dict):
        self.segment_length_spin_box.setValue(settings["segment_length"])
        self.pixel_width_percentile_spin_box.setValue(int(settings["pixel_width_percentile"]))
    
    def load_operation_settings(self):
        self.segment_length_spin_box.setValue(self.operation.parameters.pixel_width_segment_length)
        self.pixel_width_percentile_spin_box.setValue(int(self.operation.parameters.pixel_width_percentile))

class ObjectTypeFilterRefinementWidget(OperationRefinementBaseWidget):
    def __init__(self, parent: QWidget, operation: TrunkAnalyzerAbstractOperation) -> None:
        super().__init__(parent, operation)
    
    def initialize_settings_ui(self):
        self.settings_layout = QFormLayout()

        self.trunk_class_combo_box = QSpinBox()
        self.trunk_class_combo_box.setRange(0, 100)
        self.trunk_class_combo_box.setSingleStep(1)
        self.trunk_class_combo_box.setToolTip("The class of the trunks expected from the model.")
        self.trunk_class_combo_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Model Trunk Class:", self.trunk_class_combo_box)

        self.post_class_combo_box = QSpinBox()
        self.post_class_combo_box.setRange(0, 100)
        self.post_class_combo_box.setSingleStep(1)
        self.post_class_combo_box.setToolTip("The class of the posts expected from the model.")
        self.post_class_combo_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Model Post Class:", self.post_class_combo_box)

        self.operation_parameters_layout.addLayout(self.settings_layout)

    def on_apply(self):
        trunk_class = self.trunk_class_combo_box.value()
        post_class = self.post_class_combo_box.value()

        params: ParametersWidthEstimation = self.get_params_func()
        params.seg_model_trunk_class = trunk_class
        params.seg_model_post_class = post_class
        self.set_params_func(params)

        self.image_reseg_signal.emit()

    def save_settings(self, settings: dict):
        settings["trunk_class"] = self.trunk_class_combo_box.value()
        settings["post_class"] = self.post_class_combo_box.value()

    def load_settings(self, settings: dict):
        self.trunk_class_combo_box.setValue(settings["trunk_class"])
        self.post_class_combo_box.setValue(settings["post_class"])
    
    def load_operation_settings(self):
        self.trunk_class_combo_box.setValue(self.operation.parameters.seg_model_trunk_class)
        self.post_class_combo_box.setValue(self.operation.parameters.seg_model_post_class)

class WidthCorrectionRefinementWidget(OperationRefinementBaseWidget):
    def __init__(self, parent: QWidget, operation: TrunkAnalyzerAbstractOperation) -> None:
        super().__init__(parent, operation)
    
    def initialize_settings_ui(self):

        self.settings_layout = QFormLayout()

        self.correction_slope_spin_box = QDoubleSpinBox()
        self.correction_slope_spin_box.setRange(-0.1, 0.1)
        self.correction_slope_spin_box.setDecimals(4)
        self.correction_slope_spin_box.setSingleStep(0.001)
        
        self.correction_slope_spin_box.setToolTip("The slope of the line to correct the width.")
        self.correction_slope_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Width Correction Slope (mm/px):", self.correction_slope_spin_box)

        self.correction_intercept_spin_box = QDoubleSpinBox()
        self.correction_intercept_spin_box.setRange(-30, 30)
        self.correction_intercept_spin_box.setSingleStep(0.1)
        self.correction_intercept_spin_box.setDecimals(2)
        self.correction_intercept_spin_box.setToolTip("The intercept of the line to correct the width.")
        self.correction_intercept_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Width Correction Intercept (mm):", self.correction_intercept_spin_box)

        self.operation_parameters_layout.addLayout(self.settings_layout)

    def on_apply(self):
        correction_slope = self.correction_slope_spin_box.value() / 1000
        correction_intercept = self.correction_intercept_spin_box.value() / 1000

        params: ParametersWidthEstimation = self.get_params_func()
        params.width_correction_slope = correction_slope
        params.width_correction_intercept = correction_intercept
        self.set_params_func(params)

        self.image_reseg_signal.emit()

    def save_settings(self, settings: dict):
        settings["correction_slope"] = self.correction_slope_spin_box.value()
        settings["correction_intercept"] = self.correction_intercept_spin_box.value()

    def load_settings(self, settings: dict):
        self.correction_slope_spin_box.setValue(settings["correction_slope"])
        self.correction_intercept_spin_box.setValue(settings["correction_intercept"])
    
    def load_operation_settings(self):
        self.correction_slope_spin_box.setValue(self.operation.parameters.width_correction_slope * 1000)
        self.correction_intercept_spin_box.setValue(self.operation.parameters.width_correction_intercept * 1000)


class SegmenterRefinementWidget(OperationRefinementBaseWidget):
    def __init__(self, parent: QWidget, operation: TrunkAnalyzerAbstractOperation) -> None:
        super().__init__(parent, operation)

    def initialize_settings_ui(self):
        self.settings_layout = QFormLayout()

        self.model_confidence_threshold_spin_box = QDoubleSpinBox()
        self.model_confidence_threshold_spin_box.setRange(0.01, 1.0)
        self.model_confidence_threshold_spin_box.setSingleStep(0.02)
        self.model_confidence_threshold_spin_box.setToolTip("The confidence threshold for the model. Masks with a confidence below this are removed.")
        self.model_confidence_threshold_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Model Confidence Threshold:", self.model_confidence_threshold_spin_box)

        self.model_nms_threshold_spin_box = QDoubleSpinBox()
        self.model_nms_threshold_spin_box.setRange(0.01, 1.0)
        self.model_nms_threshold_spin_box.setSingleStep(0.02)
        self.model_nms_threshold_spin_box.setToolTip("The non-maximum suppression threshold for the model, this is only for masks of the same class.")
        self.model_nms_threshold_spin_box.valueChanged.connect(self.on_value_changed)
        self.settings_layout.addRow("Model NMS Threshold:", self.model_nms_threshold_spin_box)

        self.operation_parameters_layout.addLayout(self.settings_layout)
    
    def on_apply(self):
        model_confidence_threshold = self.model_confidence_threshold_spin_box.value()
        model_nms_threshold = self.model_nms_threshold_spin_box.value()

        params: ParametersWidthEstimation = self.get_params_func()
        params.seg_model_confidence_threshold = model_confidence_threshold
        params.seg_model_nms_threshold = model_nms_threshold
        self.set_params_func(params)

        self.image_reseg_signal.emit()
    
    def save_settings(self, settings: dict):
        settings["model_confidence_threshold"] = self.model_confidence_threshold_spin_box.value()
        settings["model_nms_threshold"] = self.model_nms_threshold_spin_box.value()
    
    def load_settings(self, settings: dict):
        self.model_confidence_threshold_spin_box.setValue(settings["model_confidence_threshold"])
        self.model_nms_threshold_spin_box.setValue(settings["model_nms_threshold"])
    
    def load_operation_settings(self):
        self.model_confidence_threshold_spin_box.setValue(self.operation.parameters.seg_model_confidence_threshold)
        self.model_nms_threshold_spin_box.setValue(self.operation.parameters.seg_model_nms_threshold)


class AnalysisWidget(QWidget):
    def __init__(self, parent: QWidget, package_paths: PackagePaths, set_image_paths_func: callable):
        super().__init__(parent)

        self.set_image_paths_func = set_image_paths_func
        self.package_paths = package_paths
        
        self.analysis_results_data: pd.DataFrame = None

        self.keep_filtered_all: np.ndarray = None
        self.keep_filtered: np.ndarray = None

        self.initialize_ui()

        self.load_analysis_data()

    def initialize_filters(self):
        self.filter_selection_widget = QWidget()
        self.filter_selection_layout = QVBoxLayout(self.filter_selection_widget)
        self.filter_selection_scroll_area.setWidget(self.filter_selection_widget)
        self.filter_selection_scroll_area.setWidgetResizable(True)

        all_widths_data = self.analysis_results_data["estimated_widths"].apply(lambda x: AnalysisWidget.analysis_data_2_list(x, dtype=float, scale=100))
        self.analysis_filter_widths = AnalysisFilterParamRange(self, "Estimated Width (cm):", all_widths_data, unit="cm")
        
        self.missing_data_idxs = np.array([1 if len(img_data) == 0 else 0 for img_data in all_widths_data], dtype=bool)
        self.missing_data_checkbox = AnalysisFilterParamCheckbox("Lacks Width Estimate", self.missing_data_idxs)

        x_positions_in_image = self.analysis_results_data["x_positions_in_image"].apply(lambda x: AnalysisWidget.analysis_data_2_list(x, dtype=int, scale=1))
        self.analysis_filter_img_x_position = AnalysisFilterParamRange(self, "X Position in Image (px):", x_positions_in_image, dtype=int, unit="pixels")

        largest_segment_finder_applied = self.analysis_results_data["LargestSegmentFinder_applied"]
        self.largest_segment_finder_applied_checkbox = AnalysisFilterParamCheckbox("Largest Segment Finder Applied", largest_segment_finder_applied)

        object_classes = self.analysis_results_data["classes"].apply(lambda x: AnalysisWidget.analysis_data_2_list(x, dtype=int, scale=1))
        has_tree = np.array([1 if 0 in classes else 0 for classes in object_classes], dtype=bool) # TODO Jostan: Change to get class from config
        has_post = np.array([1 if 1 in classes else 0 for classes in object_classes], dtype=bool)
        has_sprinkler = self.analysis_results_data["FilterObjectType_applied"]
        self.object_type_filter_posts = AnalysisFilterParamCheckbox("Contains Post", has_post)
        self.object_type_filter_trees = AnalysisFilterParamCheckbox("Contains Tree", has_tree)
        self.contains_sprinkler_checkbox = AnalysisFilterParamCheckbox("Contains Sprinkler", has_sprinkler)

        object_depth = self.analysis_results_data["object_y_positions"].apply(lambda x: AnalysisWidget.analysis_data_2_list(x, dtype=int, scale=1000))
        self.object_depth_filter = AnalysisFilterParamRange(self, "Object Depth (mm):", object_depth, dtype=int, unit="mm")
        self.object_depth_filter.setToolTip("The depth of the object in the image.")

        self.width_estimate_error = np.abs(self.analysis_results_data["matched_width"] - self.analysis_results_data["ground_truth_width"]) * 1000
        self.mean_width_error_label.setText(f"Mean Width Error: {self.width_estimate_error.mean():.2f} mm")
        self.width_estimate_error_filter = AnalysisFilterParamRange(self, "Width Estimate Error (mm):", self.width_estimate_error.copy(), single_value_per_image=True, unit="mm")

        if "FilterNMS_applied" in self.analysis_results_data.columns:
            filter_nms_applied = self.analysis_results_data["FilterNMS_applied"]
            self.filter_nms_applied_checkbox = AnalysisFilterParamCheckbox("NMS Filter Applied", filter_nms_applied)
        else:
            self.filter_nms_applied_checkbox = None

        depth_calc_applied = self.analysis_results_data["DepthCalculation_applied"]
        self.lacked_depth_data_checkbox = AnalysisFilterParamCheckbox("A Mask Lacked Enough Depth Data", depth_calc_applied)

        depth_calc_num_points = self.analysis_results_data["DepthCalculation_num_points"].apply(lambda x: AnalysisWidget.analysis_data_2_list(x, dtype=int, scale=1))
        self.depth_calc_num_points_filter = AnalysisFilterParamRange(self, "Number of Depth Points:", depth_calc_num_points, dtype=int, unit="points")

        if "FilterDepth_applied" in self.analysis_results_data.columns:
            depth_filter_applied = self.analysis_results_data["FilterDepth_applied"]
            self.filter_depth_applied_checkbox = AnalysisFilterParamCheckbox("Depth Filter Applied", depth_filter_applied)
        else:
            self.filter_depth_applied_checkbox = None

        if "FilterEdge_applied" in self.analysis_results_data.columns:
            filter_edge_applied = self.analysis_results_data["FilterEdge_applied"]
            self.filter_edge_applied_checkbox = AnalysisFilterParamCheckbox("Edge Filter Applied", filter_edge_applied)

            percent_in_edge_filter = self.analysis_results_data["FilterEdge_percent_in_edge"].apply(lambda x: AnalysisWidget.analysis_data_2_list(x, dtype=float, scale=100))
            self.percent_in_edge_filter = AnalysisFilterParamRange(self, "Percent in Edge:", percent_in_edge_filter, unit="%")  
        else:
            self.filter_edge_applied_checkbox = None
            self.percent_in_edge_filter = None
        
        if "FilterPosition_applied" in self.analysis_results_data.columns:
            position_filter_applied = self.analysis_results_data["FilterPosition_applied"]
            self.filter_position_applied_checkbox = AnalysisFilterParamCheckbox("Position Filter Applied", position_filter_applied)

            distances_from_top = self.analysis_results_data["FilterPosition_distances_from_top"].apply(lambda x: AnalysisWidget.analysis_data_2_list(x, dtype=int, scale=1))
            self.distances_from_top_filter = AnalysisFilterParamRange(self, "Distance from Top (px):", distances_from_top, dtype=int, unit="pixels")

            self.distances_from_bottom = self.analysis_results_data["FilterPosition_distances_from_bottom"].apply(lambda x: AnalysisWidget.analysis_data_2_list(x, dtype=int, scale=1))
            self.distances_from_bottom_filter = AnalysisFilterParamRange(self, "Distance from Bottom (px):", self.distances_from_bottom, dtype=int, unit="pixels")
        else:
            self.filter_position_applied_checkbox = None
            self.distances_from_top_filter = None
            self.distances_from_bottom_filter = None


        self.filter_widgets: List[Union[AnalysisFilterParamRange, AnalysisFilterParamCheckbox]] = \
                                [self.object_type_filter_trees,
                                self.object_type_filter_posts,                                
                                self.contains_sprinkler_checkbox,
                                self.analysis_filter_widths, 
                                self.width_estimate_error_filter,
                                self.missing_data_checkbox,
                                self.analysis_filter_img_x_position,
                                self.largest_segment_finder_applied_checkbox,
                                self.filter_nms_applied_checkbox,
                                self.lacked_depth_data_checkbox,
                                self.depth_calc_num_points_filter,
                                self.filter_depth_applied_checkbox,
                                self.object_depth_filter,
                                self.filter_edge_applied_checkbox,
                                self.percent_in_edge_filter,
                                self.filter_position_applied_checkbox,
                                self.distances_from_top_filter,
                                self.distances_from_bottom_filter
                                ]
        
        for i, filter_widget in enumerate(self.filter_widgets):
            if filter_widget is None:
                self.filter_widgets.pop(i)
            filter_widget.filter_changed.connect(self.on_filter_changed)
            self.filter_selection_layout.addWidget(filter_widget)

        self.filter_selection_layout.addStretch()

    def initialize_ui(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        filter_selection_label = QLabel("Filter Images Based on Analysis Results:")
        set_widget_font(filter_selection_label, font_scale=1.2)
        self.main_layout.addWidget(filter_selection_label)

        self.filter_selection_scroll_area = QScrollArea()
        self.main_layout.addWidget(self.filter_selection_scroll_area) 

        self.current_num_images_label = QLabel(f"Images Remaining: 0/0")
        self.current_num_images_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_widget_font(self.current_num_images_label, font_scale=1.4)
        self.main_layout.addSpacing(10)
        self.main_layout.addWidget(self.current_num_images_label)
        self.main_layout.addSpacing(10)

        self.mean_width_error_label = QLabel("Mean Width Error: 0.0 mm")
        self.mean_width_error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_widget_font(self.mean_width_error_label, font_scale=1.1)
        self.main_layout.addWidget(self.mean_width_error_label)

        self.apply_filter_button = QPushButton("Apply Filters")
        set_widget_font(self.apply_filter_button, font_scale=1.25)
        self.apply_filter_button.setEnabled(False)
        self.apply_filter_button.clicked.connect(self.on_apply_filter)
        self.main_layout.addWidget(self.apply_filter_button)

        self.reset_filters_button = QPushButton("Reset Filters")
        self.reset_filters_button.clicked.connect(self.on_reset_filters)
        self.main_layout.addWidget(self.reset_filters_button)

    def load_analysis_data(self):

        self.analysis_results_data = pd.read_csv(self.package_paths.analysis_results_data_path)
        dataset_names = self.analysis_results_data["dataset"]
        image_names = self.analysis_results_data["image"]

        self.image_paths = [os.path.join(self.package_paths.analysis_gt_data_dir, dataset_name, "rgb", image_name) for dataset_name, image_name in zip(dataset_names, image_names)]

        self.image_paths = np.array(self.image_paths) 

        self.set_image_paths_func(self.image_paths, image_index=-1)

        self.reset_keep_idxs()

        self.initialize_filters()

        self.on_filter_changed()
    
    def reset_keep_idxs(self):
        self.keep_filtered_all = np.array([True] * len(self.image_paths))
        self.prev_keep_filtered_all = self.keep_filtered_all.copy()

    def on_apply_filter(self):
        self.prev_keep_filtered_all = self.keep_filtered_all.copy()

        self.on_filter_changed()

        if self.num_images_kept == 0:
            return

        self.mean_width_error_label.setText(f"Mean Width Error: {np.mean(self.width_estimate_error[self.keep_filtered_all]):.2f} mm")
        self.set_image_paths_func(self.image_paths[self.keep_filtered_all])
        self.apply_filter_button.setEnabled(False)
    
    @pyqtSlot()
    def on_filter_changed(self):

        self.keep_filtered_all = np.logical_and.reduce([filter_widget.to_keep for filter_widget in self.filter_widgets])

        self.num_images_kept = self.keep_filtered_all.sum()
        self.current_num_images_label.setText(f"Images Remaining: {self.num_images_kept}/{len(self.image_paths)}")
        
        if not np.all(self.prev_keep_filtered_all == self.keep_filtered_all) and self.num_images_kept > 0:
            self.apply_filter_button.setEnabled(True)
        else:
            self.apply_filter_button.setEnabled(False)
    
    @staticmethod
    def analysis_data_2_list(value, dtype=float, scale=1):
        if pd.isna(value):
            return []
        else:
            # Remove brackets and split by whitespace, then convert to the desired type
            numbers = re.sub(r"[\[\]]", "", value).split()
            # return [dtype(float(x) * scale) for x in numbers]
            return [dtype(float(x) * scale) for x in numbers]
    
    def on_reset_filters(self):
        for filter_widget in self.filter_widgets:
            filter_widget.set_to_initial_range()
        self.on_filter_changed()

class AnalysisFilterParamRange(QWidget):
    filter_changed = pyqtSignal()

    def __init__(self, parent, name: str, all_data: np.ndarray, value_range: Tuple[float, float] = None, dtype=float, single_value_per_image=False, unit: str = None):
        super().__init__(parent)

        self.name = name
        self.single_value_per_image = single_value_per_image
        self.value_range = value_range
        self.unit = unit
        

        if single_value_per_image:
            self.vals = all_data
            self.nan_idxs = np.isnan(self.vals)
            if self.value_range is None:
                self.value_range = (np.min(self.vals), np.max(self.vals))
            self.vals[self.nan_idxs] = 0
        else:
            self.min_vals = np.array([np.min(img_data) if len(img_data) > 0 else 0 for img_data in all_data], dtype=dtype)
            self.max_vals = np.array([np.max(img_data) if len(img_data) > 0 else 0 for img_data in all_data], dtype=dtype)

            if self.value_range is None:    
                self.nan_idxs = np.array([True if len(img_data) == 0 else False for img_data in all_data], dtype=bool)
                self.value_range = (np.min(self.min_vals[~self.nan_idxs]), np.max(self.max_vals[~self.nan_idxs]))

        self.to_keep = np.array([True] * len(all_data), dtype=bool)

        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        self.name_label = QLabel(self.name)
        self.name_label.setContentsMargins(0, 0, 5, 0)
        self.main_layout.addWidget(self.name_label)

        if dtype == int:
            self.value_slider = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        elif dtype == float:
            self.value_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)

        self.set_range(*self.value_range)
        self.set_values(self.value_range)
        
        self.value_slider.valueChanged.connect(self.on_slider_moved)
        self.main_layout.addWidget(self.value_slider)

        self.plot_button = QPushButton("Plot")
        self.plot_button.setFixedWidth(50)
        self.plot_button.clicked.connect(self.plot_data)
        self.main_layout.addWidget(self.plot_button)

    def set_range(self, min_val, max_val):
        self.value_slider.setRange(min_val, max_val)
    
    def on_slider_moved(self, vals):
        min_val, max_val = vals
        
        if self.single_value_per_image:
            self.to_keep = np.logical_and(self.vals >= min_val, self.vals <= max_val)
        else:
            self.to_keep = ((self.min_vals >= min_val) & (self.min_vals <= max_val)) | ((self.max_vals <= max_val)  & (self.max_vals >= min_val))
        
        if min_val == self.value_range[0] and max_val == self.value_range[1]:
            self.to_keep[self.nan_idxs] = True
        else:
            self.to_keep[self.nan_idxs] = False
        self.filter_changed.emit()
    
    def set_values(self, vals):
        self.blockSignals(True)
        self.value_slider.setValue(vals)
        self.blockSignals(False)
    
    def set_to_initial_range(self):
        self.set_values(self.value_range)
        self.on_slider_moved(self.value_range) 
    
    def plot_data(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

        if not self.single_value_per_image:
            plot_vals = []
            for max_val, min_val in zip(self.max_vals[~self.nan_idxs], self.min_vals[~self.nan_idxs]):
                plot_vals.append(max_val)
                if max_val != min_val:
                    plot_vals.append(max_val)
        else:
            plot_vals = self.vals[~self.nan_idxs]
                
        # Create a dialog window to act as the popup
        dialog = QDialog(self)
        dialog.setWindowTitle("Data Plot")
        dialog.setGeometry(100, 100, 600, 400)
        
        # Create a Figure and a FigureCanvas for the Matplotlib plot
        fig, ax = plt.subplots()
        ax.hist(plot_vals, bins=50)
        ax.set_title(self.name)
        if self.unit is not None:
            ax.set_xlabel(self.unit)
        ax.set_ylabel("Frequency")
        # add a vertical line at the min and max values
        current_min, current_max = self.value_slider.value()
        ax.axvline(current_min, color='r', linestyle='--')
        ax.axvline(current_max, color='r', linestyle='--')
        canvas = FigureCanvas(fig)
        
        # Set up the layout for the dialog
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        dialog.setLayout(layout)
        
        # Display the popup window
        dialog.exec()
        


class QScrollAreaCheckBox(QCheckBox):
    def __init__(self, name):
        super().__init__(name)

        checkbox_palette = self.palette()

        checkbox_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(255, 255, 255))
        checkbox_palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(0, 0, 0))

        self.setPalette(checkbox_palette)


class AnalysisFilterParamCheckbox(QScrollAreaCheckBox):
    filter_changed = pyqtSignal()

    def __init__(self, name: str, all_data: np.ndarray, initial_state: bool = False):

        # self.name = name
        self.initial_state = initial_state  

        self.filter_data = all_data.astype(bool)
        self.keep_all = np.array([True] * len(self.filter_data))

        self.name = f"{name} ({self.filter_data.sum()})"

        super().__init__(self.name)

        self.stateChanged.connect(self.on_state_changed)

        self.set_to_initial_range()
    
    def on_state_changed(self, state):
        if self.isChecked():
            self.to_keep = self.filter_data
        else:
            self.to_keep = self.keep_all
        self.filter_changed.emit()
    
    def set_to_initial_range(self):
        self.blockSignals(True)
        if self.initial_state:
            self.setCheckState(Qt.CheckState.Checked)
        else:
            self.setCheckState(Qt.CheckState.Unchecked)
        self.blockSignals(False)
        self.on_state_changed(None)

class RunAnalysisDialog(QDialog):
    def __init__(self, parent: QWidget, package_paths: PackagePaths, trunk_analyzer: TrunkAnalyzerAnalyzer):
        super().__init__(parent)

        self.package_paths = package_paths
        self.trunk_analyzer = trunk_analyzer

        self.abort_analysis = False

        self.initialize_ui()

    def initialize_ui(self):

        self.setFixedWidth(800)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        
        default_save_location = os.path.dirname(self.package_paths.analysis_results_dir)
        self.save_location_line_edit = QLineEdit(default_save_location)
        self.save_location_line_edit.setReadOnly(True)
        self.change_save_location_button = QPushButton("Change")
        self.change_save_location_button.clicked.connect(self.on_change_save_location)
        
        self.save_location_layout = QHBoxLayout()
        self.save_location_layout.addWidget(QLabel("Save Location:"))
        self.save_location_layout.addWidget(self.save_location_line_edit)
        self.save_location_layout.addWidget(self.change_save_location_button)
        self.main_layout.addLayout(self.save_location_layout)

        
        self.gt_datasets_dir_line_edit = QLineEdit(self.package_paths.analysis_gt_data_dir)
        self.gt_datasets_dir_line_edit.setReadOnly(True)
        self.change_gt_datasets_dir_button = QPushButton("Change")
        self.change_gt_datasets_dir_button.clicked.connect(self.on_change_gt_datasets_dir)

        self.gt_dataset_dir_layout = QHBoxLayout()
        self.gt_dataset_dir_layout.addWidget(QLabel("Ground Truth Datasets Directory:"))
        self.gt_dataset_dir_layout.addWidget(self.gt_datasets_dir_line_edit)
        self.gt_dataset_dir_layout.addWidget(self.change_gt_datasets_dir_button)
        self.main_layout.addLayout(self.gt_dataset_dir_layout)
       
        self.load_data_when_done_checkbox = QCheckBox("Load Dataset When Done")
        self.load_data_when_done_checkbox.setChecked(True)
        self.main_layout.addWidget(self.load_data_when_done_checkbox)

        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.clicked.connect(self.run_analysis)
        self.main_layout.addWidget(self.run_analysis_button)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.main_layout.addWidget(self.log_text_edit)

        self.log_handler = QTextEditLogger(self.log_text_edit)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)
    
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.main_layout.addWidget(self.progress_bar)
    
    def on_change_save_location(self):
        save_location = QFileDialog.getExistingDirectory(self, "Select Save Location", self.package_paths.analysis_results_dir)
        if save_location:
            self.save_location_line_edit.setText(save_location)
    
    def on_change_gt_datasets_dir(self):
        gt_datasets_dir = QFileDialog.getExistingDirectory(self, "Select Ground Truth Datasets Directory", self.package_paths.data_dir)
        if gt_datasets_dir:
            self.gt_datasets_dir_line_edit.setText(gt_datasets_dir)

    def set_progress(self, progress, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(progress)

        QApplication.processEvents()

    def set_running_state(self, starting: bool):
        self.trunk_analyzer.set_visualization_flag(not starting)
        self.run_analysis_button.setDisabled(starting)
        self.change_save_location_button.setDisabled(starting)
        self.change_gt_datasets_dir_button.setDisabled(starting)
        self.load_data_when_done_checkbox.setDisabled(starting)
                                                         
    def run_analysis(self):

        self.set_running_state(True)
        
        self.trunk_analyzer.do_performance_analysis(gt_datasets=True, from_app=True, progress_callback=self.set_progress)

        self.set_running_state(False)

        self.disconnect_log_handler()

        if self.load_data_when_done_checkbox.isChecked():
            self.accept()
        else:
            self.reject()
    
    def disconnect_log_handler(self):
        logging.getLogger().removeHandler(self.log_handler)
        self.log_handler.close()
        self.log_handler = None

    def closeEvent(self, event):
        self.trunk_analyzer.abort_analysis()
        self.trunk_analyzer.set_visualization_flag(True)    
        self.disconnect_log_handler()    

        event.accept()


class QTextEditLogger(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        # Format the log message and append it to the QTextEdit
        msg = self.format(record)
        self.text_edit.append(msg)
    
if __name__ == "__main__":

    # package_path = "/path/to/trunk_width_estimation"
    # os.environ['WIDTH_ESTIMATION_PACKAGE_PATH'] = package_path
    # package_data_path = "/path/to/trunk_width_estimation_package_data"
    # os.environ['WIDTH_ESTIMATION_PACKAGE_DATA_PATH'] = package_data_path

    initial_analysis_name = "sample_results"

    app = QApplication([])
    app.setApplicationDisplayName("Tuning App")
    window = MainWindow(initial_analysis_name)
    window.show()
    app.exec()
