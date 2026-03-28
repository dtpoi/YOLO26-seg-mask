"""
全景图Mask蒙版批量生成器 - Panorama Mask Generator
基于 YOLO26-seg 实例分割的全景图物体遮罩生成器

Copyright (C) 2026
Licensed under AGPL-3.0 License
"""

import flet as ft
import cv2
import numpy as np
import os
import sys
import threading
import torch
import locale
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Optional, Callable, Set

from ultralytics import YOLO
import py360convert


# ===== 资源路径处理（支持 PyInstaller 打包）=====
def resource_path(relative_path: str) -> str:
    """获取资源文件的绝对路径，支持开发环境和 PyInstaller 打包环境"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 打包后的临时目录
        base_path = sys._MEIPASS
    else:
        # 开发环境，使用 assets 目录
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    return os.path.join(base_path, relative_path)


# ===== 常量配置 =====
APP_VERSION = "1.0.0"
MODEL_NAME = "yolo26n-seg.pt"

# 默认参数
DEFAULT_KERNEL = 9
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.15
DEFAULT_FACE_W = 1024
DEFAULT_BATCH = True
DEFAULT_INVERT = True  # Mask反相默认勾选

# COCO 类别定义
COCO_CLASSES = {
    0: ("person", "人"),
    1: ("bicycle", "自行车"),
    2: ("car", "汽车"),
    3: ("motorcycle", "摩托车"),
    5: ("bus", "公交车"),
    7: ("truck", "卡车"),
    8: ("boat", "船"),
}


# ===== 多语言支持 =====
class I18n:
    """国际化文本管理"""
    
    TEXTS = {
        "zh_CN": {
            "app_name": "全景图Mask蒙版批量生成器",
            "subtitle": "基于 YOLO26-seg 的全景图物体遮罩生成器",
            "path_settings": "路径设置",
            "input_path": "输入路径",
            "input_hint": "选择包含全景图片的文件夹",
            "output_path": "输出路径",
            "output_hint": "选择输出文件夹",
            "browse": "浏览",
            "param_settings": "参数设置",
            "extract_targets": "提取目标",
            "kernel": "膨胀程度",
            "imgsz": "推理尺寸",
            "confidence": "置信度",
            "face_width": "Cubemap分辨率",
            "batch_mode": "启用批量推理（显存不足时可关闭）",
            "bypass_panorama": "绕过全景图拆分流程（适合平面图）",
            "invert_mask": "Mask反相（目标为黑，背景为白）",
            "filename_suffix": "文件名后缀",
            "reset_defaults": "恢复默认",
            "add_suffix": "添加后缀",
            "suffix_hint": "例如: _mask",
            "start": "开始处理",
            "stop": "停止",
            "about": "关于",
            "waiting": "等待开始...",
            "processing": "正在处理...",
            "stopping": "正在停止...",
            "stopped": "已中断",
            "completed": "完成！成功处理 {count}/{total} 张图片",
            "select_paths": "请选择输入和输出路径",
            "param_error": "参数格式错误: {error}",
            "model_loading": "正在加载模型...",
            "model_loaded": "模型加载完成 ({device})",
            "model_failed": "模型加载失败: {error}",
            "progress": "处理进度: {current}/{total}",
            "about_title": "关于 {name}",
            "version": "版本: {version}",
            "desc_line1": "全景图Mask蒙版批量生成器",
            "desc_line2": "基于 YOLO26-seg 实例分割模型",
            "project_link": "项目链接",
            "github": "GitHub: [待填写]",
            "dependencies": "开源依赖",
            "license": "许可证",
            "license_note": "本项目以 AGPL-3.0 协议开源。",
            "close": "关闭",
            "ok": "好的",
            "lang_cn": "中文",
            "lang_en": "EN",
            "param_info": "参数说明",
            "param_info_content": """• 膨胀程度：Mask边缘扩展像素，值越大遮罩范围越大（推荐7-11）
• 推理尺寸：YOLO推理分辨率，越高精度越高但速度越慢（推荐1280/1920）
• 置信度：检测阈值，越低检测越灵敏但可能有误检（推荐0.1-0.2）
• Cubemap分辨率：Cubemap每面分辨率，仅全景模式有效（推荐1024/1536）
• 批量推理：将CPU拆分的6个Cubemap打包推理，开启提升GPU利用率，内存显存不足建议关闭
• 绕过全景拆分：勾选后绕过全景图拆分流程，全景图用了会影响YOLO辨识度
• Mask反相：勾选后输出反相Mask（用于3dgs的mask层要勾选）
• 添加后缀：为输出文件名添加自定义后缀""",
        },
        "en_US": {
            "app_name": "Panorama Mask Generator",
            "subtitle": "Object mask generator based on YOLO26-seg",
            "path_settings": "Path Settings",
            "input_path": "Input Path",
            "input_hint": "Select folder containing panorama images",
            "output_path": "Output Path",
            "output_hint": "Select output folder",
            "browse": "Browse",
            "param_settings": "Parameters",
            "extract_targets": "Extract Targets",
            "kernel": "Dilation",
            "imgsz": "Inference Size",
            "confidence": "Confidence",
            "face_width": "Cubemap Width",
            "batch_mode": "Enable batch inference (disable if low VRAM)",
            "bypass_panorama": "Bypass panorama split (for flat images)",
            "invert_mask": "Invert mask (target=black, background=white)",
            "filename_suffix": "Filename Suffix",
            "reset_defaults": "Reset",
            "add_suffix": "Add suffix",
            "suffix_hint": "e.g.: _mask",
            "start": "Start",
            "stop": "Stop",
            "about": "About",
            "waiting": "Waiting...",
            "processing": "Processing...",
            "stopping": "Stopping...",
            "stopped": "Stopped",
            "completed": "Done! Processed {count}/{total} images",
            "select_paths": "Please select input and output paths",
            "param_error": "Invalid parameter: {error}",
            "model_loading": "Loading model...",
            "model_loaded": "Model loaded ({device})",
            "model_failed": "Failed to load model: {error}",
            "progress": "Progress: {current}/{total}",
            "about_title": "About {name}",
            "version": "Version: {version}",
            "desc_line1": "Panorama Mask Generator",
            "desc_line2": "Based on YOLO26-seg instance segmentation",
            "project_link": "Project",
            "github": "GitHub: [TBD]",
            "dependencies": "Dependencies",
            "license": "License",
            "license_note": "This project is open-sourced under AGPL-3.0.",
            "close": "Close",
            "ok": "OK",
            "lang_cn": "中文",
            "lang_en": "EN",
            "param_info": "Parameter Info",
            "param_info_content": """• Dilation: Mask edge expansion in pixels, larger = wider mask (recommended: 7-11)
• Inference Size: YOLO inference resolution, higher = more accurate but slower (recommended: 1280/1920)
• Confidence: Detection threshold, lower = more sensitive but may have false positives (recommended: 0.1-0.2)
• Cubemap Width: Resolution per cubemap face, panorama mode only (recommended: 1024/1536)
• Batch Inference: Pack 6 cubemap faces for GPU inference, improves utilization; disable if low memory/VRAM
• Bypass Panorama: Skip panorama split process, recommended for flat images
• Invert Mask: When checked, target is black and background is white; unchecked means target is white
• Add Suffix: Add custom suffix to output filenames""",
        }
    }
    
    def __init__(self, lang: str = "zh_CN"):
        self.lang = lang if lang in self.TEXTS else "en_US"
    
    def get(self, key: str, **kwargs) -> str:
        """获取翻译文本"""
        text = self.TEXTS.get(self.lang, self.TEXTS["en_US"]).get(key, key)
        return text.format(**kwargs) if kwargs else text
    
    def get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        if class_id in COCO_CLASSES:
            en_name, cn_name = COCO_CLASSES[class_id]
            return cn_name if self.lang == "zh_CN" else en_name
        return str(class_id)
    
    def switch_lang(self) -> str:
        """切换语言"""
        self.lang = "en_US" if self.lang == "zh_CN" else "zh_CN"
        return self.lang


def detect_system_language() -> str:
    """检测系统语言"""
    try:
        # Windows 系统
        import ctypes
        windll = ctypes.windll.kernel32
        lang_id = windll.GetUserDefaultUILanguage()
        # 常见语言 ID: 0x0804=简体中文, 0x0404=繁体中文
        if lang_id in (0x0804, 0x0404):
            return "zh_CN"
    except:
        pass
    
    try:
        # 通用方法
        sys_lang = locale.getdefaultlocale()[0]
        if sys_lang and sys_lang.startswith(("zh_CN", "zh_SG")):
            return "zh_CN"
        elif sys_lang and sys_lang.startswith(("zh_TW", "zh_HK")):
            return "zh_CN"  # 繁体中文也用简体中文界面
    except:
        pass
    
    return "en_US"


class MaskProcessor:
    """核心处理类：负责模型加载和图像处理"""
    
    def __init__(self, progress_callback: Optional[Callable] = None, i18n: Optional[I18n] = None):
        self.model: Optional[YOLO] = None
        self.progress_callback = progress_callback
        self.i18n = i18n or I18n()
        self.stop_flag = False
        self.model_loaded = False
        self.device = "CPU"
        
    def load_model(self) -> bool:
        """加载模型，首次运行会自动下载"""
        try:
            if self.progress_callback:
                self.progress_callback(self.i18n.get("model_loading"))
            
            # 检测 CUDA 可用性
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"检测到显卡: {torch.cuda.get_device_name(0)}")
                print(f"CUDA 版本: {torch.version.cuda}")
                print("torch:", torch.__version__)
            else:
                self.device = "cpu"
                print("未检测到 CUDA，使用 CPU")
                print("torch:", torch.__version__)
            
            # YOLO 会自动检查本地缓存，没有则从网络下载
            self.model = YOLO(MODEL_NAME)
            
            self.model_loaded = True
            if self.progress_callback:
                self.progress_callback(self.i18n.get("model_loaded", device=self.device.upper()))
            return True
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(self.i18n.get("model_failed", error=str(e)))
            return False
    
    def stop(self):
        """停止处理"""
        self.stop_flag = True
    
    def process_flat_image(
        self,
        img_path: str,
        kernel_size: int,
        imgsz: int,
        conf: float,
        target_classes: Set[int]
    ) -> Optional[np.ndarray]:
        """处理平面图（不进行全景拆分）"""
        if not self.model_loaded:
            return None
            
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 直接推理（显式指定 device）
        results = self.model(img, imgsz=imgsz, conf=conf, device=self.device, stream=False)[0]
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        if results.masks is not None:
            for i, cls in enumerate(results.boxes.cls):
                cls_id = int(cls)
                if cls_id in target_classes:
                    m = results.masks.data[i].cpu().numpy()
                    m = (m > 0.5).astype(np.uint8)
                    m = cv2.resize(
                        m,
                        (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    mask = np.maximum(mask, m)
        
        # 后处理
        mask = cv2.dilate(mask, kernel)
        mask = (mask > 0).astype(np.uint8) * 255
        # 不在此处反相，由调用方根据 invert_mask 参数决定
        
        return mask
    
    def process_panorama_image(
        self,
        img_path: str,
        kernel_size: int,
        imgsz: int,
        conf: float,
        face_w: int,
        target_classes: Set[int]
    ) -> Optional[np.ndarray]:
        """处理全景图（进行全景拆分）"""
        if not self.model_loaded:
            return None
            
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        h, w = img.shape[:2]
        
        # ERP -> Cubemap
        faces = py360convert.e2c(
            img,
            face_w=face_w,
            mode='bilinear',
            cube_format='dict'
        )
        
        mask_faces = {}
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        for face_name, face_img in faces.items():
            if self.stop_flag:
                return None
                
            # 推理（显式指定 device）
            results = self.model(face_img, imgsz=imgsz, conf=conf, device=self.device, stream=False)[0]
            
            face_mask = np.zeros(face_img.shape[:2], dtype=np.uint8)
            
            if results.masks is not None:
                for i, cls in enumerate(results.boxes.cls):
                    cls_id = int(cls)
                    if cls_id in target_classes:
                        m = results.masks.data[i].cpu().numpy()
                        m = (m > 0.5).astype(np.uint8)
                        m = cv2.resize(
                            m,
                            (face_img.shape[1], face_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                        face_mask = np.maximum(face_mask, m)
            
            mask_faces[face_name] = face_mask
        
        # Cubemap -> ERP
        mask_erp = py360convert.c2e(
            mask_faces,
            h=h, w=w,
            cube_format='dict'
        )
        
        # 后处理
        mask_erp = cv2.dilate(mask_erp, kernel)
        mask_erp = (mask_erp > 0).astype(np.uint8) * 255
        # 不在此处反相，由调用方根据 invert_mask 参数决定
        
        return mask_erp
    
    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        kernel_size: int,
        imgsz: int,
        conf: float,
        face_w: int,
        use_batch: bool,
        bypass_panorama: bool,
        invert_mask: bool,
        suffix_enabled: bool,
        suffix_text: str,
        target_classes: Set[int]
    ) -> tuple[int, int]:
        """
        批量处理图片
        返回: (成功数量, 总数量)
        """
        if not self.model_loaded:
            self.load_model()
            if not self.model_loaded:
                return 0, 0
        
        self.stop_flag = False
        
        # 获取所有图片文件
        files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not files:
            return 0, 0
        
        os.makedirs(output_dir, exist_ok=True)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        success_count = 0
        
        # 根据是否绕过全景选择处理函数
        if bypass_panorama:
            # 平面图模式：直接推理
            for idx, name in enumerate(files):
                if self.stop_flag:
                    break
                
                img_path = os.path.join(input_dir, name)
                mask = self.process_flat_image(img_path, kernel_size, imgsz, conf, target_classes)
                
                if mask is not None:
                    # 应用反相设置（勾选时反相：目标为黑背景为白；取消勾选不反相：目标为白背景为黑）
                    if invert_mask:
                        mask = 255 - mask
                    
                    # 处理文件名
                    output_name = name
                    if suffix_enabled and suffix_text:
                        base, ext = os.path.splitext(name)
                        output_name = base + suffix_text + ext
                    
                    output_path = os.path.join(output_dir, output_name)
                    cv2.imwrite(output_path, mask)
                    success_count += 1
                
                if self.progress_callback:
                    self.progress_callback(
                        self.i18n.get("progress", current=idx+1, total=len(files))
                    )
        elif use_batch:
            # 批量推理模式（全景图）
            batch_size = 4
            
            for i in range(0, len(files), batch_size):
                if self.stop_flag:
                    break
                    
                batch_files = files[i:i+batch_size]
                all_faces = []
                meta = []
                
                # 收集所有面
                for name in batch_files:
                    path = os.path.join(input_dir, name)
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    
                    h, w = img.shape[:2]
                    faces = py360convert.e2c(
                        img, face_w=face_w, mode='bilinear', cube_format='dict'
                    )
                    
                    for face_name, face_img in faces.items():
                        all_faces.append(face_img)
                        meta.append((name, face_name, h, w))
                
                # 批量推理（显式指定 device）
                results_list = self.model(all_faces, imgsz=imgsz, conf=conf, device=self.device, stream=False)
                
                # 结果处理
                mask_faces_dict = {}
                
                for idx, results in enumerate(results_list):
                    name, face_name, h, w = meta[idx]
                    face_img = all_faces[idx]
                    
                    if name not in mask_faces_dict:
                        mask_faces_dict[name] = {}
                    
                    face_mask = np.zeros(face_img.shape[:2], dtype=np.uint8)
                    
                    if results.masks is not None:
                        for j, cls in enumerate(results.boxes.cls):
                            cls_id = int(cls)
                            if cls_id in target_classes:
                                m = results.masks.data[j].cpu().numpy()
                                m = (m > 0.5).astype(np.uint8)
                                m = cv2.resize(
                                    m,
                                    (face_img.shape[1], face_img.shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                )
                                face_mask = np.maximum(face_mask, m)
                    
                    mask_faces_dict[name][face_name] = face_mask
                
                # 拼回ERP并保存
                for name in batch_files:
                    if self.stop_flag:
                        break
                    
                    if name not in mask_faces_dict:
                        continue
                    
                    # 获取原始尺寸
                    for m in meta:
                        if m[0] == name:
                            h, w = m[2], m[3]
                            break
                    
                    mask_erp = py360convert.c2e(
                        mask_faces_dict[name],
                        h=h, w=w,
                        cube_format='dict'
                    )
                    
                    mask_erp = cv2.dilate(mask_erp, kernel)
                    mask_erp = (mask_erp > 0).astype(np.uint8) * 255
                    
                    # 应用反相设置（勾选时反相：目标为黑背景为白；取消勾选不反相：目标为白背景为黑）
                    if invert_mask:
                        mask_erp = 255 - mask_erp
                    
                    # 处理文件名
                    output_name = name
                    if suffix_enabled and suffix_text:
                        base, ext = os.path.splitext(name)
                        output_name = base + suffix_text + ext
                    
                    output_path = os.path.join(output_dir, output_name)
                    cv2.imwrite(output_path, mask_erp)
                    success_count += 1
                
                if self.progress_callback:
                    self.progress_callback(
                        self.i18n.get("progress", current=min(i+batch_size, len(files)), total=len(files))
                    )
        else:
            # 单张处理模式（全景图）
            for idx, name in enumerate(files):
                if self.stop_flag:
                    break
                
                img_path = os.path.join(input_dir, name)
                mask = self.process_panorama_image(img_path, kernel_size, imgsz, conf, face_w, target_classes)
                
                if mask is not None:
                    # 应用反相设置（勾选时反相：目标为黑背景为白；取消勾选不反相：目标为白背景为黑）
                    if invert_mask:
                        mask = 255 - mask
                    
                    # 处理文件名
                    output_name = name
                    if suffix_enabled and suffix_text:
                        base, ext = os.path.splitext(name)
                        output_name = base + suffix_text + ext
                    
                    output_path = os.path.join(output_dir, output_name)
                    cv2.imwrite(output_path, mask)
                    success_count += 1
                
                if self.progress_callback:
                    self.progress_callback(
                        self.i18n.get("progress", current=idx+1, total=len(files))
                    )
        
        return success_count, len(files)


class MaskYOLOApp:
    """Flet GUI 应用"""
    
    def __init__(self, page: ft.Page):
        self.page = page
        
        # 初始化国际化
        system_lang = detect_system_language()
        self.i18n = I18n(system_lang)
        
        # 初始化处理器
        self.processor = MaskProcessor(self.update_progress, self.i18n)
        
        # 目标类别选择状态
        self.target_classes: Set[int] = {0}  # 默认只选人
        self.class_checkboxes = {}
        
        self.setup_page()
        self.create_ui()
        
    def setup_page(self):
        """设置页面属性"""
        app_name = self.i18n.get("app_name")
        self.page.title = f"{app_name} v{APP_VERSION}"
        self.page.window.width = 750
        self.page.window.height = 800
        self.page.window.min_width = 650
        self.page.window.min_height = 700
        self.page.theme_mode = ft.ThemeMode.SYSTEM
        self.page.padding = 20
        self.page.fonts = {
            "Sarasa": resource_path("fonts/SarasaUiSC-Regular.ttf"),
        }
        self.page.theme = ft.Theme(font_family="Sarasa")
        
    def create_ui(self):
        """创建 UI 组件"""
        
        # 标题栏（含语言切换按钮）
        title_text = ft.Text(
            self.i18n.get('app_name'),
            size=22,
            weight=ft.FontWeight.BOLD
        )
        
        self.subtitle = ft.Text(
            self.i18n.get("subtitle"),
            size=11,
            color=ft.Colors.GREY_600
        )
        
        # 语言切换按钮组（胶囊样式）
        self.lang_cn_btn = ft.Container(
            content=ft.Text("中文", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
            bgcolor=ft.Colors.BLUE_700,
            border_radius=ft.BorderRadius(15, 15, 15, 15),
            padding=ft.Padding(left=10, right=10, top=4, bottom=4),
            on_click=lambda e: self.set_language("zh_CN"),
            ink=True,
        )
        self.lang_en_btn = ft.Container(
            content=ft.Text("EN", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_600),
            bgcolor=ft.Colors.GREY_300,
            border_radius=ft.BorderRadius(15, 15, 15, 15),
            padding=ft.Padding(left=10, right=10, top=4, bottom=4),
            on_click=lambda e: self.set_language("en_US"),
            ink=True,
        )
        
        lang_row = ft.Row([
            ft.IconButton(
                icon=ft.Icons.LANGUAGE,
                tooltip="中文/English",
                on_click=self.switch_language
            ),
            self.lang_cn_btn,
            self.lang_en_btn,
        ], spacing=3)
        
        title_row = ft.Row([
            ft.Column([
                title_text,
                self.subtitle,
            ], spacing=2),
            lang_row,
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        
        # 路径选择区
        self.path_label = ft.Text(
            self.i18n.get("path_settings"),
            size=16,
            weight=ft.FontWeight.BOLD
        )
        
        self.input_path = ft.TextField(
            label=self.i18n.get("input_path"),
            expand=True,
            hint_text=self.i18n.get("input_hint")
        )
        
        self.output_path = ft.TextField(
            label=self.i18n.get("output_path"),
            expand=True,
            hint_text=self.i18n.get("output_hint")
        )
        
        self.input_browse_btn = ft.Button(
            self.i18n.get("browse"),
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self.select_input
        )
        
        self.output_browse_btn = ft.Button(
            self.i18n.get("browse"),
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self.select_output
        )
        
        input_row = ft.Row([self.input_path, self.input_browse_btn])
        output_row = ft.Row([self.output_path, self.output_browse_btn])
        
        # ===== 左侧：参数设置 =====
        self.kernel_input = ft.TextField(
            label=self.i18n.get("kernel"),
            value=str(DEFAULT_KERNEL),
            width=100,
            text_align=ft.TextAlign.CENTER
        )
        self.imgsz_input = ft.TextField(
            label=self.i18n.get("imgsz"),
            value=str(DEFAULT_IMGSZ),
            width=100,
            text_align=ft.TextAlign.CENTER
        )
        self.conf_input = ft.TextField(
            label=self.i18n.get("confidence"),
            value=str(DEFAULT_CONF),
            width=100,
            text_align=ft.TextAlign.CENTER
        )
        self.face_w_input = ft.TextField(
            label=self.i18n.get("face_width"),
            value=str(DEFAULT_FACE_W),
            width=100,
            text_align=ft.TextAlign.CENTER
        )
        self.batch_checkbox = ft.Checkbox(
            label=self.i18n.get("batch_mode"),
            value=DEFAULT_BATCH,
        )
        self.bypass_checkbox = ft.Checkbox(
            label=self.i18n.get("bypass_panorama"),
            value=False,
        )
        self.invert_checkbox = ft.Checkbox(
            label=self.i18n.get("invert_mask"),
            value=DEFAULT_INVERT,
        )
        
        # 复选框组（紧凑间距）
        checkboxes_column = ft.Column([
            self.batch_checkbox,
            self.bypass_checkbox,
            self.invert_checkbox,
        ], spacing=0)
        
        # 后缀设置
        self.suffix_checkbox = ft.Checkbox(
            label=self.i18n.get("add_suffix"),
            value=False,
            on_change=self.toggle_suffix
        )
        self.suffix_input = ft.TextField(
            hint_text=self.i18n.get("suffix_hint"),
            value="_mask",
            width=120,
            disabled=True
        )
        suffix_row = ft.Row([
            self.suffix_checkbox,
            self.suffix_input,
        ], alignment=ft.MainAxisAlignment.START)
        
        # 参数区标题
        self.param_label = ft.Text(
            self.i18n.get("param_settings"),
            size=16,
            weight=ft.FontWeight.BOLD
        )
        
        # 参数区标题行（含折叠按钮和恢复默认按钮）
        self.param_info_expanded = False
        self.param_info_icon = ft.IconButton(
            icon=ft.Icons.INFO_OUTLINE,
            icon_size=18,
            tooltip=self.i18n.get("param_info"),
            on_click=self.toggle_param_info
        )
        
        self.reset_btn = ft.TextButton(
            self.i18n.get("reset_defaults"),
            icon=ft.Icons.RESTORE,
            on_click=self.reset_defaults,
            style=ft.ButtonStyle(
                padding=ft.Padding(5, 0, 5, 0),
            )
        )
        
        self.param_label_row = ft.Row([
            self.param_label,
            self.param_info_icon,
            ft.Container(expand=True),
            self.reset_btn,
        ], alignment=ft.MainAxisAlignment.START, spacing=5)
        
        # 参数说明（可折叠）
        self.param_info_text = ft.Container(
            content=ft.Column([
                ft.Text(self.i18n.get("param_info_content"), size=11, color=ft.Colors.GREY_700)
            ]),
            padding=10,
            bgcolor=ft.Colors.GREY_100,
            border_radius=5,
            visible=False,
        )
        
        # 左侧参数卡片
        left_params = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    self.param_label_row,
                    self.param_info_text,
                    ft.Row([
                        self.kernel_input,
                        self.imgsz_input,
                        self.conf_input,
                        self.face_w_input,
                    ], wrap=True),
                    checkboxes_column,
                    suffix_row,
                    ft.Container(expand=True),  # 填充剩余空间
                ], spacing=10, expand=True),
                padding=12,
            ),
            elevation=2,
            expand=True,
        )
        
        # ===== 右侧：提取目标 =====
        self.extract_label = ft.Text(
            self.i18n.get("extract_targets"),
            size=16,
            weight=ft.FontWeight.BOLD
        )
        
        # 创建类别选择复选框
        class_checkboxes_list = []
        for class_id, (en_name, cn_name) in COCO_CLASSES.items():
            default_checked = (class_id == 0)  # 默认只勾选人
            checkbox = ft.Checkbox(
                label=f"{cn_name}" if self.i18n.lang == "zh_CN" else en_name,
                value=default_checked,
                on_change=lambda e, cid=class_id: self.toggle_class(cid, e)
            )
            self.class_checkboxes[class_id] = checkbox
            class_checkboxes_list.append(checkbox)
        
        # 复选框组（紧凑间距，与左侧一致）
        class_checkboxes_column = ft.Column(class_checkboxes_list, spacing=2)
        
        # 右侧提取目标卡片
        right_params = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    self.extract_label,
                    class_checkboxes_column,
                    ft.Container(expand=True),  # 填充剩余空间
                ], spacing=8, expand=True),
                padding=12,
            ),
            elevation=2,
            expand=True,
        )
        
        # 两列布局
        params_row = ft.Row([
            left_params,
            right_params,
        ], spacing=15, expand=True)
        
        # 控制按钮
        self.start_btn = ft.Button(
            self.i18n.get("start"),
            icon=ft.Icons.PLAY_ARROW,
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.GREEN,
            on_click=self.start_processing,
            width=140
        )
        
        self.stop_btn = ft.Button(
            self.i18n.get("stop"),
            icon=ft.Icons.STOP,
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.RED_700,
            on_click=self.stop_processing,
            width=140,
            disabled=True
        )
        
        self.about_btn = ft.TextButton(
            self.i18n.get("about"),
            icon=ft.Icons.INFO_OUTLINE,
            on_click=self.show_about
        )
        
        btn_row = ft.Row([
            self.start_btn,
            self.stop_btn,
            self.about_btn
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
        
        # 进度显示
        self.progress_text = ft.Text(
            self.i18n.get("waiting"),
            size=14,
            text_align=ft.TextAlign.CENTER
        )
        
        self.progress_bar = ft.ProgressBar(
            width=400,
            bar_height=8,
            visible=False
        )
        
        # 组装界面
        self.page.add(
            ft.Column([
                title_row,
                ft.Divider(height=15),
                
                input_row,
                output_row,
                
                ft.Divider(height=12),
                params_row,
                
                ft.Divider(height=15),
                btn_row,
                
                ft.Container(height=5),
                self.progress_text,
                self.progress_bar,
                
            ], scroll=ft.ScrollMode.AUTO)
        )
        
        # 页面加载时预加载模型
        threading.Thread(target=self.preload_model, daemon=True).start()
    
    def toggle_param_info(self, e):
        """切换参数说明展开/折叠"""
        self.param_info_expanded = not self.param_info_expanded
        self.param_info_text.visible = self.param_info_expanded
        self.param_info_icon.icon = ft.Icons.INFO if self.param_info_expanded else ft.Icons.INFO_OUTLINE
        self.page.update()
    
    def toggle_class(self, class_id: int, e):
        """切换目标类别选择"""
        if e.control.value:
            self.target_classes.add(class_id)
        else:
            self.target_classes.discard(class_id)
    
    def toggle_suffix(self, e):
        """切换后缀输入框状态"""
        self.suffix_input.disabled = not self.suffix_checkbox.value
        self.page.update()
    
    def set_language(self, lang: str):
        """设置语言"""
        if self.i18n.lang != lang:
            self.i18n.lang = lang
            self.update_lang_buttons()
            self.refresh_ui()
    
    def update_lang_buttons(self):
        """更新语言按钮状态"""
        if self.i18n.lang == "zh_CN":
            self.lang_cn_btn.bgcolor = ft.Colors.BLUE_700
            self.lang_cn_btn.content.color = ft.Colors.WHITE
            self.lang_en_btn.bgcolor = ft.Colors.GREY_300
            self.lang_en_btn.content.color = ft.Colors.GREY_600
        else:
            self.lang_cn_btn.bgcolor = ft.Colors.GREY_300
            self.lang_cn_btn.content.color = ft.Colors.GREY_600
            self.lang_en_btn.bgcolor = ft.Colors.BLUE_700
            self.lang_en_btn.content.color = ft.Colors.WHITE
    
    def switch_language(self, e):
        """切换语言"""
        self.i18n.switch_lang()
        self.update_lang_buttons()
        self.refresh_ui()
    
    def refresh_ui(self):
        """刷新界面文本"""
        # 更新标题
        self.page.title = f"{self.i18n.get('app_name')} v{APP_VERSION}"
        self.page.controls[0].controls[0].controls[0].value = f"🖼️ {self.i18n.get('app_name')}"
        self.subtitle.value = self.i18n.get("subtitle")
        
        # 更新路径区
        self.path_label.value = self.i18n.get("path_settings")
        self.input_path.label = self.i18n.get("input_path")
        self.input_path.hint_text = self.i18n.get("input_hint")
        self.output_path.label = self.i18n.get("output_path")
        self.output_path.hint_text = self.i18n.get("output_hint")
        self.input_browse_btn.text = self.i18n.get("browse")
        self.output_browse_btn.text = self.i18n.get("browse")
        
        # 更新参数区
        self.param_label.value = self.i18n.get("param_settings")
        self.extract_label.value = self.i18n.get("extract_targets")
        self.kernel_input.label = self.i18n.get("kernel")
        self.imgsz_input.label = self.i18n.get("imgsz")
        self.conf_input.label = self.i18n.get("confidence")
        self.face_w_input.label = self.i18n.get("face_width")
        self.batch_checkbox.label = self.i18n.get("batch_mode")
        self.bypass_checkbox.label = self.i18n.get("bypass_panorama")
        self.invert_checkbox.label = self.i18n.get("invert_mask")
        self.suffix_checkbox.label = self.i18n.get("add_suffix")
        self.suffix_input.hint_text = self.i18n.get("suffix_hint")
        
        # 更新类别复选框标签
        for class_id, checkbox in self.class_checkboxes.items():
            en_name, cn_name = COCO_CLASSES[class_id]
            checkbox.label = cn_name if self.i18n.lang == "zh_CN" else en_name
        
        # 更新参数说明
        self.param_info_text.content.controls[0].value = self.i18n.get("param_info_content")
        self.param_info_icon.tooltip = self.i18n.get("param_info")
        self.reset_btn.text = self.i18n.get("reset_defaults")
        
        # 更新按钮
        self.start_btn.text = self.i18n.get("start")
        self.stop_btn.text = self.i18n.get("stop")
        self.about_btn.text = self.i18n.get("about")
        
        # 更新进度文本
        self.progress_text.value = self.i18n.get("waiting")
        
        self.page.update()
    
    def preload_model(self):
        """后台预加载模型"""
        self.processor.load_model()
        self.page.update()
    
    def reset_defaults(self, e):
        """恢复默认参数"""
        self.kernel_input.value = str(DEFAULT_KERNEL)
        self.imgsz_input.value = str(DEFAULT_IMGSZ)
        self.conf_input.value = str(DEFAULT_CONF)
        self.face_w_input.value = str(DEFAULT_FACE_W)
        self.batch_checkbox.value = DEFAULT_BATCH
        self.bypass_checkbox.value = False
        self.invert_checkbox.value = DEFAULT_INVERT
        self.suffix_checkbox.value = False
        self.suffix_input.value = "_mask"
        self.suffix_input.disabled = True
        self.page.update()
    
    def update_progress(self, message: str):
        """更新进度显示"""
        self.progress_text.value = message
        self.page.update()
    
    def select_input(self, e):
        """选择输入路径 - 使用系统文件夹选择对话框"""
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.wm_attributes('-topmost', 1)  # 置顶
        result = filedialog.askdirectory(
            title="选择输入文件夹" if self.i18n.lang == "zh_CN" else "Select Input Folder"
        )
        root.destroy()
        
        if result:
            self.input_path.value = result
            self.page.update()
    
    def select_output(self, e):
        """选择输出路径 - 使用系统文件夹选择对话框"""
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.wm_attributes('-topmost', 1)  # 置顶
        result = filedialog.askdirectory(
            title="选择输出文件夹" if self.i18n.lang == "zh_CN" else "Select Output Folder"
        )
        root.destroy()
        
        if result:
            self.output_path.value = result
            self.page.update()
    
    def start_processing(self, e):
        """开始处理"""
        if not self.input_path.value or not self.output_path.value:
            self.show_snackbar(self.i18n.get("select_paths"))
            return
        
        # 检查是否选择了目标类别
        if not self.target_classes:
            self.show_snackbar("请至少选择一个提取目标" if self.i18n.lang == "zh_CN" else "Please select at least one target")
            return
        
        # 获取参数
        try:
            kernel = int(self.kernel_input.value)
            imgsz = int(self.imgsz_input.value)
            conf = float(self.conf_input.value)
            face_w = int(self.face_w_input.value)
            use_batch = self.batch_checkbox.value
            bypass_panorama = self.bypass_checkbox.value
            invert_mask = self.invert_checkbox.value
            suffix_enabled = self.suffix_checkbox.value
            suffix_text = self.suffix_input.value
            
        except ValueError as ex:
            self.show_snackbar(self.i18n.get("param_error", error=str(ex)))
            return
        
        # 更新UI状态
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.progress_bar.visible = True
        self.progress_text.value = self.i18n.get("processing")
        self.page.update()
        
        # 后台线程处理
        def process_task():
            success, total = self.processor.process_batch(
                self.input_path.value,
                self.output_path.value,
                kernel, imgsz, conf, face_w, use_batch,
                bypass_panorama, invert_mask, suffix_enabled, suffix_text,
                self.target_classes.copy()
            )
            
            # 恢复UI状态
            self.start_btn.disabled = False
            self.stop_btn.disabled = True
            self.progress_bar.visible = False
            
            if self.processor.stop_flag:
                self.progress_text.value = self.i18n.get("stopped")
            else:
                self.progress_text.value = self.i18n.get("completed", count=success, total=total)
            
            self.page.update()
            
            if success > 0 and not self.processor.stop_flag:
                self.show_snackbar(self.i18n.get("completed", count=success, total=total))
        
        threading.Thread(target=process_task, daemon=True).start()
    
    def stop_processing(self, e):
        """停止处理"""
        self.processor.stop()
        self.progress_text.value = self.i18n.get("stopping")
        self.stop_btn.disabled = True
        self.page.update()
    
    def show_snackbar(self, message: str):
        """显示提示消息"""
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text(message),
            action=self.i18n.get("ok")
        )
        self.page.snack_bar.open = True
        self.page.update()
    
    def show_about(self, e):
        """显示关于对话框"""
        app_name = self.i18n.get("app_name")
        about_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(self.i18n.get("about_title", name=app_name)),
            content=ft.Column([
                ft.Text(self.i18n.get("version", version=APP_VERSION), size=16, weight=ft.FontWeight.BOLD),
                ft.Divider(height=10),
                ft.Text(self.i18n.get("desc_line1"), size=14),
                ft.Text(self.i18n.get("desc_line2"), size=12),
                ft.Divider(height=10),
                ft.Text(self.i18n.get("project_link"), size=14, weight=ft.FontWeight.BOLD),
                ft.TextButton(
                    content=ft.Text("dtpoi/YOLO26-seg-mask", size=12, color=ft.Colors.BLUE),
                    url="https://github.com/dtpoi/YOLO26-seg-mask",
                    style=ft.ButtonStyle(padding=0)
                ),
                ft.Divider(height=10),
                ft.Text(self.i18n.get("dependencies"), size=14, weight=ft.FontWeight.BOLD),
                ft.Text("• ultralytics - YOLO26", size=11),
                ft.Text("• opencv-python", size=11),
                ft.Text("• py360convert", size=11),
                ft.Text("• flet", size=11),
                ft.Text("• torch/torchvision", size=11),
                ft.Divider(height=10),
                ft.Text(self.i18n.get("license"), size=14, weight=ft.FontWeight.BOLD),
                ft.Text("AGPL-3.0 License", size=12),
                ft.Text(self.i18n.get("license_note"), size=11),
            ], tight=True, scroll=ft.ScrollMode.AUTO),
            actions=[
                ft.TextButton(self.i18n.get("close"), on_click=lambda e: self.page.pop_dialog())
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        
        self.page.show_dialog(about_dialog)


def main(page: ft.Page):
    """应用入口"""
    MaskYOLOApp(page)


if __name__ == "__main__":
    ft.run(main)
