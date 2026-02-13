from dataclasses import dataclass
from collections import deque
from datetime import datetime
from enum import IntEnum
from functools import lru_cache
from typing import Optional, List
import csv
import subprocess
import psutil
from PySide6.QtCore import Qt, QObject, QPointF, QTimer, QThread, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QProgressBar, QMenu
from PySide6.QtGui import QPainter, QColor, QPolygon, QPainterPath, QPen, QPixmap, QLinearGradient
from math import sin, cos, pi

PALETTE = {
    "CPU": "#FF4136",
    "RAM": "#B10DC9",
    "GPU": "#0074D9",
    "VRAM": "#2ECC40",
    "GPU_POWER": "#FFD700"
}

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    ram_usage_percent: float
    gpu_utilization: Optional[float] = None
    vram_usage_percent: Optional[float] = None
    power_usage_percent: Optional[float] = None
    power_limit_percent: Optional[float] = None

def is_nvidia_gpu_available():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

HAS_NVIDIA_GPU = is_nvidia_gpu_available()
if HAS_NVIDIA_GPU:
    import pynvml
    pynvml.nvmlInit()
    HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    def _shutdown_nvml():
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
else:
    HANDLE = None

class MetricsStore(QObject):
    metrics_added = Signal(object)
    def __init__(self, buffer_size: int = 100):
        super().__init__()
        self._history: deque[SystemMetrics] = deque(maxlen=buffer_size)
    def add_metrics(self, metrics: SystemMetrics) -> None:
        self._history.append(metrics)
        self.metrics_added.emit(metrics)
    def subscribe(self, callback):
        self.metrics_added.connect(callback)
    def unsubscribe(self, callback):
        try:
            self.metrics_added.disconnect(callback)
        except Exception:
            pass
    @property
    def history(self) -> List[SystemMetrics]:
        return list(self._history)

class BatchCSVLogger(QObject):
    def __init__(self, filepath: str, flush_interval: int = 5000):
        super().__init__()
        self.filepath = filepath
        self.flush_interval = flush_interval
        self.buffer = []
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['timestamp', 'cpu_usage', 'ram_usage_percent', 'gpu_utilization', 'vram_usage_percent', 'power_usage_percent'])
        self.timer = QTimer(self)
        self.timer.setInterval(self.flush_interval)
        self.timer.timeout.connect(self.flush)
        self.timer.start()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    def log(self, metrics):
        self.buffer.append(metrics)
    def flush(self):
        if not self.buffer:
            return
        for m in self.buffer:
            self.writer.writerow([m.timestamp.isoformat(), m.cpu_usage, m.ram_usage_percent, m.gpu_utilization if m.gpu_utilization is not None else '', m.vram_usage_percent if m.vram_usage_percent is not None else '', m.power_usage_percent if m.power_usage_percent is not None else ''])
        self.file.flush()
        self.buffer.clear()
    def close(self):
        self.timer.stop()
        self.flush()
        self.file.close()
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

def collect_cpu_metrics():
    cpu_times = psutil.cpu_times_percent(interval=None, percpu=True)
    cpu_percentages = []
    for cpu in cpu_times:
        total_active = sum(v for f, v in cpu._asdict().items() if f not in ('idle', 'iowait'))
        cpu_percentages.append(total_active)
    return sum(cpu_percentages) / len(cpu_percentages)

def collect_ram_metrics():
    ram = psutil.virtual_memory()
    return ram.percent, ram.used

def collect_gpu_metrics(handle):
    if handle is None:
        return None, None
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    vram_usage_percent = (memory_info.used / memory_info.total) * 100 if memory_info.total else 0
    return gpu_utilization, vram_usage_percent

def collect_power_metrics(handle):
    if handle is None:
        return None, None
    try:
        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except pynvml.NVMLError:
        return None, None
    try:
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
    except pynvml.NVMLError_NotSupported:
        try:
            power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        except pynvml.NVMLError:
            power_limit = None
    if power_limit and power_limit > 0:
        power_percentage = (power_usage / power_limit) * 100
    else:
        power_percentage = 0
    return power_percentage, power_limit

class MetricsCollectorThread(QThread):
    metrics_updated = Signal(object)
    def __init__(self, interval: int = 200):
        super().__init__()
        self.interval = interval
        self.gpu_available = HAS_NVIDIA_GPU
    def _collect_once(self):
        try:
            cpu_usage = collect_cpu_metrics()
            ram_usage_percent, _ = collect_ram_metrics()
            if self.gpu_available:
                gpu_util, vram_usage = collect_gpu_metrics(HANDLE)
                power_usage, power_limit = collect_power_metrics(HANDLE)
            else:
                gpu_util = vram_usage = power_usage = power_limit = None
            metrics = SystemMetrics(timestamp=datetime.now(), cpu_usage=cpu_usage, ram_usage_percent=ram_usage_percent, gpu_utilization=gpu_util, vram_usage_percent=vram_usage, power_usage_percent=power_usage, power_limit_percent=power_limit)
            self.metrics_updated.emit(metrics)
        except Exception as e:
            print(f"Error collecting metrics: {e}")
    def run(self):
        timer = QTimer()
        timer.setInterval(self.interval)
        timer.timeout.connect(self._collect_once)
        timer.start()
        self.exec()
    def stop(self):
        self.quit()
        self.wait()

class BaseVisualization(QWidget):
    _metric_mappings = []
    _gpu_metric_mappings = []
    def __init__(self, metrics_store: MetricsStore):
        super().__init__()
        self.metrics_store = metrics_store
        self.metrics_store.subscribe(self.update_metrics)
        self.has_nvidia_gpu = HAS_NVIDIA_GPU
    def _update_widget(self, widget, value):
        raise NotImplementedError
    def _format_label(self, prefix, value):
        return f"{prefix} {value:.1f}%"
    def update_metrics(self, m: SystemMetrics):
        for attr_name, widget_attr, label_attr, prefix in self._metric_mappings:
            value = getattr(m, attr_name)
            self._update_widget(getattr(self, widget_attr), value)
            getattr(self, label_attr).setText(self._format_label(prefix, value))
        if self.has_nvidia_gpu:
            for attr_name, widget_attr, label_attr, prefix in self._gpu_metric_mappings:
                value = getattr(m, attr_name)
                if value is not None:
                    self._update_widget(getattr(self, widget_attr), value)
                    getattr(self, label_attr).setText(self._format_label(prefix, value))
    def cleanup(self):
        self.metrics_store.unsubscribe(self.update_metrics)

def color_for(name: str) -> str:
    return PALETTE[name]

class BarVisualization(BaseVisualization):
    def __init__(self, metrics_store: MetricsStore):
        super().__init__(metrics_store)
        self.initUI()
        self._metric_mappings = [
            ("cpu_usage", "cpu_bar", "cpu_percent_label", "CPU"),
            ("ram_usage_percent", "ram_bar", "ram_percent_label", "RAM"),
        ]
        if self.has_nvidia_gpu:
            self._gpu_metric_mappings = [
                ("gpu_utilization", "gpu_bar", "gpu_percent_label", "GPU"),
                ("vram_usage_percent", "vram_bar", "vram_percent_label", "VRAM"),
                ("power_usage_percent", "power_bar", "power_percent_label", "GPU Power"),
            ]
    def _update_widget(self, widget, value):
        widget.setValue(int(value))
    def _format_label(self, prefix, value):
        return f"{int(value)}%"
    def initUI(self):
        grid_layout = QGridLayout(self)
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        self.cpu_bar, self.cpu_percent_label = self.add_metric_to_grid("CPU Usage:", color_for("CPU"), grid_layout, 0)
        self.ram_bar, self.ram_percent_label = self.add_metric_to_grid("RAM Usage:", color_for("RAM"), grid_layout, 1)
        if self.has_nvidia_gpu:
            self.gpu_bar, self.gpu_percent_label = self.add_metric_to_grid("GPU Usage:", color_for("GPU"), grid_layout, 2)
            self.vram_bar, self.vram_percent_label = self.add_metric_to_grid("VRAM Usage:", color_for("VRAM"), grid_layout, 3)
            self.power_bar, self.power_percent_label = self.add_metric_to_grid("GPU Power:", color_for("GPU_POWER"), grid_layout, 4)
    def add_metric_to_grid(self, label_text, color, grid_layout, row):
        label = QLabel(label_text)
        grid_layout.addWidget(label, row, 0)
        percent_label = QLabel("0%")
        grid_layout.addWidget(percent_label, row, 1)
        progress_bar = self.create_progress_bar(color)
        grid_layout.addWidget(progress_bar, row, 2)
        return progress_bar, percent_label
    def create_progress_bar(self, color):
        bar = QProgressBar()
        bar.setMaximum(100)
        bar.setMaximumHeight(11)
        bar.setStyleSheet(f"QProgressBar {{ background-color: #1e2126; border: none; }}QProgressBar::chunk {{ background-color: {color}; }}")
        bar.setTextVisible(False)
        return bar

@lru_cache(maxsize=8)
def gradient_pixmap(color: str, height: int) -> QPixmap:
    pixmap = QPixmap(1, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    gradient = QLinearGradient(0, 0, 0, height)
    fill_color = QColor(color)
    fill_color.setAlpha(60)
    gradient.setColorAt(0, fill_color)
    gradient.setColorAt(1, QColor(0, 0, 0, 0))
    painter.fillRect(pixmap.rect(), gradient)
    painter.end()
    return pixmap

class Sparkline(QWidget):
    def __init__(self, max_values=125, color="#0074D9"):
        super().__init__()
        self.values = deque(maxlen=max_values)
        self.setFixedSize(125, 65)
        self.color = QColor(color)
    def add_value(self, value):
        self.values.append(value)
        self.update()
    def paintEvent(self, event):
        if not self.values:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        margin = 5
        min_value = 0
        max_value = 100
        value_range = max_value - min_value
        path = QPainterPath()
        x_step = (width - 2 * margin) / (len(self.values) - 1) if len(self.values) > 1 else 0
        points = []
        for i, value in enumerate(self.values):
            x = margin + i * x_step
            y = height - margin - (value / value_range) * (height - 2 * margin)
            points.append(QPointF(x, y))
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        fill_path = QPainterPath(path)
        fill_path.lineTo(points[-1].x(), height - margin)
        fill_path.lineTo(points[0].x(), height - margin)
        fill_path.closeSubpath()
        painter.save()
        painter.setClipPath(fill_path)
        grad_pm = gradient_pixmap(self.color.name(), height)
        for x in range(0, width, grad_pm.width()):
            painter.drawPixmap(x, 0, grad_pm)
        painter.restore()
        painter.setPen(QPen(self.color, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

class SparklineVisualization(BaseVisualization):
    def __init__(self, metrics_store: MetricsStore):
        super().__init__(metrics_store)
        self.initUI()
        self._metric_mappings = [
            ("cpu_usage", "cpu_spark", "cpu_lbl", "CPU"),
            ("ram_usage_percent", "ram_spark", "ram_lbl", "RAM"),
        ]
        if self.has_nvidia_gpu:
            self._gpu_metric_mappings = [
                ("gpu_utilization", "gpu_spark", "gpu_lbl", "GPU"),
                ("vram_usage_percent", "vram_spark", "vram_lbl", "VRAM"),
                ("power_usage_percent", "power_spark", "power_lbl", "GPU Power"),
            ]
    def _update_widget(self, widget, value):
        widget.add_value(value)
    def initUI(self):
        main_layout = QGridLayout(self)
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(1, 1, 1, 1)
        def create_group(name, color_key):
            w = QWidget()
            l = QVBoxLayout(w)
            l.setSpacing(1)
            l.setContentsMargins(0, 0, 0, 0)
            s = Sparkline(color=color_for(color_key))
            l.addWidget(s, alignment=Qt.AlignCenter)
            lbl = QLabel(f"{name} 0.0%")
            lbl.setAlignment(Qt.AlignCenter)
            l.addWidget(lbl, alignment=Qt.AlignCenter)
            return w, s, lbl
        cpu_group, self.cpu_spark, self.cpu_lbl = create_group("CPU", "CPU")
        main_layout.addWidget(cpu_group, 0, 0)
        ram_group, self.ram_spark, self.ram_lbl = create_group("RAM", "RAM")
        main_layout.addWidget(ram_group, 0, 1)
        if self.has_nvidia_gpu:
            gpu_group, self.gpu_spark, self.gpu_lbl = create_group("GPU", "GPU")
            main_layout.addWidget(gpu_group, 0, 2)
            vram_group, self.vram_spark, self.vram_lbl = create_group("VRAM", "VRAM")
            main_layout.addWidget(vram_group, 0, 3)
            power_group, self.power_spark, self.power_lbl = create_group("GPU Power", "GPU_POWER")
            main_layout.addWidget(power_group, 0, 4)
        for i in range(main_layout.columnCount()):
            main_layout.setColumnStretch(i, 1)

class Speedometer(QWidget):
    def __init__(self, min_value=0, max_value=100, colors=None):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        self.colors = colors or ["#00FF00", "#FFFF00", "#FF0000"]
        self.setFixedSize(105, 105)
    def set_value(self, value):
        self.current_value = max(self.min_value, min(self.max_value, value))
        self.update()
    def get_color_at_angle(self, angle):
        t = angle / 180
        if t <= 0:
            return QColor(self.colors[0])
        if t >= 1:
            return QColor(self.colors[-1])
        segment = t * (len(self.colors) - 1)
        idx = int(segment)
        t = segment - idx
        idx = min(idx, len(self.colors) - 2)
        c1 = QColor(self.colors[idx])
        c2 = QColor(self.colors[idx + 1])
        r = int(c1.red() * (1 - t) + c2.red() * t)
        g = int(c1.green() * (1 - t) + c2.green() * t)
        b = int(c1.blue() * (1 - t) + c2.blue() * t)
        return QColor(r, g, b)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        cx = w / 2
        cy = h / 2
        r = min(w, h) / 2 * 0.7
        start_angle = 180 * 16
        for i in range(180):
            painter.setPen(self.get_color_at_angle(i))
            painter.drawArc(cx - r, cy - r, r * 2, r * 2, start_angle - i * 16, -16)
        angle = 180 - (self.current_value - self.min_value) / (self.max_value - self.min_value) * 180
        n_len = r * 0.9
        n_w = 5
        rad = angle * (pi / 180)
        tip_x = cx + n_len * cos(rad)
        tip_y = cy - n_len * sin(rad)
        perp = rad + pi / 2
        hw = n_w / 2
        p1 = QPointF(cx + hw * cos(perp), cy - hw * sin(perp))
        p2 = QPointF(cx - hw * cos(perp), cy + hw * sin(perp))
        needle = QPolygon([p1.toPoint(), p2.toPoint(), QPointF(tip_x, tip_y).toPoint()])
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.white)
        painter.drawPolygon(needle)

class SpeedometerVisualization(BaseVisualization):
    def __init__(self, metrics_store: MetricsStore):
        super().__init__(metrics_store)
        self.initUI()
        self._metric_mappings = [
            ("cpu_usage", "cpu_sm", "cpu_lbl", "CPU"),
            ("ram_usage_percent", "ram_sm", "ram_lbl", "RAM"),
        ]
        if self.has_nvidia_gpu:
            self._gpu_metric_mappings = [
                ("gpu_utilization", "gpu_sm", "gpu_lbl", "GPU"),
                ("vram_usage_percent", "vram_sm", "vram_lbl", "VRAM"),
                ("power_usage_percent", "power_sm", "power_lbl", "GPU Power"),
            ]
    def _update_widget(self, widget, value):
        widget.set_value(value)
    def initUI(self):
        main_layout = QGridLayout(self)
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(1, 1, 1, 1)
        def create_group(name, color_key=None):
            l = QVBoxLayout()
            l.setSpacing(2)
            sm = Speedometer(colors=["#00FF00", "#FFFF00", "#FF0000"])
            sm.setFixedSize(105, 105)
            l.addWidget(sm, alignment=Qt.AlignCenter)
            lbl = QLabel(f"{name} 0.0%")
            lbl.setAlignment(Qt.AlignCenter)
            l.addWidget(lbl, alignment=Qt.AlignCenter)
            return l, sm, lbl
        cpu_group, self.cpu_sm, self.cpu_lbl = create_group("CPU")
        main_layout.addLayout(cpu_group, 0, 0)
        ram_group, self.ram_sm, self.ram_lbl = create_group("RAM")
        main_layout.addLayout(ram_group, 0, 1)
        if self.has_nvidia_gpu:
            gpu_group, self.gpu_sm, self.gpu_lbl = create_group("GPU")
            main_layout.addLayout(gpu_group, 0, 2)
            vram_group, self.vram_sm, self.vram_lbl = create_group("VRAM")
            main_layout.addLayout(vram_group, 0, 3)
            power_group, self.power_sm, self.power_lbl = create_group("GPU Power")
            main_layout.addLayout(power_group, 0, 4)
        for i in range(main_layout.columnCount()):
            main_layout.setColumnStretch(i, 1)

@lru_cache(maxsize=8)
def arc_background(w: int, h: int) -> QPixmap:
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    r = min(w, h) / 2 - 10
    c = QPointF(w / 2, h / 2)
    painter.setPen(QPen(QColor("#1e2126"), 8))
    painter.drawArc(int(c.x() - r), int(c.y() - r), int(r * 2), int(r * 2), 180 * 16, -180 * 16)
    painter.end()
    return pm

class ArcGraph(QWidget):
    def __init__(self, color="#0074D9"):
        super().__init__()
        self.color = QColor(color)
        self.value = 0
        self.setFixedSize(100, 100)
    def set_value(self, value):
        self.value = min(100, max(0, value))
        self.update()
    def paintEvent(self, event):
        bg = arc_background(self.width(), self.height())
        painter = QPainter(self)
        painter.drawPixmap(0, 0, bg)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        r = min(w, h) / 2 - 10
        c = QPointF(w / 2, h / 2)
        painter.setPen(QPen(self.color, 8))
        span = -(self.value / 100.0) * 180
        painter.drawArc(int(c.x() - r), int(c.y() - r), int(r * 2), int(r * 2), 180 * 16, span * 16)
        painter.setPen(Qt.white)
        f = painter.font()
        f.setPointSize(14)
        painter.setFont(f)
        painter.drawText(self.rect(), Qt.AlignCenter, f"{int(self.value)}%")

class ArcGraphVisualization(BaseVisualization):
    def __init__(self, metrics_store: MetricsStore):
        super().__init__(metrics_store)
        self.initUI()
        self._metric_mappings = [
            ("cpu_usage", "cpu_arc", "cpu_lbl", "CPU"),
            ("ram_usage_percent", "ram_arc", "ram_lbl", "RAM"),
        ]
        if self.has_nvidia_gpu:
            self._gpu_metric_mappings = [
                ("gpu_utilization", "gpu_arc", "gpu_lbl", "GPU"),
                ("vram_usage_percent", "vram_arc", "vram_lbl", "VRAM"),
                ("power_usage_percent", "power_arc", "power_lbl", "GPU Power"),
            ]
    def _update_widget(self, widget, value):
        widget.set_value(value)
    def initUI(self):
        main_layout = QGridLayout(self)
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(1, 1, 1, 1)
        def create_group(name, color_key):
            l = QVBoxLayout()
            l.setSpacing(2)
            arc = ArcGraph(color=color_for(color_key))
            l.addWidget(arc, alignment=Qt.AlignCenter)
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            l.addWidget(lbl, alignment=Qt.AlignCenter)
            return l, arc, lbl
        cpu_group, self.cpu_arc, self.cpu_lbl = create_group("CPU", "CPU")
        main_layout.addLayout(cpu_group, 0, 0)
        ram_group, self.ram_arc, self.ram_lbl = create_group("RAM", "RAM")
        main_layout.addLayout(ram_group, 0, 1)
        if self.has_nvidia_gpu:
            gpu_group, self.gpu_arc, self.gpu_lbl = create_group("GPU", "GPU")
            main_layout.addLayout(gpu_group, 0, 2)
            vram_group, self.vram_arc, self.vram_lbl = create_group("VRAM", "VRAM")
            main_layout.addLayout(vram_group, 0, 3)
            power_group, self.power_arc, self.power_lbl = create_group("GPU Power", "GPU_POWER")
            main_layout.addLayout(power_group, 0, 4)
        for i in range(main_layout.columnCount()):
            main_layout.setColumnStretch(i, 1)

class VizType(IntEnum):
    BAR = 0
    SPARKLINE = 1
    SPEEDO = 2
    ARC = 3

VIZ_FACTORY = {
    VizType.BAR: BarVisualization,
    VizType.SPARKLINE: SparklineVisualization,
    VizType.SPEEDO: SpeedometerVisualization,
    VizType.ARC: ArcGraphVisualization
}

class MetricsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_store = MetricsStore(buffer_size=100)
        self.init_ui()
        self.current_visualization_type = VizType.SPARKLINE
        self.setToolTip("Right click for display options")
        self.collector_thread = MetricsCollectorThread()
        self.collector_thread.metrics_updated.connect(self.metrics_store.add_metrics)
        self.start_metrics_collector()
    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.current_visualization = VIZ_FACTORY[VizType.SPARKLINE](self.metrics_store)
        self.layout.addWidget(self.current_visualization)
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        visual_menu = menu.addMenu("Visualization")
        bar_action = visual_menu.addAction("Bar")
        spark_action = visual_menu.addAction("Sparkline")
        speed_action = visual_menu.addAction("Speedometer")
        arc_action = visual_menu.addAction("Arc")
        actions_map = {bar_action: VizType.BAR, spark_action: VizType.SPARKLINE, speed_action: VizType.SPEEDO, arc_action: VizType.ARC}
        actions_map_inv = {v: k for k, v in actions_map.items()}
        actions_map_inv[self.current_visualization_type].setCheckable(True)
        actions_map_inv[self.current_visualization_type].setChecked(True)
        menu.addSeparator()
        running = self.collector_thread and self.collector_thread.isRunning()
        control_action = menu.addAction("Stop Monitoring" if running else "Start Monitoring")
        action = menu.exec_(event.globalPos())
        if action in actions_map:
            self.change_visualization(actions_map[action])
        elif action == control_action:
            if running:
                self.stop_metrics_collector()
            else:
                self.start_metrics_collector()

    def change_visualization(self, kind: VizType):
        if kind == self.current_visualization_type:
            return
        self.current_visualization_type = kind
        self.current_visualization.cleanup()
        self.layout.removeWidget(self.current_visualization)
        self.current_visualization.deleteLater()
        self.current_visualization = VIZ_FACTORY[kind](self.metrics_store)
        self.current_visualization.setToolTip("Right click for display options")
        self.layout.addWidget(self.current_visualization)

    def start_metrics_collector(self):
        if not self.collector_thread.isRunning():
            self.collector_thread.start()

    def stop_metrics_collector(self):
        if self.collector_thread.isRunning():
            self.collector_thread.stop()

    def cleanup(self):
        if self.collector_thread.isRunning():
            self.collector_thread.stop()
        self.current_visualization.cleanup()

        if HAS_NVIDIA_GPU:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)
