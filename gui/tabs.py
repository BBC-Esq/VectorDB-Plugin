from PySide6.QtWidgets import QTabWidget
from gui.tabs_settings.settings import GuiSettingsTab
from gui.tabs_tools.tools import GuiSettingsTab as ToolsSettingsTab
from gui.tabs_databases.create import DatabasesTab
from gui.tabs_models.models import VectorModelsTab
from gui.tabs_databases.query import DatabaseQueryTab
from gui.tabs_databases.manage import ManageDatabasesTab

def create_tabs():
    tab_widget = QTabWidget()
    tab_widget.setTabPosition(QTabWidget.South)
    
    tab_font = tab_widget.font()
    tab_font.setPointSize(13)
    tab_widget.setFont(tab_font)
    
    tabs = [
        (GuiSettingsTab(), 'Settings'),
        (VectorModelsTab(), 'Models'),
        (ToolsSettingsTab(), 'Tools'),
        (DatabasesTab(), 'Create Database'),
        (ManageDatabasesTab(), 'Manage Databases'),
        (DatabaseQueryTab(), 'Query Database')
    ]
    
    for tab, name in tabs:
        tab_widget.addTab(tab, name)
    
    return tab_widget
