from PySide6.QtWidgets import QTabWidget
from gui_tabs_settings import GuiSettingsTab
from gui_tabs_tools import GuiSettingsTab as ToolsSettingsTab
from gui_tabs_databases import DatabasesTab
from gui_tabs_models import VectorModelsTab
from gui_tabs_database_query import DatabaseQueryTab
from gui_tabs_manage_databases import ManageDatabasesTab

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
