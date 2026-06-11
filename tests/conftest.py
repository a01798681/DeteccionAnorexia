# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: tests/conftest.py
# Descripción general: Archivo de configuración para pytest. Asegura que el
# directorio raíz del proyecto se encuentre en el PYTHONPATH para poder importar
# correctamente los módulos de la carpeta 'src' durante las pruebas.

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))