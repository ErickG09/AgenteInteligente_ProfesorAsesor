Agente Inteligente Asesor

Asesor local basado en Google Gemini para responder preguntas y explicar temas paso a paso.

Requisitos

Python 3.10 o 3.11

pip instalado

Clave de API de Google Gemini (Google AI Studio)

Instalación
git clone [A este proyecto]
cd [Este proyecto]
pip install -r requirements.txt

Configuración

Crea un archivo .env en la raíz del proyecto con tu clave de Gemini:

GOOGLE_API_KEY=TU_CLAVE_DE_GEMINI

Uso
python main.py

Archivos principales

main.py: punto de entrada de la aplicación.

interfaz.py: orquestación de la interacción con el usuario.

herramientas.py: utilidades y funciones auxiliares.

memoria.py: manejo de memoria simple.

requirements.txt: lista de dependencias.
