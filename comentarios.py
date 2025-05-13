# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt  
from scipy.fft import fft, fftfreq 

# Configuración de estilo para las gráficas
plt.style.use('ggplot')  # Usar estilo 'ggplot' para visualización
plt.rcParams['font.size'] = 12  # Tamaño de fuente por defecto

# Carga de datos desde archivos CSV
df_temp = pd.read_csv('temperatura.csv')  # Datos de temperatura
df_hum = pd.read_csv('humedad.csv')  # Datos de humedad
df_viento = pd.read_csv('viento.csv')  # Datos de velocidad del viento

# Configuración del eje de tiempo
n_muestras = len(df_temp)  # Número total de muestras (asumiendo misma longitud)
tiempo = np.arange(0, n_muestras * 5, 5)  # Vector de tiempo (muestra cada 5 segundos)
fs = 0.2  # Frecuencia de muestreo (0.2 Hz = 1 muestra cada 5 segundos)

# --- Funciones de filtrado digital ---

def butter_lowpass(signal, cutoff=0.012, fs=fs, order=2):
    """
    Filtro pasa bajas Butterworth de fase lineal (filtfilt)
    
    Parámetros:
        signal: Señal de entrada
        cutoff: Frecuencia de corte en Hz (default 0.012 Hz)
        fs: Frecuencia de muestreo (default 0.2 Hz)
        order: Orden del filtro (default 2)
    
    Retorna:
        Señal filtrada
    """
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyq  # Normalización de la frecuencia de corte
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Diseño del filtro
    return filtfilt(b, a, signal)  # Aplicación con fase lineal

def butter_bandpass(signal, low=0.008, high=0.03, fs=fs, order=2):
    """
    Filtro pasa banda Butterworth de fase lineal (filtfilt)
    
    Parámetros:
        signal: Señal de entrada
        low: Frecuencia inferior de corte en Hz (default 0.008 Hz)
        high: Frecuencia superior de corte en Hz (default 0.03 Hz)
        fs: Frecuencia de muestreo (default 0.2 Hz)
        order: Orden del filtro (default 2)
    
    Retorna:
        Señal filtrada
    """
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    low_norm = low / nyq  # Normalización de frecuencia inferior
    high_norm = high / nyq  # Normalización de frecuencia superior
    b, a = butter(order, [low_norm, high_norm], btype='band')  # Diseño del filtro
    return filtfilt(b, a, signal)  # Aplicación con fase lineal

# --- Procesamiento de señales ---

# Diccionario con las señales originales
senales = {
    'Temperatura': df_temp.iloc[:, 1],  # Asume que la columna 1 tiene los datos
    'Humedad': df_hum.iloc[:, 1],
    'Viento': df_viento['Velocidad_Viento_mps']  # Usando nombre de columna explícito
}

# Aplicación de filtros a todas las señales
filtradas = {
    'Temperatura LP': butter_lowpass(senales['Temperatura']),
    'Humedad LP': butter_lowpass(senales['Humedad']),
    'Viento LP': butter_lowpass(senales['Viento']),
    'Temperatura BP': butter_bandpass(senales['Temperatura']),
    'Humedad BP': butter_bandpass(senales['Humedad']),
    'Viento BP': butter_bandpass(senales['Viento'])
}

# --- Visualización de resultados ---

# Creación de figura con subplots (3 variables x 2 tipos de filtro)
fig, axs = plt.subplots(3, 2, figsize=(16, 12))

# Función auxiliar para configuración común de subplots
def setup_plot(ax, title, ylabel, xlabel=False):
    """Configura propiedades comunes de los subplots"""
    ax.set_title(title, pad=15)
    ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel('Tiempo (s)')  # Solo mostrar en última fila
    ax.grid(True, alpha=0.3)  # Grid semi-transparente
    ax.legend(loc='upper right')  # Leyenda en esquina superior derecha

# Generación de gráficas para cada señal
for i, (nombre, señal) in enumerate(senales.items()):
    # Gráficas de filtro pasa bajas (columna izquierda)
    axs[i,0].plot(tiempo, señal, label='Original', alpha=0.5, linewidth=1)
    axs[i,0].plot(tiempo, filtradas[f'{nombre} LP'], 
                 label=f'Pasa Bajas (0.012 Hz)', linewidth=2)
    setup_plot(axs[i,0], f'{nombre}: Filtro Pasa Bajas', 
              '°C' if nombre=='Temperatura' else '%' if nombre=='Humedad' else 'm/s',
              xlabel=(i==2))  # Mostrar label solo en última fila
    
    # Gráficas de filtro pasa bandas (columna derecha)
    axs[i,1].plot(tiempo, señal, label='Original', alpha=0.5, linewidth=1)
    axs[i,1].plot(tiempo, filtradas[f'{nombre} BP'], 
                 label='Pasa Bandas (0.008-0.03 Hz)', linewidth=2)
    setup_plot(axs[i,1], f'{nombre}: Filtro Pasa Bandas', 
              '°C' if nombre=='Temperatura' else '%' if nombre=='Humedad' else 'm/s',
              xlabel=(i==2))

plt.tight_layout()  # Ajustar espaciado entre subplots
plt.show()

# --- Análisis espectral (FFT) ---

def plot_fft_comparison(signal, filtered, title):
    """
    Genera gráficas comparativas de FFT entre señal original y filtrada
    
    Parámetros:
        signal: Señal original
        filtered: Señal filtrada
        title: Título para la gráfica
    """
    freqs = fftfreq(n_muestras, 5)[:n_muestras//2]  # Vector de frecuencias (solo positivas)
    fft_orig = np.abs(fft(signal)[:n_muestras//2]) * 2/n_muestras  # FFT normalizada (original)
    fft_filt = np.abs(fft(filtered)[:n_muestras//2]) * 2/n_muestras  # FFT normalizada (filtrada)
    
    plt.figure(figsize=(12, 4))
    plt.plot(freqs, fft_orig, label='Original', alpha=0.7, linewidth=1)
    plt.plot(freqs, fft_filt, label='Filtrada', linewidth=2)
    
    # Líneas verticales para marcar frecuencias de corte
    if 'Pasa Bajas' in title:
        plt.axvline(0.012, color='red', linestyle='--', alpha=0.7, label='Frecuencia de corte')
    else:
        plt.axvline(0.008, color='red', linestyle='--', alpha=0.7)
        plt.axvline(0.03, color='red', linestyle='--', alpha=0.7)
    
    plt.title(f'Análisis FFT: {title}', pad=15)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud Normalizada')
    plt.xlim(0, 0.1)  # Limitar rango de frecuencias mostrado
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Generar análisis FFT para todas las señales y filtros
for nombre in senales.keys():
    plot_fft_comparison(senales[nombre], filtradas[f'{nombre} LP'], 
                       f'{nombre} - Pasa Bajas')
    plot_fft_comparison(senales[nombre], filtradas[f'{nombre} BP'], 
                       f'{nombre} - Pasa Bandas')