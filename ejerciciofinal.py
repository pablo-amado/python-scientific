"""
==================
Final Assignment
==================

Todos los archivos están subidos a campus.
El largo de los registros es entre 10 y 11 minutos
Fs = 512

FECHA DE ENTREGA: 24/12/2024
SEGUNDA FECHA DE ENTREGA: 10/01/2025


|---- BASELINE --------|
|---- PESTANEO ------|
|---- RISA ------- |
|---- TRUCO_DOS ----|
|---- TRUCO_SIETE --------|
|---- TRUCO_SECUENCIA ------|
|---- DEATHMETAL --------|
|---- BETHOVEN --------|

* Baseline: esta parte la pueden utilizar para tener ejemplos negativos de cualquier cosa que deseen detectar.  Por 
ejemplo si quieren detectar que algo cambia cuando hay "imaginación en colores violeta", extraen features de ese momento y de
este e intentan armar un clasificador.
* Pestaneos: los pestaneos son eventos temporales que pueden ser detectados directamente en la señal.
* Risa: pueden tratar de detectar eventos temporales (picos) en la señal que se correspondan con la risa.
* Truco_dos: pueden tratar de detectar cambios que representen el movimiento de la cara de los besitos.
* Truco_siete: pueden tratar de detectar cambios que representen el movimiento de la cara de las muecas.
* Truco_secuencia: entrenando con los otros dos tienen que tratar de detectar aca la secuencia que genero Agustina.
* Deathmetal: pueden tratar de detectar cambios ritmicos (espectrales) relacionados con la musica.
* Bethoven: pueden tratar de detectar cambios ritmicos (espectrales) relacionados con la musica.

Objetivo:
El objetivo es dado este registro implementar un análisis de estos datos, exploratorio, superviado 
o no supervisado, para intentar identificar que es lo que el sujeto está haciendo en cada bloque.  Pueden 
intentar separar dos bloques entre sí, un bloque particular frente al BASELINE (esto es el momento cuando el sujeto
no hace nada particular).  Pueden usar una parte de dos bloques para entrenar y luego intentar predecir las otras partes.
Tienen que producir un PDF informe con gráficos/tablas breve y resumido (no más de 4 páginas)

"""

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
from scipy.fftpack import fft
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy.signal import butter, filtfilt, buttord
from scipy.signal import butter, lfilter

# El protocolo experimental que implementamos tiene 2 datasets:
# 1- Dataset de las señales de EEG
# 2- El video de las imágenes (de la grabación de la clase)
#
#
# La idea es tomar estos datasets y derivar de forma automática las diferentes secciones.  Esto se puede hacer en base self-supervised, es
# decir tomar los datos de algún dataset, derivar los labels para cada secciones y luego intentar implementar un clasificador multiclase.
#
# Tienen que entregar un PDF, tipo Markdown con código, gráficos y cualquier insight obtenido del dataset.

#Importar Baseline, Bethoven y Deathmetal

signals_beth = pd.read_csv('data/datafinal/bethoven.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

print('Estructura de la informacion:')
print(signals_beth.head())

data_beth = signals_beth.values
eeg_beth = data_beth[:,2]

print(len(eeg_beth))

signals_death = pd.read_csv('data/datafinal/deathmetal.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

print('Estructura de la informacion:')
print(signals_death.head())

data_death = signals_death.values
eeg_death = data_death[:,2]

print(eeg_death)

signals_baseline = pd.read_csv('data/datafinal/baseline.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

print('Estructura de la informacion:')
print(signals_baseline.head())

data_baseline = signals_baseline.values
eeg_baseline = data_baseline[:,2]

print(eeg_baseline)

#Graficar las señales originales

plt.plot(eeg_baseline,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal Baseline')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg_death)])
plt.show()

plt.plot(eeg_beth,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal Bethoven')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg_death)])
plt.show()

plt.plot(eeg_death,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal Deathmetal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg_death)])
plt.show()

#-----FILTROS TEMPORALES-----

# La operación de convolución permite implementar el suavizado del Moving Average
windowlength = 10
avgeeg_beth = np.convolve(eeg_beth, np.ones((windowlength,))/windowlength, mode='same')
avgeeg_death = np.convolve(eeg_death, np.ones((windowlength,))/windowlength, mode='same')
avgeeg_baseline = np.convolve(eeg_baseline, np.ones((windowlength,))/windowlength, mode='same')

# Crear la grilla de gráficos (3 filas, 1 columna)
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Primer gráfico: Señal EEG original
axes[0].plot(avgeeg_baseline, 'r', label='Smoothed EEG')
axes[0].set_xlabel('t')
axes[0].set_ylabel('eeg(t)')
axes[0].set_title(r'Smoothed EEG Baseline')
axes[0].set_ylim([-2000, 2000])
axes[0].set_xlim([0, len(avgeeg_baseline)])
axes[0].legend()

# Segundo gráfico: Señal suavizada bethoven
axes[1].plot(avgeeg_beth, 'b', label='Smoothed EEG')
axes[1].set_xlabel('t')
axes[1].set_ylabel('eeg(t)')
axes[1].set_title(r'Smoothed EEG Bethoven')
axes[1].set_ylim([-2000, 2000])
axes[1].set_xlim([0, len(avgeeg_beth)])
axes[1].legend()

# Tercer gráfico: Señal suavizada deathmetal
axes[2].plot(avgeeg_death, 'b', label='Smoothed EEG')
axes[2].set_xlabel('t')
axes[2].set_ylabel('eeg(t)')
axes[2].set_title(r'Smoothed EEG Deathmetal')
axes[2].set_ylim([-2000, 2000])
axes[2].set_xlim([0, len(avgeeg_death)])
axes[2].legend()

# Ajustar los espacios entre gráficos
plt.tight_layout()
plt.show()


#-----FILTRO ESPECTRAL-----

Fs = 512  # Frecuencia de muestreo
N = len(eeg_baseline)  # Número de muestras en la señal
T = 1.0 / Fs  # Intervalo de tiempo entre muestras

#Baseline

# FFT de la señal original
yf = fft(eeg_baseline)
xf = np.linspace(0.0, Fs / 2, N // 2)  # Frecuencias positivas

plt.figure(figsize=(10, 6))
plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
plt.title("Espectro de la señal original (Baseline)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

# Filtro pasa-banda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Aplicar filtro a la señal
lowcut = 1.0  # Frecuencia de corte baja (Hz)
highcut = 50.0  # Frecuencia de corte alta (Hz)

filtered_signal = butter_bandpass_filter(eeg_baseline, lowcut, highcut, Fs, order=5)

# FFT de la señal filtrada
yf_filtered = fft(filtered_signal)

plt.figure(figsize=(10, 6))
plt.plot(xf, 2.0 / N * np.abs(yf_filtered[:N // 2]))
plt.title("Espectro de la señal filtrada (Baseline)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

# Visualizar la señal filtrada en el tiempo
plt.figure(figsize=(10, 6))
plt.plot(filtered_signal, 'b', label='Señal Filtrada')
plt.title("Señal filtrada (Baseline)")
plt.xlabel("Tiempo (muestras)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()


#Bethoven

# FFT de la señal original
yf1 = fft(eeg_beth)
xf = np.linspace(0.0, Fs / 2, N // 2)  # Frecuencias positivas

plt.figure(figsize=(10, 6))
plt.plot(xf, 2.0 / N * np.abs(yf1[:N // 2]))
plt.title("Espectro de la señal original (Bethoven)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

# Filtro pasa-banda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Aplicar filtro a la señal
lowcut = 1.0  # Frecuencia de corte baja (Hz)
highcut = 50.0  # Frecuencia de corte alta (Hz)

filtered_signal1 = butter_bandpass_filter(eeg_beth, lowcut, highcut, Fs, order=5)

# FFT de la señal filtrada
yf_filtered1 = fft(filtered_signal1)

plt.figure(figsize=(10, 6))
plt.plot(xf, 2.0 / N * np.abs(yf_filtered1[:N // 2]))
plt.title("Espectro de la señal filtrada (Bethoven)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

# Visualizar la señal filtrada en el tiempo
plt.figure(figsize=(10, 6))
plt.plot(filtered_signal1, 'b', label='Señal Filtrada')
plt.title("Señal filtrada (Bethoven)")
plt.xlabel("Tiempo (muestras)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()


#Deathmetal

# FFT de la señal original
yf2 = fft(eeg_death)
xf = np.linspace(0.0, Fs / 2, N // 2)  # Frecuencias positivas

plt.figure(figsize=(10, 6))
plt.plot(xf, 2.0 / N * np.abs(yf2[:N // 2]))
plt.title("Espectro de la señal original (Deathmetal)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

# Filtro pasa-banda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Aplicar filtro a la señal
lowcut = 1.0  # Frecuencia de corte baja (Hz)
highcut = 50.0  # Frecuencia de corte alta (Hz)

filtered_signal2 = butter_bandpass_filter(eeg_death, lowcut, highcut, Fs, order=5)

# FFT de la señal filtrada
yf_filtered2 = fft(filtered_signal2)

plt.figure(figsize=(10, 6))
plt.plot(xf, 2.0 / N * np.abs(yf_filtered2[:N // 2]))
plt.title("Espectro de la señal filtrada (Deathmetal)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

# Visualizar la señal filtrada en el tiempo
plt.figure(figsize=(10, 6))
plt.plot(filtered_signal2, 'b', label='Señal Filtrada')
plt.title("Señal filtrada (Deathmetal)")
plt.xlabel("Tiempo (muestras)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()


#Los tres graficos para comparar

# Crear la grilla de gráficos (3 filas, 1 columna)
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Primer gráfico: Espectro de la señal filtrada (Baseline)
axes[0].plot(xf, 2.0 / N * np.abs(yf_filtered[:N // 2]), 'r', label='Baseline')
axes[0].set_xlabel('Frecuencia (Hz)')
axes[0].set_ylabel('Amplitud')
axes[0].set_title('Espectro de la señal filtrada (Baseline)')
axes[0].grid()
axes[0].legend()

# Segundo gráfico: Espectro de la señal filtrada (Bethoven)
axes[1].plot(xf, 2.0 / N * np.abs(yf_filtered1[:N // 2]), 'b', label='Bethoven')
axes[1].set_xlabel('Frecuencia (Hz)')
axes[1].set_ylabel('Amplitud')
axes[1].set_title('Espectro de la señal filtrada (Bethoven)')
axes[1].grid()
axes[1].legend()

# Tercer gráfico: Espectro de la señal filtrada (Deathmetal)
axes[2].plot(xf, 2.0 / N * np.abs(yf_filtered2[:N // 2]), 'g', label='Deathmetal')
axes[2].set_xlabel('Frecuencia (Hz)')
axes[2].set_ylabel('Amplitud')
axes[2].set_title('Espectro de la señal filtrada (Deathmetal)')
axes[2].grid()
axes[2].legend()

# Ajustar el espacio entre gráficos
plt.tight_layout()
plt.show()