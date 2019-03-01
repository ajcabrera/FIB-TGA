# FIB-TGA
Asignatura de tarjetas gráficas y aceleradores (2018)

Proyecto final de la asignatura.
Filtro Sobel - Procesado de imágenes BMP en C y CUDA.
 - Antonio J Cabrera
 - Paul Gazel-Anthoine


Ficheros:

BYNC
Simple proceso de pasar imagen a blanco y negro (C).

BYNElem
Recorrido elemento a elemento del blanco y negro (CUDA).

BYNFila
Recorrido por filas del blanco y negro (CUDA).

SobelC
Aplicación del filtro Sobel (C).

SobelElemSinFactor
Aplicación del filtro Sobel con recorrido elemento a elemento y sin cálculo de factor de contraste de la imagen (CUDA).

SobelFilasSinFactor
Aplicación del filtro Sobel con recorrido por filas y sin cálculo de factor de contraste de la imagen (CUDA).

SobelFinal
Aplicación del filtro Sobel con cálculo de factor de contraste vía reducción en GPU (CUDA).

SobelReducCPU
Aplicación del filtro Sobel con cálculo de factor de contraste vía reducción en CPU (CUDA).



