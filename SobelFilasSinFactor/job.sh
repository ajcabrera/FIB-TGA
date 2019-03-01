#!/bin/bash
export PATH=/Soft/cuda/8.0.61/bin:$PATH
### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N proj1 
# Cambiar el shell
#$ -S /bin/bash

#nvprof --unified-memory-profiling off 

./proj1.exe subaru.bmp sobel 0.25
