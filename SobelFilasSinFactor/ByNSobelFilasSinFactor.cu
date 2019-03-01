#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define SIZE 32

/* Autores:
 *
 * Antonio J. Cabrera
 * Paul Gazel-Anthoine
 */

// STRUCTS

typedef struct bmpFileHeader {
  /* 2 bytes de identificación */
  uint32_t size;        /* Tamaño del archivo */
  uint16_t resv1;       /* Reservado */
  uint16_t resv2;       /* Reservado */
  uint32_t offset;      /* Offset hasta hasta los datos de imagen */
} bmpFileHeader;

typedef struct bmpInfoHeader {
  uint32_t headersize;  /* Tamaño de la cabecera */
  uint32_t width;       /* Ancho */
  uint32_t height;      /* Alto */
  uint16_t planes;      /* Planos de color (Siempre 1) */
  uint16_t bpp;         /* bits por pixel */
  uint32_t compress;    /* compresion */
  uint32_t imgsize;     /* tamaño de los datos de imagen */
  uint32_t bpmx;        /* Resolucion X en bits por metro */
  uint32_t bpmy;        /* Resolucion Y en bits por metro */
  uint32_t colors;      /* colors used en la paleta */
  uint32_t imxtcolors;  /* Colores importantes. 0 si son todos */
} bmpInfoHeader;


// Rutinas BMP

unsigned char *LoadBMP(char *filename, bmpInfoHeader *bInfoHeader) {
  FILE *f;
  bmpFileHeader header;     /* cabecera */
  unsigned char *imgdata;   /* datos de imagen */
  uint16_t type;            /* 2 bytes identificativos */

  f=fopen (filename, "r");
  if (!f) { /* Si no podemos leer, no hay imagen */
    printf("NO se puede abrir el fichero %s\n", filename);
    return NULL;
  }

  /* Leemos los dos primeros bytes y comprobamos el formato */
  fread(&type, sizeof(uint16_t), 1, f);
  if (type !=0x4D42) {
    fclose(f);
    printf("%s NO es una imagen BMP\n", filename);
    return NULL;
  }

  /* Leemos la cabecera del fichero */
  fread(&header, sizeof(bmpFileHeader), 1, f);

  printf("File size: %u\n", header.size);
  printf("Reservado: %u\n", header.resv1);
  printf("Reservado: %u\n", header.resv2);
  printf("Offset:    %u\n", header.offset);

  /* Leemos la cabecera de información del BMP */
  fread(bInfoHeader, sizeof(bmpInfoHeader), 1, f);

  /* Reservamos memoria para la imagen, lo que indique imgsize */
  if (bInfoHeader->imgsize == 0) bInfoHeader->imgsize = ((bInfoHeader->width*3 +3) / 4) * 4 * bInfoHeader->height;
  imgdata = (unsigned char*) malloc(bInfoHeader->imgsize);
  if (imgdata == NULL) {
    printf("Fallo en el malloc, del fichero %s\n", filename);
    exit(0);
  }
  /* Nos situamos en donde empiezan los datos de imagen, lo indica el offset de la cabecera de fichero */
  fseek(f, header.offset, SEEK_SET);

  /* Leemos los datos de la imagen, tantos bytes como imgsize */
  fread(imgdata, bInfoHeader->imgsize,1, f);

  /* Cerramos el fichero */
  fclose(f);

  /* Devolvemos la imagen */
  return imgdata;
}

bmpInfoHeader *createInfoHeader(uint32_t width, uint32_t height, uint32_t ppp) {
  bmpInfoHeader *InfoHeader;
  bool IH;
  IH = malloc(sizeof(bmpInfoHeader));
  if (!IH) return NULL;
  InfoHeader->headersize = sizeof(bmpInfoHeader);
  InfoHeader->width = width;
  InfoHeader->height = height;
  InfoHeader->planes = 1;
  InfoHeader->bpp = 24;
  InfoHeader->compress = 0;
  /* 3 bytes por pixel, width*height pixels, el tamaño de las filas ha de ser multiplo de 4 */
  InfoHeader->imgsize = ((width*3 + 3) / 4) * 4 * height;
  InfoHeader->bpmx = (unsigned) ((double)ppp*100/2.54);
  InfoHeader->bpmy= InfoHeader->bpmx;          /* Misma resolucion vertical y horiontal */
  InfoHeader->colors = 0;
  InfoHeader->imxtcolors = 0;

  return InfoHeader;
}

void SaveBMP(char *filename, bmpInfoHeader *InfoHeader, unsigned char *imgdata) {
  bmpFileHeader header;
  FILE *f;
  uint16_t type;

  f=fopen(filename, "w+");

  header.size = InfoHeader->imgsize + sizeof(bmpFileHeader) + sizeof(bmpInfoHeader) +2;//2
  header.resv1 = 0;
  header.resv2 = 0;
  /* El offset será el tamaño de las dos cabeceras + 2 (información de fichero)*/
  header.offset=sizeof(bmpFileHeader)+sizeof(bmpInfoHeader) +2;//2

  /* Escribimos la identificación del archivo */
  type=0x4D42;
  fwrite(&type, sizeof(type),1,f);

  /* Escribimos la cabecera de fichero */
  fwrite(&header, sizeof(bmpFileHeader),1,f);

  /* Escribimos la información básica de la imagen */
  fwrite(InfoHeader, sizeof(bmpInfoHeader),1,f);

  /* Escribimos la imagen */
  fwrite(imgdata, InfoHeader->imgsize, 1, f);

  fclose(f);
}

void DisplayInfo(char *FileName, bmpInfoHeader *InfoHeader)
{
  printf("\n");
  printf("Informacion de %s\n", FileName);
  printf("Tamaño de la cabecera: %u bytes\n", InfoHeader->headersize);
  printf("Anchura:               %d pixels\n", InfoHeader->width);
  printf("Altura:                %d pixels\n", InfoHeader->height);
  printf("Planos (1):            %d\n", InfoHeader->planes);
  printf("Bits por pixel:        %d\n", InfoHeader->bpp);
  printf("Compresion:            %d\n", InfoHeader->compress);
  printf("Tamaño de la imagen:   %u bytes\n", InfoHeader->imgsize);
  printf("Resolucion horizontal: %u px/m\n", InfoHeader->bpmx);
  printf("Resolucion vertical:   %u px/m\n", InfoHeader->bpmy);
  if (InfoHeader->bpmx == 0)
    InfoHeader->bpmx = (unsigned) ((double)24*100/2.54);
  if (InfoHeader->bpmy == 0)
    InfoHeader->bpmy = (unsigned) ((double)24*100/2.54);

  printf("Colores en paleta:     %d\n", InfoHeader->colors);
  printf("Colores importantes:   %d\n", InfoHeader->imxtcolors);
}


/*
------------------------------------------------
 Nuestro Código
------------------------------------------------
*/
__global__ void KernelByN (int N, int M, unsigned char *A, int *S, int NS) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if(row < M){
    for(int i = 0; i < NS; i++)
      S[row*NS+i] = (A[row*N+i*3] + A[row*N+i*3+1] + A[row*N+i*3+2])/3;
  }
}

__global__ void KernelSobel(int N, int M, unsigned char *A, int *S, int NS, float *factor) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
 
    if(row < M) {
     for(int col = 0; col < NS; col++){
      double magnitudX, magnitudY;

        if(col > 0 && row > 0 && col < NS-1 && row < M-1) {
            magnitudX = (double)(S[(row-1)*NS+col-1]*(-1) +
                        S[(row)*NS+col-1]*(-2) + S[(row+1)*NS+col+1]*(-1) +
                        S[(row-1)*NS+col+1] + S[row*NS+col+1]*2 +
                        S[(row+1)*NS+col+1]);

            magnitudY = (double)(S[(row-1)*NS+col-1]*(-1) +
                        S[(row+1)*NS+col-1] + S[(row-1)*NS+col]*(-2) +
                        S[(row+1)*NS+col]*2 + S[(row-1)*NS+col+1]*(-1) +
                        S[(row+1)*NS+col+1]);

	    float aux = (float)(sqrt(magnitudX*magnitudX + magnitudY*magnitudY) * factor[0]);

            A[row*N+col*3] = A[row*N+col*3+1] = A[row*N+col*3+2] = (unsigned char) aux;
        } else A[row*N+col*3] = A[row*N+col*3+1] = A[row*N+col*3+2] = 0;
     }
    }
}


int main(int argc, char** argv)
{
  unsigned int N, M, NS;
  unsigned int numBytesA, numBytesS;
  unsigned int nBlocksX, nBlocksY, nThreads;

  float TiempoTotal, TiempoKernel, *d_factor, *h_factor, factor;
  cudaEvent_t E0, E1, E2, E3;

  unsigned char *d_A;
  int *d_S;

  if (argc != 4) { printf("Usage: ./exe img.bmp prefix factor\n"); exit(0); }
  else {
	factor = atof(argv[3]);
	h_factor = &factor;
  }
  printf("INICIO\n");

  bmpInfoHeader header;
  unsigned char *image;
  image = LoadBMP(argv[1], &header);

  unsigned int N3 = header.width * 3;
  N = (N3+3) & 0xFFFFFFFC; //linea multiplo de 4 (BMP)
  M = header.height;
  NS = header.width;

  // numero de Threads en cada dimension
  nThreads = SIZE;

  // numero de Blocks en cada dimension
  nBlocksY = (M+nThreads*nThreads-1)/(nThreads*nThreads);

  numBytesA = N * M * sizeof(unsigned char);
  numBytesS = NS * M * sizeof(int);

  dim3 dimGrid(1, nBlocksY, 1);
  dim3 dimBlock(1, nThreads*nThreads, 1);

  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);

  cudaEventRecord(E0, 0);
  cudaEventSynchronize(E0);

  // Obtener Memoria en el device
  cudaMalloc((unsigned char**)&d_A, numBytesA);
  cudaMalloc((int**)&d_S, numBytesS);
  cudaMalloc((float**)&d_factor, sizeof(float));

  // Copiar datos del host al device
  cudaMemcpy(d_A, image, numBytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_factor, h_factor, sizeof(float), cudaMemcpyHostToDevice);
  
  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);

  // Ejecutar el kernel
  KernelByN<<<dimGrid, dimBlock>>>(N, M, d_A, d_S, NS);
  KernelSobel<<<dimGrid, dimBlock>>>(N, M, d_A, d_S, NS, d_factor);

  cudaEventRecord(E2, 0);

  cudaEventSynchronize(E2);

  // Obtener el resultado desde el host
  cudaMemcpy(image, d_A, numBytesA, cudaMemcpyDeviceToHost);

  // Liberar Memoria del device
  cudaFree(d_A);
  cudaFree(d_S);
  cudaFree(d_factor);

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);

  cudaEventElapsedTime(&TiempoTotal,  E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);
  printf("\nKERNEL 00\n");
  printf("Dimensiones: %dx%d\n", NS, M);
  printf("nThreads: %dx%d (%d)\n", 1, nThreads*nThreads, nThreads * nThreads);
  printf("nBlocks: %dx%d (%d)\n", 1, nBlocksY, nBlocksY);
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);
  //printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoTotal));
  //printf("Rendimiento Kernel: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoKernel));

  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

  char nom[32];
  strcpy(nom, argv[2]);
  strcat(nom, "_");
  strcat(nom,argv[1]);

  SaveBMP(nom, &header, image);

}



