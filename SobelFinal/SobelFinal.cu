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
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < M && col < NS)
    A[row*N+col*3] = ((A[row*N+col*3] + A[row*N+col*3+1] + A[row*N+col*3+2])/3);
}

__global__ void KernelSobel1(int N,int M, unsigned char *A, int *S, int NS) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < NS) {
      double magnitudX, magnitudY;

        if(col != 0 && row != 0 && col != NS-1 && row != M-1) {
            magnitudX = (double)(A[(row-1)*N+(col-1)*3]*(-1) +
                        A[(row)*N+(col-1)*3]*(-2) + A[(row+1)*N+(col+1)*3]*(-1) +
                        A[(row-1)*N+(col+1)*3] + A[row*N+(col+1)*3]*2 +
                        A[(row+1)*N+(col+1)*3]);

            magnitudY = (double)(A[(row-1)*N+(col-1)*3]*(-1) +
                        A[(row+1)*N+(col-1)*3] + A[(row-1)*N+col*3]*(-2) +
                        A[(row+1)*N+col*3]*2 + A[(row-1)*N+(col+1)*3]*(-1) +
                        A[(row+1)*N+(col+1)*3]);

            S[row*NS+col] = (int)sqrt(magnitudX*magnitudX + magnitudY*magnitudY);
        } else S[row*NS+col] = 0;
    }
}


__global__ void KernelReduction1(int NT,int *S, int *oMin, int *oMax) { //Reduction
    __shared__ int sdataMax[SIZE*SIZE];
    __shared__ int sdataMin[SIZE*SIZE];

    unsigned int s;

    int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;
    sdataMax[tid] = 0;
    sdataMin[tid] = 0x7FFFFFFF;

    while(i< NT) {
        if (S[i] > -1) {
            if (sdataMax[tid] < S[i]) sdataMax[tid] = S[i];
            if (sdataMin[tid] > S[i]) sdataMin[tid] = S[i];
        }
        if (i+blockDim.x < NT && S[i+blockDim.x] > -1) {
            if (sdataMax[tid] < S[i+blockDim.x]) sdataMax[tid] = S[i+blockDim.x];
            if (sdataMin[tid] > S[i+blockDim.x]) sdataMin[tid] = S[i+blockDim.x];
        }
        i += gridSize;
    }
    __syncthreads();

    for (s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) {
            if (sdataMax[tid] < sdataMax[tid+s]) sdataMax[tid] = sdataMax[tid+s];
            if (sdataMin[tid] > sdataMin[tid+s]) sdataMin[tid] = sdataMin[tid+s];
        }
        __syncthreads();
    }
    // desenrrollamos el ultimo warp activo
    if (tid < 32) {
        volatile int *smemMax = sdataMax;
        volatile int *smemMin = sdataMin;

        if (smemMax[tid] < smemMax[tid+32]) smemMax[tid] = smemMax[tid+32];
        if (smemMax[tid] < smemMax[tid+16]) smemMax[tid] = smemMax[tid+16];
        if (smemMax[tid] < smemMax[tid+8]) smemMax[tid] = smemMax[tid+8];
        if (smemMax[tid] < smemMax[tid+4]) smemMax[tid] = smemMax[tid+4];
        if (smemMax[tid] < smemMax[tid+2]) smemMax[tid] = smemMax[tid+2];
        if (smemMax[tid] < smemMax[tid+1]) smemMax[tid] = smemMax[tid+1];

        if (smemMin[tid] < smemMin[tid+32]) smemMin[tid] = smemMin[tid+32];
        if (smemMin[tid] < smemMin[tid+16]) smemMin[tid] = smemMin[tid+16];
        if (smemMin[tid] < smemMin[tid+8]) smemMin[tid] = smemMin[tid+8];
        if (smemMin[tid] < smemMin[tid+4]) smemMin[tid] = smemMin[tid+4];
        if (smemMin[tid] < smemMin[tid+2]) smemMin[tid] = smemMin[tid+2];
        if (smemMin[tid] < smemMin[tid+1]) smemMin[tid] = smemMin[tid+1];
    }


    // El thread 0 escribe el resultado de este bloque en la memoria global
    if (tid == 0) {
        oMax[blockIdx.x] = sdataMax[0];
        oMin[blockIdx.x] = sdataMin[0];
    }

}


__global__ void KernelReduction2(int NT, int *Min, int *Max, float *factor) { //Last step of reduction
    __shared__ int sdataMax[SIZE*SIZE];
    __shared__ int sdataMin[SIZE*SIZE];

    unsigned int s;

    int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;
    sdataMax[tid] = 0;
    sdataMin[tid] = 0x7FFFFFFF;

    while(i< NT) {
        if (sdataMax[tid] < Max[i]) sdataMax[tid] = Max[i];
        if (sdataMin[tid] > Min[i]) sdataMin[tid] = Min[i];
        if(i+blockDim.x < NT){
            if (sdataMax[tid] < Max[i+blockDim.x]) sdataMax[tid] = Max[i+blockDim.x];
            if (sdataMin[tid] > Min[i+blockDim.x]) sdataMin[tid] = Min[i+blockDim.x];
        }
        i += gridSize;
    }
    __syncthreads();

    for (s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) {
            if (sdataMax[tid] < sdataMax[tid+s]) sdataMax[tid] = sdataMax[tid+s];
            if (sdataMin[tid] > sdataMin[tid+s]) sdataMin[tid] = sdataMin[tid+s];
        }
        __syncthreads();
    }
    // desenrrollamos el ultimo warp activo
    if (tid < 32) {
        volatile int *smemMax = sdataMax;
        volatile int *smemMin = sdataMin;

        if (smemMax[tid] < smemMax[tid+32]) smemMax[tid] = smemMax[tid+32];
        if (smemMax[tid] < smemMax[tid+16]) smemMax[tid] = smemMax[tid+16];
        if (smemMax[tid] < smemMax[tid+8]) smemMax[tid] = smemMax[tid+8];
        if (smemMax[tid] < smemMax[tid+4]) smemMax[tid] = smemMax[tid+4];
        if (smemMax[tid] < smemMax[tid+2]) smemMax[tid] = smemMax[tid+2];
        if (smemMax[tid] < smemMax[tid+1]) smemMax[tid] = smemMax[tid+1];

        if (smemMin[tid] < smemMin[tid+32]) smemMin[tid] = smemMin[tid+32];
        if (smemMin[tid] < smemMin[tid+16]) smemMin[tid] = smemMin[tid+16];
        if (smemMin[tid] < smemMin[tid+8]) smemMin[tid] = smemMin[tid+8];
        if (smemMin[tid] < smemMin[tid+4]) smemMin[tid] = smemMin[tid+4];
        if (smemMin[tid] < smemMin[tid+2]) smemMin[tid] = smemMin[tid+2];
        if (smemMin[tid] < smemMin[tid+1]) smemMin[tid] = smemMin[tid+1];
    }


    // El thread 0 escribe el resultado de este bloque en la memoria global
    if (tid == 0) {
        *factor = (float)(255.0/(float)(sdataMax[0]-sdataMin[0]));
        
    }
}


__global__ void KernelSobel2 (int N,int M,unsigned char *A, int *S, int NS, float *factor) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < M && col < NS) {
        if(col != 0 && row != 0 && col != NS-1 && row != M-1)
            A[row*N+col*3] = A[row*N+col*3+1] = A[row*N+col*3+2] = (unsigned char)(S[row*NS+col] * factor[0]);
        else A[row*N+col*3] = A[row*N+col*3+1] = A[row*N+col*3+2] = 0;
    }
}


int main(int argc, char** argv)
{
  unsigned int N, M, NS;
  unsigned int numBytesA, numBytesS, numBytesR;
  unsigned int nBlocksX, nBlocksY, nBlocksR, nThreads;

  float TiempoTotal, TiempoKernel, *d_factor, *h_factor, factor;
  cudaEvent_t E0, E1, E2, E3;

  unsigned char *d_A;
  int *d_S, *d_OutMax, *d_OutMin;

  if (argc != 3 && argc != 4) { printf("Usage: ./exe img.bmp prefix\n"); exit(0); }
  if (argc == 4){
	factor = atof(argv[3]);
	h_factor = &factor;
  }
  printf("INICIO\n");

  bmpInfoHeader header;
  unsigned char *image;
  image = LoadBMP(argv[1], &header);

  unsigned int N3 = header.width * 3;
  N = (N3+3) & 0xFFFFFFFC; //fila multiplo de 4 (BMP)
  M = header.height; 
  NS = header.width;

  nThreads = SIZE;

  // numero de Blocks en cada dimension
  nBlocksX = (NS+nThreads-1)/nThreads;
  nBlocksY = (M+nThreads-1)/nThreads;

  numBytesA = N * M * sizeof(unsigned char);
  numBytesS = NS * M * sizeof(int);

  nBlocksR = ((NS * M)+(nThreads*nThreads-1)) / (nThreads*nThreads);
  numBytesR = nBlocksR *sizeof(int);

  dim3 dimGrid(nBlocksX, nBlocksY, 1);
  dim3 dimBlock(nThreads, nThreads, 1);
  dim3 dimGridR(nBlocksR, 1, 1);
  dim3 dimBlockR(nThreads * nThreads, 1, 1);
  
  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);

  cudaEventRecord(E0, 0);
  cudaEventSynchronize(E0);

  // Obtener Memoria en el device
  cudaMalloc((unsigned char**)&d_A, numBytesA);
  cudaMalloc((int**)&d_S, numBytesS);
  cudaMalloc((int**)&d_OutMax, numBytesR);
  cudaMalloc((int**)&d_OutMin, numBytesR);
  cudaMalloc((float**)&d_factor, sizeof(float));

  // Copiar datos del host al device
  cudaMemcpy(d_A, image, numBytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_factor, h_factor, sizeof(float), cudaMemcpyHostToDevice);
  
  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);

  // Ejecutar el kernel
  KernelByN<<<dimGrid, dimBlock>>>(N, M, d_A, d_S, NS);
  KernelSobel1<<<dimGrid, dimBlock>>>(N, M, d_A, d_S, NS);
  
  KernelReduction1<<<dimGridR,dimBlockR>>>(NS*M, d_S, d_OutMin, d_OutMax);
  if(argc==3) KernelReduction2<<<1,dimBlockR>>>(nBlocksR, d_OutMin, d_OutMax, d_factor);
  
  KernelSobel2<<<dimGrid, dimBlock>>>(N, M, d_A, d_S, NS, d_factor);


  cudaEventRecord(E2, 0);

  cudaEventSynchronize(E2);

  // Obtener el resultado desde el host
  cudaMemcpy(image, d_A, numBytesA, cudaMemcpyDeviceToHost);

  // Liberar Memoria del device
  cudaFree(d_A);
  cudaFree(d_S);
  cudaFree(d_OutMin);
  cudaFree(d_OutMax);
  cudaFree(d_factor);

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);

  cudaEventElapsedTime(&TiempoTotal,  E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);
  printf("\nKERNEL ByN & Reductions & Sobel\n");
  printf("Dimensiones: %dx%d\n", NS, M);
  printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
  printf("nBlocks: %dx%d (%d)\n", nBlocksX, nBlocksY, nBlocksX*nBlocksY);
  printf("nThreadsR1: %dx%d (%d)\n", nThreads*nThreads, 1, nThreads * nThreads);
  printf("nBlocksR1: %dx%d (%d)\n", nBlocksR, 1, nBlocksR);
  printf("nThreadsR2: %dx%d (%d)\n", nThreads*nThreads, 1, nThreads * nThreads);
  printf("nBlocksR2: %dx%d (%d)\n", 1, 1, 1);
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);

  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

  char nom[32];
  strcpy(nom, argv[2]);
  strcat(nom, "_");
  strcat(nom,argv[1]);

  SaveBMP(nom, &header, image);
}

