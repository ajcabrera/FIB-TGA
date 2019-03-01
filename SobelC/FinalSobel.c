#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <sys/times.h>
#include <sys/resource.h>

/* Autores:
 *
 * Antonio J. Cabrera
 * Paul Gazel-Anthoine
 */

// Para calcular el tiempo de exe
float GetTime(void)        
{
  struct timeval tim;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  tim=ru.ru_utime;
  return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}


/* ########################
        Structs (H)
   ########################
*/


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



/* ########################
        Rutinas BMP (C)
   ########################
*/


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
int valorPixel(unsigned char *M, int NC, int i)
{
  double magnitudX, magnitudY;

  magnitudX = M[(i-3)-NC]*(-1) + M[i-NC]*(-2) + M[(i+3)-NC]*(-1)
            + M[(i-3)+NC] + M[i+NC]*2 + M[(i+3)+NC];

  magnitudY = M[(i-3)-NC]*(-1) + M[(i+3)-NC]
            + M[(i-3)]*(-2) + M[(i+3)]*2
            + M[(i-3)+NC]*(-1) + M[(i+3)+NC];

  return (int)sqrt(magnitudX*magnitudX + magnitudY*magnitudY);
}


int main(int argc, char** argv) {

  if(argc < 3) {
     perror("USAGE: ./prog IMAGE_TO_EDIT.BMP PREFIX_NEW_IMAGE\n");
     return(0);
  }

  bmpInfoHeader header;

  unsigned char *image;
  image = LoadBMP(argv[1], &header);
  float t1,t2;

  int N = (header.width*3+3) & 0xFFFFFFFC;
  int M = header.height;
  int *sobel, i, min, max, aux;
  bool primera = true;

  t1 = GetTime();

  sobel = (int *) malloc(header.width*M*sizeof(int));

  for(i = 0; i < N*M; i+=3) // --> ByN
      image[i] = (image[i] + image[i+1] + image[i+2])/3;

  for(i = 0; i < N*M; i+=3)
  {
      if( (i > N) && (i%N != 0) && (i%(N-3) !=0) && (i < (M-1)*N)) {
         aux = valorPixel(image, N, i);
         if(primera)
         {
            min = max = aux;
            primera = false;
         }
         else if(aux < min) min = aux;
         else if(aux > max) max = aux;
      } else aux = 0;
      sobel[i/3] = aux;
  }

  float factor = 255.0/(float)(max-min);
  for(i = 0; i < N*M; i+=3)
      image[i] = image[i+1] = image[i+2] = sobel[i/3] * factor;

  t2 = GetTime();
  printf("\nMilisegundos Tot = %9f\n\n",t2-t1); 

  char nom[32];
  strcpy(nom, argv[2]);
  strcat(nom, "_");
  strcat(nom,argv[1]);

  SaveBMP(nom, &header, image);

}


