#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <mpi.h>

// Constantes
#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*
Muestra el correspondiente errro en la lectura de fichero de datos
*/
void showFileError(int error, char *filename)
{
	printf("Error\n");
	switch (error)
	{
	case -1:
		fprintf(stderr, "\tEl fichero %s contiene demasiadas columnas.\n", filename);
		fprintf(stderr, "\tSe supero el tamano maximo de columna MAXLINE: %d.\n", MAXLINE);
		break;
	case -2:
		fprintf(stderr, "Error leyendo el fichero %s.\n", filename);
		break;
	case -3:
		fprintf(stderr, "Error escibiendo en el fichero %s.\n", filename);
		break;
	}
	fflush(stderr);
}

/*
Lectura del fichero para determinar el numero de filas y muestras (samples)
*/
int readInput(char *filename, int *lines, int *samples)
{
	FILE *fp;
	char line[MAXLINE] = "";
	char *ptr;
	const char *delim = "\t";
	int contlines, contsamples;

	contlines = 0;

	if ((fp = fopen(filename, "r")) != NULL)
	{
		while (fgets(line, MAXLINE, fp) != NULL)
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
			contlines++;
			ptr = strtok(line, delim);
			contsamples = 0;
			while (ptr != NULL)
			{
				contsamples++;
				ptr = strtok(NULL, delim);
			}
		}
		fclose(fp);
		*lines = contlines;
		*samples = contsamples;
		return 0;
	}
	else
	{
		return -2;
	}
}

/*
Carga los datos del fichero en la estructra data
*/
int readInput2(char *filename, float *data)
{
	FILE *fp;
	char line[MAXLINE] = "";
	char *ptr;
	const char *delim = "\t";
	int i = 0;

	if ((fp = fopen(filename, "rt")) != NULL)
	{
		while (fgets(line, MAXLINE, fp) != NULL)
		{
			ptr = strtok(line, delim);
			while (ptr != NULL)
			{
				data[i] = atof(ptr);
				i++;
				ptr = strtok(NULL, delim);
			}
		}
		fclose(fp);
		return 0;
	}
	else
	{
		return -2; // No file found
	}
}

/*
Escribe en el fichero de salida la clase a la que perteneces cada muestra (sample)
*/
int writeResult(int *classMap, int lines, const char *filename)
{
	FILE *fp;

	if ((fp = fopen(filename, "wt")) != NULL)
	{
		for (int i = 0; i < lines; i++)
		{
			fprintf(fp, "%d\n", classMap[i]);
		}
		fclose(fp);

		return 0;
	}
	else
	{
		return -3; // No file found
	}
}

/*
Copia el valor de los centroides de data a centroids usando centroidPos como
mapa de la posicion que ocupa cada centroide en data
*/
void initCentroids(const float *data, float *centroids, int *centroidPos, int samples, int K)
{
	int i;
	int idx;
	for (i = 0; i < K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i * samples], &data[idx * samples], (samples * sizeof(float)));
	}
}

/*
Calculo de la distancia euclidea
*/
float euclideanDistance(float *point, float *center, int samples)
{
	float dist = 0.0;
	for (int i = 0; i < samples; i++)
	{
		dist += (point[i] - center[i]) * (point[i] - center[i]);
	}
	dist = sqrt(dist);
	return (dist);
}

/*
Funcion de clasificacion, asigna una clase a cada elemento de data
*/
int classifyPoints(float *data, float *centroids, int *classMap, int lines, int samples, int K)
{
	int i, j;
	int class;
	float dist, minDist;
	int changes = 0;
	for (i = 0; i < lines; i++)
	{
		class = 1;
		minDist = FLT_MAX;
		for (j = 0; j < K; j++)
		{
			dist = euclideanDistance(&data[i * samples], &centroids[j * samples], samples);

			if (dist < minDist)
			{
				minDist = dist;
				class = j + 1;
			}
		}
		if (classMap[i] != class)
		{
			changes++;
		}
		classMap[i] = class;
	}
	return (changes);
}

/*
Recalcula los centroides a partir de una nueva clasificacion
*/
float recalculateCentroids(float *data, float *centroids, int *classMap, int lines, int samples, int K, int rank)
{
	int class, i, j;

	float maxDist = FLT_MIN;

	int *localPointsPerClass;
	localPointsPerClass = (int *)calloc(K, sizeof(int));
	int *pointsPerClass;
	float *localAuxCentroids;
	localAuxCentroids = (float *)calloc(K * samples, sizeof(float));
	float *auxCentroids;
	float *distCentroids;
	if (rank == 0)
	{
		auxCentroids = (float *)calloc(K * samples, sizeof(float));
		pointsPerClass = (int *)calloc(K, sizeof(int));
		distCentroids = (float *)malloc(K * sizeof(float));
		if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
		{
			fprintf(stderr, "Error alojando memoria\n");
			exit(-4);
		}
	}
	// pointPerClass: numero de puntos clasificados en cada clase
	// auxCentroids: media de los puntos de cada clase

	for (i = 0; i < lines; i++)
	{
		class = classMap[i];

		localPointsPerClass[class - 1] = localPointsPerClass[class - 1] + 1;

		for (j = 0; j < samples; j++)
		{
			localAuxCentroids[(class - 1) * samples + j] += data[i * samples + j];
		}
	}

	MPI_Reduce(localPointsPerClass, pointsPerClass, K, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(localAuxCentroids, auxCentroids, K * samples, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{

		for (i = 0; i < K; i++)
		{
			for (j = 0; j < samples; j++)
			{
				auxCentroids[i * samples + j] /= pointsPerClass[i];
			}
		}

		for (i = 0; i < K; i++)
		{
			distCentroids[i] = euclideanDistance(&centroids[i * samples], &auxCentroids[i * samples], samples);
			if (distCentroids[i] > maxDist)
			{
				maxDist = distCentroids[i];
			}
		}
		memcpy(centroids, auxCentroids, (K * samples * sizeof(float)));
	}

	free(localPointsPerClass);
	free(localAuxCentroids);
	if (rank == 0)
	{
		free(distCentroids);
		free(pointsPerClass);
		free(auxCentroids);
	}
	return (maxDist);
}

int printArray(float *array, int tam)
{
	for (int i = 0; i < tam; i++)
		printf("%lf\n", array[i]);
	fflush(stdout);
}

int printArray2(int *array, int tam)
{
	for (int i = 0; i < tam; i++)
		printf("%d\n", array[i]);
	fflush(stdout);
}

int main(int argc, char *argv[])
{

	int rank, nprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// START CLOCK***************************************
	double time = MPI_Wtime();
	//**************************************************
	/*
	 * PARAMETROS
	 *
	 * argv[1]: Fichero de datos de entrada
	 * argv[2]: Numero de clusters
	 * argv[3]: Numero maximo de iteraciones del metodo. Condicion de fin del algoritmo
	 * argv[4]: Porcentaje minimo de cambios de clase. Condicion de fin del algoritmo.
	 * 			Si entre una iteracion y la siguiente el porcentaje cambios de clase es menor que
	 * 			este procentaje, el algoritmo para.
	 * argv[5]: Precision en la distancia de centroides depuesde la actualizacion
	 * 			Es una condicion de fin de algoritmo. Si entre una iteracion del algoritmo y la
	 * 			siguiente la distancia maxima entre centroides es menor que esta precsion el
	 * 			algoritmo para.
	 * argv[6]: Fichero de salida. Clase asignada a cada linea del fichero.
	 * */
	if (argc != 7)
	{
		fprintf(stderr, "EXECUTION ERROR KMEANS Iterative: Parameters are not correct.\n");
		fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	// Lectura de los datos de entrada
	//  lines = numero de puntos;  samples = numero de dimensiones por punto
	int lines = 0, samples = 0, my_lines = 0;

	int *sendcnts;
	int *displs;

	int *classMapCnts;
	int *classMapDispls;

	float *data;

	int K = atoi(argv[2]);
	int maxIterations = atoi(argv[3]);
	float maxThreshold = atof(argv[5]);
	int minChanges;

	int *centroidPos;
	float *centroids;
	int *classMap;

	float distCent;

	int it = 0;
	int changes = 0;

	int error;

	if (rank == 0)
	{
		error = readInput(argv[1], &lines, &samples);
		if (error != 0)
		{
			showFileError(error, argv[1]);
			MPI_Abort(MPI_COMM_WORLD, error);
		}

		data = (float *)calloc(lines * samples, sizeof(float));
		if (data == NULL)
		{
			fprintf(stderr, "Error alojando memoria\n");
			MPI_Abort(MPI_COMM_WORLD, -4);
		}
		error = readInput2(argv[1], data);
		if (error != 0)
		{
			showFileError(error, argv[1]);
			MPI_Abort(MPI_COMM_WORLD, error);
		}

		// prametros del algoritmo. La entrada no esta valdidada

		minChanges = (int)(lines * atof(argv[4]) / 100.0);

		// poscion de los centroides en data
		centroidPos = (int *)calloc(K, sizeof(int));
		centroids = (float *)calloc(K * samples, sizeof(float));
		classMap = (int *)calloc(lines, sizeof(int));
		// Otras variables

		if (centroidPos == NULL || centroids == NULL || classMap == NULL)
		{
			fprintf(stderr, "Error alojando memoria\n");
			MPI_Abort(MPI_COMM_WORLD, -4);
		}

		// Centroides iniciales
		srand(0);
		int i;
		for (i = 0; i < K; i++)
			centroidPos[i] = rand() % lines;

		// Carga del array centroids con los datos del array data
		// los centroides son puntos almacenados en data
		initCentroids(data, centroids, centroidPos, samples, K);

		// Resumen de datos caragos
		printf("\n\tFichero de datos: %s \n\tPuntos: %d\n\tDimensiones: %d\n", argv[1], lines, samples);
		printf("\tNumero de clusters: %d\n", K);
		printf("\tNumero maximo de iteraciones: %d\n", maxIterations);
		printf("\tNumero minimo de cambios: %d [%g%% de %d puntos]\n", minChanges, atof(argv[4]), lines);
		printf("\tPrecision maxima de los centroides: %f\n", maxThreshold);

		displs = (int *)malloc(nprocs * sizeof(int));

		sendcnts = (int *)malloc(nprocs * sizeof(int));

		classMapDispls = (int *)malloc(nprocs * sizeof(int));

		classMapCnts = (int *)malloc(nprocs * sizeof(int));

		for (int proc = 0; proc < nprocs; proc++)
		{
			int tmpSize = lines / nprocs;

			if (proc < (lines % nprocs))
				tmpSize++;

			sendcnts[proc] = tmpSize * samples;
			classMapCnts[proc] = tmpSize;
		}

		for (int proc = 0; proc < nprocs; proc++)
		{
			int tmpBegin = 0;
			int classBegin = 0;
			for (int i = 0; i < proc; i++)
			{
				tmpBegin += sendcnts[i];
				classBegin += sendcnts[i] / samples;
			}
			classMapDispls[proc] = classBegin;
			displs[proc] = tmpBegin;
		}
	}
	MPI_Bcast(&samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&minChanges, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lines, 1, MPI_INT, 0, MPI_COMM_WORLD);

	my_lines = lines / nprocs;

	if (rank < (lines % nprocs))
		my_lines++;

	float *my_data = (float *)calloc(my_lines * samples, sizeof(float));
	MPI_Scatterv(data, sendcnts, displs, MPI_FLOAT, my_data, my_lines * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

	int *my_classMap = calloc(my_lines, sizeof(int));

	if (rank != 0)
	{
		centroids = (float *)calloc(K * samples, sizeof(float));
	}
	else
	{
		free(data);
		free(centroidPos);
	}

	// END CLOCK*****************************************
	time = MPI_Wtime() - time;
	if (rank == 0)
		printf("\nAlojado de memoria: %f segundos\n", time);
	fflush(stdout);
	//**************************************************
	// START CLOCK***************************************
	time = MPI_Wtime();

	//**************************************************
	do
	{
		it++;
		// Calcula la distancia desde cada punto al centroide

		MPI_Bcast(centroids, K * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// Asigna cada punto al centroide mas cercano
		int localChanges = classifyPoints(my_data, centroids, my_classMap, my_lines, samples, K);

		MPI_Allreduce(&localChanges, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		// Recalcula los centroides: calcula la media dentro de cada centroide
		distCent = recalculateCentroids(my_data, centroids, my_classMap, my_lines, samples, K, rank);
		MPI_Bcast(&distCent, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

		if (rank == 0)
			printf("\n[%d] Cambios de cluster: %d\tMax. dist. centroides: %f", it, changes, distCent);

	} while ((changes > minChanges) && (it < maxIterations) && (distCent > maxThreshold));

	MPI_Gatherv(my_classMap, my_lines, MPI_INT, classMap, classMapCnts, classMapDispls, MPI_INT, 0, MPI_COMM_WORLD);

	// Condiciones de fin de la ejecucion
	if (rank == 0)
	{

		if (changes <= minChanges)
		{
			printf("\n\nCondicion de parada: Numero minimo de cambios alcanzado: %d [%d]", changes, minChanges);
		}
		else if (it >= maxIterations)
		{
			printf("\n\nCondicion de parada: Numero maximo de iteraciones alcanzado: %d [%d]", it, maxIterations);
		}
		else
		{
			printf("\n\nCondicion de parada: Precision en la actualizacion de centroides alcanzada: %g [%g]", distCent, maxThreshold);
		}

		// Escritura en fichero de la clasificacion de cada punto
		error = writeResult(classMap, lines, argv[6]);
		if (error != 0)
		{
			showFileError(error, argv[6]);
			MPI_Abort(MPI_COMM_WORLD, error);
		}

		// END CLOCK*****************************************
		time = MPI_Wtime() - time;
		printf("\nComputacion: %f segundos", time);
		fflush(stdout);
		//**************************************************
		// START CLOCK***************************************
		time = MPI_Wtime();
		//**************************************************

		// Liberacion de la memoria dinamica
		free(classMap);
		free(sendcnts);
		free(classMapCnts);
		free(classMapDispls);

		// END CLOCK*****************************************
		time = MPI_Wtime() - time;
		printf("\n\nLiberacion: %f segundos\n", time);
		fflush(stdout);
		//***************************************************/
	}

	free(centroids);
	free(my_classMap);

	MPI_Finalize();
	return 0;
}
