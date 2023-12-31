/**
 * @file      main.cu
 *
 * @author    Lukáš Plevač \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xpleva07@vutbr.cz
 *
 * @brief     PCG Assignment 1
 *
 * @version   2023
 *
 * @date      04 October   2023, 09:00 (created) \n
 */

#include <cmath>
#include <cstdio>
#include <chrono>
#include <string>

#include "nbody.cuh"
#include "h5Helper.h"

/**
 * @brief CUDA error checking macro
 * @param call CUDA API call
 */
#define CUDA_CALL(call) \
  do { \
    const cudaError_t _error = (call); \
    if (_error != cudaSuccess) \
    { \
      std::fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(_error)); \
      std::exit(EXIT_FAILURE); \
    } \
  } while(0)

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  if (argc != 10)
  {
    std::printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
    std::exit(1);
  }

  // Number of particles
  const unsigned N                   = static_cast<unsigned>(std::stoul(argv[1]));
  // Length of time step
  const float    dt                  = std::stof(argv[2]);
  // Number of steps
  const unsigned steps               = static_cast<unsigned>(std::stoul(argv[3]));
  // Number of thread blocks
  const unsigned simBlockDim         = static_cast<unsigned>(std::stoul(argv[4]));
  // Write frequency
  const unsigned writeFreq           = static_cast<unsigned>(std::stoul(argv[5]));
  // number of reduction threads
  const unsigned redTotalThreadCount = static_cast<unsigned>(std::stoul(argv[6]));
  // Number of reduction threads/blocks
  const unsigned redBlockDim         = static_cast<unsigned>(std::stoul(argv[7]));

  // Size of the simulation CUDA grid - number of blocks
  const unsigned simGridDim = (N + simBlockDim - 1) / simBlockDim;
  // Size of the reduction CUDA grid - number of blocks
  const unsigned redGridDim = (redTotalThreadCount + redBlockDim - 1) / redBlockDim;

  // Log benchmark setup
  std::printf("       NBODY GPU simulation\n"
              "N:                       %u\n"
              "dt:                      %f\n"
              "steps:                   %u\n"
              "threads/block:           %u\n"
              "blocks/grid:             %u\n"
              "reduction threads/block: %u\n"
              "reduction blocks/grid:   %u\n",
              N, dt, steps, simBlockDim, simGridDim, redBlockDim, redGridDim);

  const std::size_t recordsCount = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;

  Particles hParticles{};
  float4*   hCenterOfMass{};

  dim3 simBlockDimDim3(simBlockDim, 1, 1);
  dim3 simGridDimDim3 (simGridDim,  1, 1);

  dim3 redBlockDimDim3(redBlockDim, 1, 1);
  dim3 redGridDimDim3 (redGridDim,  1, 1);

  /********************************************************************************************************************/
  /*                              TODO: CPU side memory allocation (pinned)                                           */
  /********************************************************************************************************************/
  hParticles.posX   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.posY   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.posZ   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.velX   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.velY   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.velZ   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.weight = static_cast<float*>(operator new[](N * sizeof(float)));
  hCenterOfMass     = static_cast<float4*>(operator new[](sizeof(float4)));


  /********************************************************************************************************************/
  /*                              TODO: Fill memory descriptor layout                                                 */
  /********************************************************************************************************************/
  /*
   * Caution! Create only after CPU side allocation
   * parameters:
   *                            Stride of two            Offset of the first
   *       Data pointer       consecutive elements        element in FLOATS,
   *                          in FLOATS, not bytes            not bytes
  */
  MemDesc md(hParticles.posX,           1,                         0,
             hParticles.posY,           1,                         0,
             hParticles.posZ,           1,                         0,
             hParticles.velX,           1,                         0,
             hParticles.velY,           1,                         0,
             hParticles.velZ,           1,                         0,
             hParticles.weight,         1,                         0,
             N,
             recordsCount);

  // Initialisation of helper class and loading of input data
  H5Helper h5Helper(argv[8], argv[9], md);

  try
  {
    h5Helper.init();
    h5Helper.readParticleData();
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return EXIT_FAILURE;
  }

  Particles dParticles[2]{};
  float4*   dCenterOfMass{};
  int*      dLock{};

  /********************************************************************************************************************/
  /*                                     TODO: GPU side memory allocation                                             */
  /********************************************************************************************************************/
  #pragma unroll
  for (unsigned i = 0; i < 2; i++) {
    CUDA_CALL(cudaMalloc(&(dParticles[i].posX),  N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&(dParticles[i].posY),  N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&(dParticles[i].posZ),  N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&(dParticles[i].velX),  N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&(dParticles[i].velY),  N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&(dParticles[i].velZ),  N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&(dParticles[i].weight), N * sizeof(float)));
  }

  CUDA_CALL(cudaMalloc(&dCenterOfMass, sizeof(float4)));
  CUDA_CALL(cudaMalloc(&dLock, sizeof(int)));
  

  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer CPU -> GPU                                             */
  /********************************************************************************************************************/
  CUDA_CALL(cudaMemcpy(dParticles[0].posX,   hParticles.posX,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].posY,   hParticles.posY,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].posZ,   hParticles.posZ,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velX,   hParticles.velX,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velY,   hParticles.velY,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velZ,   hParticles.velZ,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].weight, hParticles.weight, N * sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemset(dLock, 0, sizeof(int)));


  /********************************************************************************************************************/
  /*                                     TODO: Clear GPU center of mass                                               */
  /********************************************************************************************************************/
  CUDA_CALL(cudaMemset(dCenterOfMass, 0, sizeof(float4)));


  /********************************************************************************************************************/
  /*                           TODO: Declare and create necessary CUDA streams and events                             */
  /********************************************************************************************************************/
  CUDA_CALL(cudaMemset(dCenterOfMass, 0, sizeof(float4)));

  unsigned nStreams = 3;

  cudaStream_t stream[nStreams];
  for (int i = 0; i < nStreams; i ++) {
    CUDA_CALL(cudaStreamCreate(&stream[i]));
  }

  cudaEvent_t stopEvent, startEvent;
  CUDA_CALL(cudaEventCreate(&stopEvent));
  CUDA_CALL(cudaEventCreate(&startEvent));
  
  // Get CUDA device warp size
  int device;
  int warpSize;

  CUDA_CALL(cudaGetDevice(&device));
  CUDA_CALL(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device));

  /********************************************************************************************************************/
  /*                                  TODO: Set dynamic shared memory computation                                     */
  /********************************************************************************************************************/
  const std::size_t sharedMemSize = 7 * BLOCK_SIZE * sizeof(float);
  const std::size_t redSharedMemSize = (redBlockDim / warpSize + 0.5) * sizeof(float4);   // you can use warpSize variable

  // Lambda for checking if we should write current step to the file
  auto shouldWrite = [writeFreq](unsigned s) -> bool
  {
    return writeFreq > 0u && (s % writeFreq == 0u);
  };

  // Lamda for getting record number
  auto getRecordNum = [writeFreq](unsigned s) -> unsigned
  {
    return s / writeFreq;
  };

  // Start measurement
  const auto start = std::chrono::steady_clock::now();

  /********************************************************************************************************************/
  /*            TODO: Edit the loop to work asynchronously and overlap computation with data transfers.               */
  /*                  Use shouldWrite lambda to determine if data should be outputted to file.                        */
  /*                           if (shouldWrite(s, writeFreq)) { ... }                                                 */
  /*                        Use getRecordNum lambda to get the record number.                                         */
  /********************************************************************************************************************/
  for (unsigned s = 0u; s < steps; ++s)
  {
    const unsigned srcIdx = s % 2;        // source particles index
    const unsigned dstIdx = (s + 1) % 2;  // destination particles index

    // mark all under for wait
    CUDA_CALL(cudaEventRecord(startEvent, 0));

    /******************************************************************************************************************/
    /*                TODO: GPU kernels invocation with correctly set dynamic memory size and stream                  */
    /******************************************************************************************************************/
    if (shouldWrite(s)) {
      // copy particles positions from GPU to CPU STREAM 0
      CUDA_CALL(cudaMemcpyAsync(hParticles.posX,   dParticles[srcIdx].posX,   N * sizeof(float), cudaMemcpyDeviceToHost, stream[0]));
      CUDA_CALL(cudaMemcpyAsync(hParticles.posY,   dParticles[srcIdx].posY,   N * sizeof(float), cudaMemcpyDeviceToHost, stream[0]));
      CUDA_CALL(cudaMemcpyAsync(hParticles.posZ,   dParticles[srcIdx].posZ,   N * sizeof(float), cudaMemcpyDeviceToHost, stream[0]));
      CUDA_CALL(cudaMemcpyAsync(hParticles.velX,   dParticles[srcIdx].velX,   N * sizeof(float), cudaMemcpyDeviceToHost, stream[0]));
      CUDA_CALL(cudaMemcpyAsync(hParticles.velY,   dParticles[srcIdx].velY,   N * sizeof(float), cudaMemcpyDeviceToHost, stream[0]));
      CUDA_CALL(cudaMemcpyAsync(hParticles.velZ,   dParticles[srcIdx].velZ,   N * sizeof(float), cudaMemcpyDeviceToHost, stream[0]));

      // calculate center of mass and copy to CPU STREAM 1
      centerOfMass <<< redGridDimDim3, redBlockDimDim3, redSharedMemSize, stream[1]>>> (dParticles[srcIdx], dCenterOfMass, dLock, N);
      CUDA_CALL(cudaMemcpyAsync(hCenterOfMass, dCenterOfMass, sizeof(float4), cudaMemcpyDeviceToHost, stream[1]));
      CUDA_CALL(cudaMemsetAsync(dCenterOfMass, 0, sizeof(float4), stream[1]));
    }

    // claclulate new particles pocition STREAM 2
    calculateVelocity <<< simGridDimDim3, simBlockDimDim3, sharedMemSize, stream[2]>>> (dParticles[srcIdx], dParticles[dstIdx], N, dt);

    // wait until all asic jobs done
    CUDA_CALL(cudaEventRecord(stopEvent, 0));
    CUDA_CALL(cudaEventSynchronize(stopEvent));

    if (shouldWrite(s)) {
      const auto recordNum = getRecordNum(s);
      
      h5Helper.writeParticleData(recordNum);
      h5Helper.writeCom(*hCenterOfMass, recordNum);
    }
  }

  const unsigned resIdx = steps % 2;    // result particles index

  /********************************************************************************************************************/
  /*                          TODO: Invocation of center of mass kernel, do not forget to add                         */
  /*                              additional synchronization and set appropriate stream                               */
  /********************************************************************************************************************/
  centerOfMass <<< redGridDimDim3, redBlockDimDim3, redSharedMemSize >>> (dParticles[resIdx], dCenterOfMass, dLock, N);

  // Wait for all CUDA kernels to finish
  CUDA_CALL(cudaDeviceSynchronize());

  // End measurement
  const auto end = std::chrono::steady_clock::now();

  // Approximate simulation wall time
  const float elapsedTime = std::chrono::duration<float>(end - start).count();
  std::printf("Time: %f s\n", elapsedTime);

  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer GPU -> CPU                                             */
  /********************************************************************************************************************/
  CUDA_CALL(cudaMemcpy(hParticles.posX,   dParticles[resIdx].posX,   N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posY,   dParticles[resIdx].posY,   N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posZ,   dParticles[resIdx].posZ,   N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velX,   dParticles[resIdx].velX,   N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velY,   dParticles[resIdx].velY,   N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velZ,   dParticles[resIdx].velZ,   N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.weight, dParticles[resIdx].weight, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hCenterOfMass,     dCenterOfMass,                sizeof(float4), cudaMemcpyDeviceToHost));


  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n",
              refCenterOfMass.x,
              refCenterOfMass.y,
              refCenterOfMass.z,
              refCenterOfMass.w);

  std::printf("Center of mass on GPU: %f, %f, %f, %f\n",
              hCenterOfMass->x,
              hCenterOfMass->y,
              hCenterOfMass->z,
              hCenterOfMass->w);

  // Writing final values to the file
  h5Helper.writeComFinal(*hCenterOfMass);
  h5Helper.writeParticleDataFinal();

  /********************************************************************************************************************/
  /*                                  TODO: CUDA streams and events destruction                                       */
  /********************************************************************************************************************/
  for (int i = 0; i < nStreams; i ++) {
    CUDA_CALL(cudaStreamDestroy(stream[i]));
  }

  CUDA_CALL( cudaEventDestroy(startEvent) );
  CUDA_CALL( cudaEventDestroy(stopEvent) );

  /********************************************************************************************************************/
  /*                                     TODO: GPU side memory deallocation                                           */
  /********************************************************************************************************************/
  #pragma unroll
  for (unsigned i = 0; i < 2; i++) {
    CUDA_CALL(cudaFree(dParticles[i].posX));
    CUDA_CALL(cudaFree(dParticles[i].posY));
    CUDA_CALL(cudaFree(dParticles[i].posZ));
    CUDA_CALL(cudaFree(dParticles[i].velX));
    CUDA_CALL(cudaFree(dParticles[i].velY));
    CUDA_CALL(cudaFree(dParticles[i].velZ));
    CUDA_CALL(cudaFree(dParticles[i].weight));
  }
  
  /********************************************************************************************************************/
  /*                                     TODO: CPU side memory deallocation                                           */
  /********************************************************************************************************************/
  operator delete[](hParticles.posX);
  operator delete[](hParticles.posY);
  operator delete[](hParticles.posZ);
  operator delete[](hParticles.velX);
  operator delete[](hParticles.velY);
  operator delete[](hParticles.velZ);
  operator delete[](hParticles.weight);
  operator delete[](hCenterOfMass);

}// end of main
//----------------------------------------------------------------------------------------------------------------------
