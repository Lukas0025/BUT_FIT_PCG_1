/**
 * @file      main.cu
 *
 * @author    Name Surname \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xlogin00@fit.vutbr.cz
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

  dim3 simBlockDimDim3(simBlockDim, 1, 1);
  dim3 simGridDimDim3 (simGridDim,  1, 1);

  dim3 redBlockDimDim3(redBlockDim, 1, 1);
  dim3 redGridDimDim3 (redGridDim,  1, 1);

  /********************************************************************************************************************/
  /*                              TODO: CPU side memory allocation (pinned)                                           */
  /********************************************************************************************************************/

  // host particles
  hParticles.posX   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.posY   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.posZ   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.velX   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.velY   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.velZ   = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.weight = static_cast<float*>(operator new[](N * sizeof(float)));

  float4 hFinalCom;

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
  float4    *dFinalCom;
  int       *dLock;

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

  CUDA_CALL(cudaMalloc(&dFinalCom, sizeof(float4)));
  CUDA_CALL(cudaMalloc(&dLock, sizeof(int)));
  

  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer CPU -> GPU                                             */
  /********************************************************************************************************************/
  // Particles
  CUDA_CALL(cudaMemcpy(dParticles[0].posX,   hParticles.posX,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].posY,   hParticles.posY,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].posZ,   hParticles.posZ,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velX,   hParticles.velX,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velY,   hParticles.velY,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velZ,   hParticles.velZ,   N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].weight, hParticles.weight, N * sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemset(dFinalCom, 0, sizeof(float4)));
  CUDA_CALL(cudaMemset(dLock, 0, sizeof(int)));


  /********************************************************************************************************************/
  /*                                  TODO: Set dynamic shared memory computation                                     */
  /********************************************************************************************************************/
  const std::size_t sharedMemSize = 7 * N * sizeof(float);

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

  for (unsigned s = 0u; s < steps; ++s)
  {
    const unsigned srcIdx = s % 2;        // source particles index
    const unsigned dstIdx = (s + 1) % 2;  // destination particles index

    /******************************************************************************************************************/
    /*                   TODO: GPU kernel invocation with correctly set dynamic memory size                           */
    /******************************************************************************************************************/
    
    if (shouldWrite(s)) {
      const auto recordNum = getRecordNum(s);

      centerOfMass <<< redGridDimDim3, redBlockDimDim3 >>> (dParticles[srcIdx], dFinalCom, dLock, N);

      CUDA_CALL(cudaMemcpy(&hFinalCom, dFinalCom, sizeof(float4), cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemset(dFinalCom, 0, sizeof(float4)));

      //h5Helper.writeParticleData(recordNum);
      h5Helper.writeCom(hFinalCom, recordNum);
    }

    calculateVelocity <<< simGridDimDim3, simBlockDimDim3, sharedMemSize >>> (dParticles[srcIdx], dParticles[dstIdx], N, dt);
    CUDA_CALL(cudaDeviceSynchronize());
  }

  // Wait for all CUDA kernels to finish
  CUDA_CALL(cudaDeviceSynchronize());

  // End measurement
  const auto end = std::chrono::steady_clock::now();

  // Approximate simulation wall time
  const float elapsedTime = std::chrono::duration<float>(end - start).count();
  std::printf("Time: %f s\n", elapsedTime);

  const unsigned resIdx = steps % 2;    // result particles index

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


  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n",
              refCenterOfMass.x,
              refCenterOfMass.y,
              refCenterOfMass.z,
              refCenterOfMass.w);

  centerOfMass <<< redGridDimDim3, redBlockDimDim3 >>> (dParticles[resIdx], dFinalCom, dLock, N);
  
  CUDA_CALL(cudaMemcpy(&hFinalCom, dFinalCom, sizeof(float4), cudaMemcpyDeviceToHost));

  std::printf("Center of mass on GPU: %f, %f, %f, %f\n",
              hFinalCom.x,
              hFinalCom.y,
              hFinalCom.z,
              hFinalCom.w);

  // Writing final values to the file
  h5Helper.writeComFinal(hFinalCom);
  h5Helper.writeParticleDataFinal();

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

}// end of main
//----------------------------------------------------------------------------------------------------------------------
