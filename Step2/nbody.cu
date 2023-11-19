/**
 * @file      nbody.cu
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

#include <device_launch_parameters.h>

#include "nbody.cuh"

#define FLOAT_MIN 1.17549e-38

/* Constants */
constexpr float G                  = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * CUDA kernel to calculate new particles velocity and position
 * @param pIn  - particles in
 * @param pOut - particles out
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
__global__ void calculateVelocity(Particles pIn, Particles pOut, const unsigned N, float dt)
{
 // determinate ID of thread and total number of threads
  const unsigned ix     = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned stride = gridDim.x * blockDim.x;

  //shared particles
  extern __shared__ float s[];

  // for simple indexing
  float* const inPosX    = pIn.posX;
  float* const inPosY    = pIn.posY;
  float* const inPosZ    = pIn.posZ;
  float* const inVelX    = pIn.velX;
  float* const inVelY    = pIn.velY;
  float* const inVelZ    = pIn.velZ;
  float* const inWeight  = pIn.weight;

  // in faster shared memory
  float* const SinPosX    = s;
  float* const SinPosY    = &(SinPosX[BLOCK_SIZE]);
  float* const SinPosZ    = &(SinPosY[BLOCK_SIZE]);
  float* const SinVelX    = &(SinPosZ[BLOCK_SIZE]);
  float* const SinVelY    = &(SinVelX[BLOCK_SIZE]);
  float* const SinVelZ    = &(SinVelY[BLOCK_SIZE]);
  float* const SinWeight  = &(SinVelZ[BLOCK_SIZE]);

  float* const outPosX   = pOut.posX;
  float* const outPosY   = pOut.posY;
  float* const outPosZ   = pOut.posZ;
  float* const outVelX   = pOut.velX;
  float* const outVelY   = pOut.velY;
  float* const outVelZ   = pOut.velZ;
  float* const outWeight = pOut.weight;

  const unsigned numBlocks = (N - 1) / BLOCK_SIZE + 1;
  
  // iterate over all object for one threat
  for (unsigned i = ix; i < N; i += stride) {
    float newVelX = 0.f;
    float newVelY = 0.f;
    float newVelZ = 0.f;

    float colisionVelX = 0.f;
    float colisionVelY = 0.f;
    float colisionVelZ = 0.f;

    const float posX   = inPosX[i];
    const float posY   = inPosY[i];
    const float posZ   = inPosZ[i];
    const float velX   = inVelX[i];
    const float velY   = inVelY[i];
    const float velZ   = inVelZ[i];
    const float weight = inWeight[i];

    // copy block size
    const unsigned relBlockDim = (i - threadIdx.x + blockDim.x > N) ? N - i + threadIdx.x : blockDim.x;

    // iterate over all objects
    for (unsigned block = 0; block < numBlocks; ++block) {
      const unsigned int start = block * BLOCK_SIZE;
      const unsigned int size  = (start + BLOCK_SIZE > N) ? N - start : BLOCK_SIZE;

      // wait until done to not rewrite shared memory
      __syncthreads();

      // load particles to shared memory
      for (unsigned j = threadIdx.x; j < size; j += relBlockDim) {
        SinPosX[j]    = inPosX[j + start];
        SinPosY[j]    = inPosY[j + start];
        SinPosZ[j]    = inPosZ[j + start];
        SinVelX[j]    = inVelX[j + start];
        SinVelY[j]    = inVelY[j + start];
        SinVelZ[j]    = inVelZ[j + start];
        SinWeight[j]  = inWeight[j + start];
      }

      // wait until load particles to shared memory
      __syncthreads();

      for (unsigned j = 0; j < size; ++j) {
        const float otherPosX   = SinPosX[j];
        const float otherPosY   = SinPosY[j];
        const float otherPosZ   = SinPosZ[j];
        const float otherVelX   = SinVelX[j];
        const float otherVelY   = SinVelY[j];
        const float otherVelZ   = SinVelZ[j];
        const float otherWeight = SinWeight[j];

        const float dx = otherPosX - posX;
        const float dy = otherPosY - posY;
        const float dz = otherPosZ - posZ;

        const float r2 = dx * dx + dy * dy + dz * dz;
        const float r = sqrt(r2) + FLOAT_MIN; // to awoid zero div

        const float f = G * weight * otherWeight / r2 + FLOAT_MIN; // to awoid zero div

        // calculate new velocity
        newVelX += (r > COLLISION_DISTANCE) ? dx / r * f : 0;
        newVelY += (r > COLLISION_DISTANCE) ? dy / r * f : 0;
        newVelZ += (r > COLLISION_DISTANCE) ? dz / r * f : 0;
        
        colisionVelX += (r > 0.f && r < COLLISION_DISTANCE) ? 
          (((weight * velX - otherWeight * velX + 2.f * otherWeight * otherVelX) / (weight + otherWeight)) - velX) : 0;
        colisionVelY += (r > 0.f && r < COLLISION_DISTANCE) ?
          (((weight * velY - otherWeight * velY + 2.f * otherWeight * otherVelY) / (weight + otherWeight)) - velY) : 0;
        colisionVelZ += (r > 0.f && r < COLLISION_DISTANCE) ? 
          (((weight * velZ - otherWeight * velZ + 2.f * otherWeight * otherVelZ) / (weight + otherWeight)) - velZ) : 0;
      }
    }

    newVelX *= dt / weight;
    newVelY *= dt / weight;
    newVelZ *= dt / weight;

    //colisition update speed

    newVelX += colisionVelX;
    newVelY += colisionVelY;
    newVelZ += colisionVelZ;

    //update position

    outVelX[i]   = velX + newVelX;
    outVelY[i]   = velY + newVelY;
    outVelZ[i]   = velZ + newVelZ;
    outWeight[i] = weight;

    outPosX[i]   = posX + outVelX[i] * dt;
    outPosY[i]   = posY + outVelY[i] * dt;
    outPosZ[i]   = posZ + outVelZ[i] * dt;
  }  
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Kernel to calculate particles center of mass
 * @param p    - particles
 * @param N    - Number of particles
 */
__device__ static inline void centerOfMassReduction(float4* a, const float4* b)
{
  float4 d = {b->x - a->x,
              b->y - a->y,
              b->z - a->z,
              (a->w + b->w) > 0.f ? (b->w / (a->w + b->w)) : 0.f};

  a->x += d.x * d.w;
  a->y += d.y * d.w;
  a->z += d.z * d.w;
  a->w += b->w;
}

/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles p, float4* com, int* lock, const unsigned N)
{
  // determinate ID of thread and total number of threads
  const unsigned ix     = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned stride = gridDim.x * blockDim.x;

  // for simple indexing
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pWeight = p.weight;

  float4 local_com = {0, 0, 0, 0};

  // iterate over all object for one threat
  for (unsigned i = ix; i < N; i += stride) {
    const float4 particle = {pPosX[i], pPosY[i], pPosZ[i], pWeight[i]};

    centerOfMassReduction(&local_com, &particle);
  }

  __syncthreads(); // do not enter in critical section to early
  if (threadIdx.x == 0) while (atomicCAS(lock,0,1) != 0); //lock  but only by one threat on block
  __syncthreads(); // wait for critical section

  for (unsigned t = 0; t < blockDim.x; t++) {
    if (t == threadIdx.x) { // only one thread in block in one time
      centerOfMassReduction(com, &local_com);
    }

    //__threadfence();
    __syncthreads(); // wait until threat done
  }

  if (threadIdx.x == 0) atomicExch(lock, 0); // unlock
}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassRef(MemDesc& memDesc)
{
  float4 com{};

  for (std::size_t i{}; i < memDesc.getDataSize(); i++)
  {
    const float3 pos = {memDesc.getPosX(i), memDesc.getPosY(i), memDesc.getPosZ(i)};
    const float  w   = memDesc.getWeight(i);

    // Calculate the vector on the line connecting current body and most recent position of center-of-mass
    // Calculate weight ratio only if at least one particle isn't massless
    const float4 d = {pos.x - com.x,
                      pos.y - com.y,
                      pos.z - com.z,
                      ((memDesc.getWeight(i) + com.w) > 0.0f)
                        ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w))
                        : 0.0f};

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += d.x * d.w;
    com.y += d.y * d.w;
    com.z += d.z * d.w;
    com.w += w;
  }

  return com;
}// enf of centerOfMassRef
//----------------------------------------------------------------------------------------------------------------------
