/**
 * @file      nbody.cu
 *
 * @author    Name Surname \n
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
 * CUDA kernel to calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateGravitationVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  // determinate ID of thread and total number of threads
  const unsigned ix     = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned stride = gridDim.x * blockDim.x;

  // for simple indexing
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pVelX   = p.velX;
  float* const pVelY   = p.velY;
  float* const pVelZ   = p.velZ;
  float* const pWeight = p.weight;

  float* const tmpVelX = tmpVel.x;
  float* const tmpVelY = tmpVel.y;
  float* const tmpVelZ = tmpVel.z;

  // iterate over all object for one threat
  for (unsigned i = ix; i < N; i += stride) {
    float newVelX{};
    float newVelY{};
    float newVelZ{};

    const float posX   = pPosX[i];
    const float posY   = pPosY[i];
    const float posZ   = pPosZ[i];
    const float weight = pWeight[i];

    // iterate over all objects
    for (unsigned j = 0u; j < N; ++j) {
      const float otherPosX   = pPosX[j];
      const float otherPosY   = pPosY[j];
      const float otherPosZ   = pPosZ[j];
      const float otherWeight = pWeight[j];

      const float dx = otherPosX - posX;
      const float dy = otherPosY - posY;
      const float dz = otherPosZ - posZ;

      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = sqrt(r2) + FLOAT_MIN; // to awoid zero div

      const float f = G * weight * otherWeight / r2 + FLOAT_MIN; // to awoid zero div

      newVelX += (r > COLLISION_DISTANCE) ? dx / r * f : 0.f;
      newVelY += (r > COLLISION_DISTANCE) ? dy / r * f : 0.f;
      newVelZ += (r > COLLISION_DISTANCE) ? dz / r * f : 0.f;
    }

    newVelX *= dt / weight;
    newVelY *= dt / weight;
    newVelZ *= dt / weight;

    // write new final tmpVel
    tmpVelX[i] = newVelX;
    tmpVelY[i] = newVelY;
    tmpVelZ[i] = newVelZ;
  }  
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateCollisionVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt) {
  // determinate ID of thread and total number of threads
  const unsigned ix     = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned stride = gridDim.x * blockDim.x;

  // for simple indexing
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pVelX   = p.velX;
  float* const pVelY   = p.velY;
  float* const pVelZ   = p.velZ;
  float* const pWeight = p.weight;

  float* const tmpVelX = tmpVel.x;
  float* const tmpVelY = tmpVel.y;
  float* const tmpVelZ = tmpVel.z;

  // iterate over all object for one threat
  for (unsigned i = ix; i < N; i += stride) {
    float newVelX{};
    float newVelY{};
    float newVelZ{};

    const float posX   = pPosX[i];
    const float posY   = pPosY[i];
    const float posZ   = pPosZ[i];
    const float velX   = pVelX[i];
    const float velY   = pVelY[i];
    const float velZ   = pVelZ[i];
    const float weight = pWeight[i];

    // iterate over all objects
    for (unsigned j = 0u; j < N; ++j) {
      const float otherPosX   = pPosX[j];
      const float otherPosY   = pPosY[j];
      const float otherPosZ   = pPosZ[j];
      const float otherVelX   = pVelX[j];
      const float otherVelY   = pVelY[j];
      const float otherVelZ   = pVelZ[j];
      const float otherWeight = pWeight[j];

      const float dx = otherPosX - posX;
      const float dy = otherPosY - posY;
      const float dz = otherPosZ - posZ;

      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = std::sqrt(r2);

      newVelX += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((weight * velX - otherWeight * velX + 2.f * otherWeight * otherVelX) / (weight + otherWeight)) - velX)
                 : 0.f;
      newVelY += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((weight * velY - otherWeight * velY + 2.f * otherWeight * otherVelY) / (weight + otherWeight)) - velY)
                 : 0.f;
      newVelZ += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((weight * velZ - otherWeight * velZ + 2.f * otherWeight * otherVelZ) / (weight + otherWeight)) - velZ)
                 : 0.f;
    }

    // update TMP vel by colisions
    tmpVelX[i] += newVelX;
    tmpVelY[i] += newVelY;
    tmpVelZ[i] += newVelZ;
  }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void updateParticles(Particles p, Velocities tmpVel, const unsigned N, float dt) {
  // determinate ID of thread and total number of threads
  const unsigned ix     = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned stride = gridDim.x * blockDim.x;

  // for simple indexing
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pVelX   = p.velX;
  float* const pVelY   = p.velY;
  float* const pVelZ   = p.velZ;
  float* const pWeight = p.weight;

  float* const tmpVelX = tmpVel.x;
  float* const tmpVelY = tmpVel.y;
  float* const tmpVelZ = tmpVel.z;

  // iterate over all object for one threat
  for (unsigned i = ix; i < N; i += stride) {

    float posX = pPosX[i];
    float posY = pPosY[i];
    float posZ = pPosZ[i];

    float velX = pVelX[i];
    float velY = pVelY[i];
    float velZ = pVelZ[i];

    const float newVelX = tmpVelX[i];
    const float newVelY = tmpVelY[i];
    const float newVelZ = tmpVelZ[i];

    velX += newVelX;
    velY += newVelY;
    velZ += newVelZ;

    posX += velX * dt;
    posY += velY * dt;
    posZ += velZ * dt;

    pPosX[i] = posX;
    pPosY[i] = posY;
    pPosZ[i] = posZ;

    pVelX[i] = velX;
    pVelY[i] = velY;
    pVelZ[i] = velZ;
  }
}// end of update_particle
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
__global__ void centerOfMass(Particles p, float4* com, int* lock, const unsigned N) {
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
