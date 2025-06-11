#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cstdlib>
#include <ctime>

// Constants
const int NUM_PARTICLES = 1023;
const float RADIUS = 0.01f;
const int GRID_SIZE = 64;
const float WORLD_SIZE = 2.0f; // from -1 to 1

struct Particle {
    float x, y;
    float vx, vy;
};

// CUDA hash function for grid cell index
__device__ int getCellIndex(float x, float y) {
    int cx = (int)((x + 1.0f) / WORLD_SIZE * GRID_SIZE);
    int cy = (int)((y + 1.0f) / WORLD_SIZE * GRID_SIZE);
    cx = max(0, min(GRID_SIZE - 1, cx));
    cy = max(0, min(GRID_SIZE - 1, cy));
    return cy * GRID_SIZE + cx;
}

// Kernel to reset grid counts to zero
__global__ void resetGridCounts(int* gridCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < GRID_SIZE * GRID_SIZE) {
        gridCount[idx] = 0;
    }
}

// Kernel to assign particles to grid cells and update positions
__global__ void simulateParticles(Particle* particles, int* grid, int* gridCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;

    Particle& p = particles[i];

    // Update position
    p.x += p.vx;
    p.y += p.vy;

    // Bounce off walls
    if (p.x > 1.0f || p.x < -1.0f) p.vx = -p.vx;
    if (p.y > 1.0f || p.y < -1.0f) p.vy = -p.vy;

    // Compute grid cell
    int cell = getCellIndex(p.x, p.y);

    // Insert particle index atomically into grid cell
    int index = atomicAdd(&gridCount[cell], 1);
    if (index < NUM_PARTICLES) {  // avoid overflow
        grid[cell * NUM_PARTICLES + index] = i;
    }
}

// Kernel to handle collisions per particle
__global__ void collideParticles(Particle* particles, int* grid, int* gridCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;

    Particle& p = particles[i];
    int cell = getCellIndex(p.x, p.y);
    int count = gridCount[cell];

    for (int j = 0; j < count; ++j) {
        int otherIdx = grid[cell * NUM_PARTICLES + j];
        if (otherIdx <= i) continue; // prevent double processing and self

        Particle& q = particles[otherIdx];

        float dx = p.x - q.x;
        float dy = p.y - q.y;
        float distSq = dx * dx + dy * dy;
        float minDist = 2.0f * RADIUS;

        if (distSq < minDist * minDist) {
            float dist = sqrtf(distSq);
            if (dist == 0.0f) continue; // avoid division by zero

            float nx = dx / dist;
            float ny = dy / dist;

            // Relative velocity
            float dvx = p.vx - q.vx;
            float dvy = p.vy - q.vy;

            // Project relative velocity onto collision normal
            float relVel = dvx * nx + dvy * ny;

            if (relVel < 0.0f) {
                // Apply elastic response (equal mass case)
                float impulse = relVel;

                p.vx -= impulse * nx;
                p.vy -= impulse * ny;
                q.vx += impulse * nx;
                q.vy += impulse * ny;

                // Optional: separate overlapping particles to avoid sticking
                float overlap = 0.5f * (minDist - dist);
                p.x += nx * overlap;
                p.y += ny * overlap;
                q.x -= nx * overlap;
                q.y -= ny * overlap;
            }
        }
    }
}

// Host-side data
std::vector<Particle> h_particles;

// Device pointers
Particle* d_particles = nullptr;
int* d_grid = nullptr;
int* d_gridCount = nullptr;

void initParticles() {
    h_particles.resize(NUM_PARTICLES);
    for (auto& p : h_particles) {
        p.x = (rand() % 2000 - 1000) / 1000.0f;
        p.y = (rand() % 2000 - 1000) / 1000.0f;
        p.vx = ((rand() % 2000) - 1000) / 100000.0f;
        p.vy = ((rand() % 2000) - 1000) / 100000.0f;
    }

    cudaMemcpy(d_particles, h_particles.data(), sizeof(Particle) * NUM_PARTICLES, cudaMemcpyHostToDevice);
}

void updateParticlesCUDA() {
    // Reset grid counts
    resetGridCounts<<<(GRID_SIZE * GRID_SIZE + 255) / 256, 256>>>(d_gridCount);

    // Simulate motion & assign particles to grid cells
    simulateParticles<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles, d_grid, d_gridCount);

    // Wait for motion update and grid assignment
    cudaDeviceSynchronize();

    // Handle collisions per particle
    collideParticles<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles, d_grid, d_gridCount);

    // Copy back to host
    cudaMemcpy(h_particles.data(), d_particles, sizeof(Particle) * NUM_PARTICLES, cudaMemcpyDeviceToHost);
}

void drawParticles() {
    glPointSize(10.0f);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f);
    for (const auto& p : h_particles) {
        glVertex2f(p.x, p.y);
    }
    glEnd();
}

int main() {
    srand((unsigned)time(0));

    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA Ideal Gas", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glewInit();

    // Allocate device memory
    cudaMalloc(&d_particles, sizeof(Particle) * NUM_PARTICLES);
    cudaMalloc(&d_grid, sizeof(int) * GRID_SIZE * GRID_SIZE * NUM_PARTICLES);
    cudaMalloc(&d_gridCount, sizeof(int) * GRID_SIZE * GRID_SIZE);

    initParticles();

    while (!glfwWindowShouldClose(window)) {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        updateParticlesCUDA();

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        drawParticles();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cudaFree(d_particles);
    cudaFree(d_grid);
    cudaFree(d_gridCount);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

