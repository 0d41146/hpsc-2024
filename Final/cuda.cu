#include <iostream>
#include <vector>
#include <cmath>

#define NX  (41)
#define NY  (41)
#define NT  (500)
#define NIT (50)
#define DX  (2.0 / (NX - 1))
#define DY  (2.0 / (NY - 1))
#define DT  (0.01)
#define RHO (1.0)
#define NU  (0.02)

__global__ void kernel1(double *b, double *u, double *v) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (1 <= x && x < NX - 1 && 1 <= y && y < NY - 1) {
        b[y*NX+x] = RHO * (1.0 / DT
                        *((u[y*NX+(x+1)] - u[y*NX+(x-1)]) / (2.0 * DX) + (v[(y+1)*NX+x] - v[(y-1)*NX+x]) / (2.0 * DY))
                        - std::pow((u[y*NX+(x+1)] - u[y*NX+(x-1)]) / (2.0 * DX), 2.0)
                        - 2.0 * ((u[(y+1)*NX+x] - u[(y-1)*NX+x]) / (2.0 * DY) * (v[y*NX+(x+1)] - v[y*NX+(x-1)]) / (2.0 * DX))
                        - std::pow((v[(y+1)*NX+x] - v[(y-1)*NX+x]) / (2.0 * DY), 2.0));
    }
}

__global__ void kernel2(double *p, double *pn, double *b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (1 <= x && x < NX - 1 && 1 <= y && y < NY - 1) {
        p[y*NX+x] = (std::pow(DY, 2) * (pn[y*NX+(x+1)] + pn[y*NX+(x-1)])
                        + std::pow(DX, 2) * (pn[(y+1)*NX+x] + pn[(y-1)*NX+x])
                        - b[y*NX+x] * std::pow(DX, 2) * std::pow(DY, 2))
                        / (2 * (std::pow(DX, 2) + std::pow(DY, 2)));
    }
}

__global__ void kernel3(double *u, double *un, double *v, double *vn, double *p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (1 <= x && x < NX - 1 && 1 <= y && y < NY - 1) {
        u[y*NX+x] = un[y*NX+x] - un[y*NX+x] * DT / DX * (un[y*NX+x] - un[y*NX+(x-1)])
                        - un[y*NX+x] * DT / DY
                        * (un[y*NX+x] - un[(y-1)*NX+x])
                        - DT / (2.0 * RHO * DX)
                        * (p[y*NX+(x+1)] - p[y*NX+(x-1)])
                        + NU * DT / std::pow(DX, 2.0)
                        * (un[y*NX+(x+1)] - 2.0 * un[y*NX+x] + un[y*NX+(x-1)])
                        + NU * DT / std::pow(DY, 2.0)
                        * (un[(y+1)*NX+x] - 2.0 * un[y*NX+x] + un[(y-1)*NX+x]);

        v[y*NX+x] = vn[y*NX+x] - vn[y*NX+x] * DT / DX
                        * (vn[y*NX+x] - vn[y*NX+(x-1)])
                        - vn[y*NX+x] * DT / DY
                        * (vn[y*NX+x] - vn[(y-1)*NX+x])
                        - DT / (2.0 * RHO * DY)
                        * (p[(y+1)*NX+x] - p[(y-1)*NX+x])
                        + NU * DT / std::pow(DX, 2.0)
                        * (vn[y*NX+(x+1)] - 2.0 * vn[y*NX+x] + vn[y*NX+(x-1)])
                        + NU * DT / std::pow(DY, 2.0)
                        * (vn[(y+1)*NX+x] - 2.0 * vn[y*NX+x] + vn[(y-1)*NX+x]);
    }
}

__global__ void cp1(double *p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (0 <= y && y < NY) p[y*NX+(NX-1)]  = p[y*NX+(NX-2)];
    if (0 <= x && x < NX) p[0*NX+x]       = p[1*NX+x];
    if (0 <= y && y < NY) p[y*NX+0]       = p[y*NX+1];
    if (0 <= x && x < NX) p[(NY-1)*NX+x]  = 0.0;
}

__global__ void cp2(double *u, double *v) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (0 <= x && x < NX) u[0*NX+x]        = 0.0;
    if (0 <= y && y < NY) u[y*NX+0]        = 0.0;
    if (0 <= y && y < NY) u[y*NX+(NX-1)]   = 0.0;
    if (0 <= x && x < NX) u[(NY-1)*NX+x]   = 1.0;
    if (0 <= x && x < NX) v[0*NX+x]        = 0.0;
    if (0 <= x && x < NX) v[(NY-1)*NX+x]   = 0.0;
    if (0 <= y && y < NY) v[y*NX+0]        = 0.0;
    if (0 <= y && y < NY) v[y*NX+(NX-1)]   = 0.0;
}

int main() {
    double *u;
    double *v;
    double *p;
    double *b;
    cudaMallocManaged(&u, NY*NX*sizeof(double));
    cudaMallocManaged(&v, NY*NX*sizeof(double));
    cudaMallocManaged(&p, NY*NX*sizeof(double));
    cudaMallocManaged(&b, NY*NX*sizeof(double));

    dim3 block(16, 16, 1);
    dim3 grid((NX + block.x - 1) / block.x,
              (NY + block.y - 1) / block.y, 1);

    for (int t = 0; t < NT; ++t) {
        kernel1<<<grid, block>>>(b, u, v);

        for (int it = 0; it < NIT; ++it) {
            double *pn;
            cudaMallocManaged(&pn, NY*NX*sizeof(double));
            cudaMemcpy(pn, p, NY*NX*sizeof(double), cudaMemcpyHostToHost);

            kernel2<<<grid, block>>>(p, pn, b);

            cudaFree(pn);
            cp1<<<grid, block>>>(p);
        }

        double* un;
        double* vn;
        cudaMallocManaged(&un, NY*NX*sizeof(double));
        cudaMallocManaged(&vn, NY*NX*sizeof(double));
        cudaMemcpy(un, u, NY*NX*sizeof(double), cudaMemcpyHostToHost);
        cudaMemcpy(vn, v, NY*NX*sizeof(double), cudaMemcpyHostToHost);

        kernel3<<<grid, block>>>(u, un, v, vn, p);

        cudaFree(un);
        cudaFree(vn);
        cp2<<<grid, block>>>(u, v);
    }

    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);

    return 0;
}