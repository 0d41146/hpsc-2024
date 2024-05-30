#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

int main() {
    int     nx  = 41;
    int     ny  = 41;
    int     nt  = 500;
    int     nit = 50;
    double  dx  = 2.0 / (nx - 1);
    double  dy  = 2.0 / (ny - 1);
    double  dt  = 0.01;
    double  rho = 1.0;
    double  nu  = 0.02;

    std::vector<double> u(ny * nx, 0.0);
    std::vector<double> v(ny * nx, 0.0);
    std::vector<double> p(ny * nx, 0.0);
    std::vector<double> b(ny * nx, 0.0);

    for (int t = 0; t < nt; ++t) {
        #pragma omp parallel for
        for (int i = 0; i < (ny - 1) * (nx - 1); ++i) {
            int y = i / (nx - 1);
            int x = i % (nx - 1);
            if (x == 0) { ++y; ++x; }
            b[y*nx+x] = rho * (1.0 / dt
                            *((u[y*nx+(x+1)] - u[y*nx+(x-1)]) / (2.0 * dx) + (v[(y+1)*nx+x] - v[(y-1)*nx+x]) / (2.0 * dy))
                            - std::pow((u[y*nx+(x+1)] - u[y*nx+(x-1)]) / (2.0 * dx), 2.0)
                            - 2.0 * ((u[(y+1)*nx+x] - u[(y-1)*nx+x]) / (2.0 * dy) * (v[y*nx+(x+1)] - v[y*nx+(x-1)]) / (2.0 * dx))
                            - std::pow((v[(y+1)*nx+x] - v[(y-1)*nx+x]) / (2.0 * dy), 2.0));
        }

        for (int it = 0; it < nit; ++it) {
            std::vector<double> pn = p;
            #pragma omp parallel for
            for (int i = 0; i < (ny - 1) * (nx - 1); ++i) {
                int y = i / (nx - 1);
                int x = i % (nx - 1);
                if (x == 0) { ++y; ++x; }
                 p[y*nx+x] = (std::pow(dy, 2) * (pn[y*nx+(x+1)] + pn[y*nx+(x-1)])
                                + std::pow(dx, 2) * (pn[(y+1)*nx+x] + pn[(y-1)*nx+x])
                                - b[y*nx+x] * std::pow(dx, 2) * std::pow(dy, 2))
                                / (2 * (std::pow(dx, 2) + std::pow(dy, 2)));
            }

            #pragma omp parallel for
            for (int y = 0; y < ny; ++y)    p[y*nx+(nx-1)]  = p[y*nx+(nx-2)];
            #pragma omp parallel for
            for (int x = 0; x < nx; ++x)    p[0*nx+x]       = p[1*nx+x];
            #pragma omp parallel for
            for (int y = 0; y < ny; ++y)    p[y*nx+0]       = p[y*nx+1];
            #pragma omp parallel for
            for (int x = 0; x < nx; ++x)    p[(ny-1)*nx+x]  = 0.0;
        }

        std::vector<double> un = u;
        std::vector<double> vn = v;

        #pragma omp parallel for
        for (int i = 0; i < (ny - 1) * (nx - 1); ++i) {
            int y = i / (nx - 1);
            int x = i % (nx - 1);
            if (x == 0) { ++y; ++x; }
            u[y*nx+x] = un[y*nx+x] - un[y*nx+x] * dt / dx * (un[y*nx+x] - un[y*nx+(x-1)])
                            - un[y*nx+x] * dt / dy * (un[y*nx+x] - un[(y-1)*nx+x])
                            - dt / (2.0 * rho * dx) * (p[y*nx+(x+1)] - p[y*nx+(x-1)])
                            + nu * dt / std::pow(dx, 2.0)
                            * (un[y*nx+(x+1)] - 2.0 * un[y*nx+x] + un[y*nx+(x-1)])
                            + nu * dt / std::pow(dy, 2.0)
                            * (un[(y+1)*nx+x] - 2.0 * un[y*nx+x] + un[(y-1)*nx+x]);
            v[y*nx+x] = vn[y*nx+x] - vn[y*nx+x] * dt / dx * (vn[y*nx+x] - vn[y*nx+(x-1)])
                            - vn[y*nx+x] * dt / dy * (vn[y*nx+x] - vn[(y-1)*nx+x])
                            - dt / (2.0 * rho * dy) * (p[(y+1)*nx+x] - p[(y-1)*nx+x])
                            + nu * dt / std::pow(dx, 2.0)
                            * (vn[y*nx+(x+1)] - 2.0 * vn[y*nx+x] + vn[y*nx+(x-1)])
                            + nu * dt / std::pow(dy, 2.0)
                            * (vn[(y+1)*nx+x] - 2.0 * vn[y*nx+x] + vn[(y-1)*nx+x]);
        }

        #pragma omp parallel for
        for (int x = 0; x < nx; ++x)    u[0*nx+x]        = 0.0;
        #pragma omp parallel for
        for (int y = 0; y < ny; ++y)    u[y*nx+0]        = 0.0;
        #pragma omp parallel for
        for (int y = 0; y < ny; ++y)    u[y*nx+(nx-1)]   = 0.0;
        #pragma omp parallel for
        for (int x = 0; x < nx; ++x)    u[(ny-1)*nx+x]   = 1.0;
        #pragma omp parallel for
        for (int x = 0; x < nx; ++x)    v[0*nx+x]        = 0.0;
        #pragma omp parallel for
        for (int x = 0; x < nx; ++x)    v[(ny-1)*nx+x]   = 0.0;
        #pragma omp parallel for
        for (int y = 0; y < ny; ++y)    v[y*nx+0]        = 0.0;
        #pragma omp parallel for
        for (int y = 0; y < ny; ++y)    v[y*nx+(nx-1)]   = 0.0;

    }
    
    return 0;
}