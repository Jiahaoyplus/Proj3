#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip> // For std::setprecision
#include <omp.h>   // Include OpenMP header


int main() {
    int size = 50;
    double J = 1.0;
    double cut_off = 0.025;

    int steps = 10000000;
    unsigned int seed = 42; // Example seed value
    int average_steps = 2000000; // Number of steps to average after burn-in



    std::ofstream results_file("curie_temperature_results.csv");
    results_file << "k_BT,Mean_Magnetization,chi,C_v\n"; // Header
    
    #pragma omp parallel for
    for (int k = 1; k <= int(4/cut_off); ++k) { // 0 to 40 corresponds to k_B T from 0.0 to 4.0 in increments of 0.1
        double k_B_T = k * cut_off;
        double beta = 1.0 / k_B_T;
        std::mt19937 gen(seed+k);
        // auto lattice = initializeLattice(size, gen);
        // std::vector<std::vector<int>> lattice(size, std::vector<int>(size));
        std::vector<int> lattice(size*size);
        std::uniform_int_distribution<> dis(0, 1);
        double energy = 0.0;
        double magnetization = 0.0;

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                lattice[i*size+j] = dis(gen) * 2 - 1; // Randomly assign +1 or -1
                // lattice[i*size+j] = 1;
            }
        }

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                double neighbors = lattice[((i + 1) % size) * size + j] + lattice[((i - 1 + size) % size) * size + j] +
                                lattice[i * size + ((j + 1) % size)] + lattice[i*size+((j - 1 + size) % size)];
                energy -= 0.5* J * lattice[i*size+j] * neighbors;
                magnetization += lattice[i*size+j]; // Sum up the spins for magnetization
            }
        }
        
        // calculateEnergyAndMagnetization(lattice, J, energy, magnetization);
        

        for (int step = 1; step <= steps; ++step) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::mt19937 gen2(rd());
            std::uniform_int_distribution<> dis(0, size - 1);
            std::uniform_real_distribution<> dis_real(0.0, 1.0);

            int i = dis(gen);
            int j = dis(gen2);

            double neighbors = lattice[((i + 1) % size )* size + j] + lattice[((i - 1 + size) % size) * size + j] +
                            lattice[i * size + ((j + 1) % size)] + lattice[i*size+((j - 1 + size) % size)];
            double deltaE = 2 * J * lattice[i*size+j] * neighbors;

            if (deltaE < 0 || dis_real(gen) < std::exp(-beta * deltaE)) {
                lattice[i*size+j] *= -1; // Flip the spin
                energy += deltaE; // Update energy
                magnetization += 2.0*lattice[i*size+j];
            }
        }

        // Calculate averages after burn-in
        double avg_energy = 0.0;
        double avg_energy_sq = 0.0;
        double avg_magnetization = 0.0;
        double avg_magnetization_sq = 0.0;
        for (int step = 0; step < average_steps; ++step) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::mt19937 gen2(rd());
            std::uniform_int_distribution<> dis(0, size - 1);
            std::uniform_real_distribution<> dis_real(0.0, 1.0);

            int i = dis(gen);
            int j = dis(gen2);

            double neighbors = lattice[(((i + 1) % size) * size) + j] + lattice[(((i - 1 + size) % size) * size) + j] +
                            lattice[i * size + ((j + 1) % size)] + lattice[i*size+((j - 1 + size) % size)];
            double deltaE = 2 * J * lattice[i*size+j] * neighbors;

            if (deltaE < 0 || dis_real(gen) < std::exp(-beta * deltaE)) {
                lattice[i*size+j] *= -1; // Flip the spin
                energy += deltaE; // Update energy
                magnetization += 2.0*lattice[i*size+j];
            }
            avg_energy += energy;
            avg_energy_sq += energy*energy;
            avg_magnetization += magnetization/(size*size);
            avg_magnetization_sq += magnetization*magnetization/(size*size*size*size);
        }


        avg_energy /= average_steps;
        avg_energy_sq /= average_steps;
        avg_magnetization /= average_steps;
        avg_magnetization_sq /= average_steps;


        double chi = beta * (avg_magnetization_sq - (avg_magnetization * avg_magnetization));
        double C_v = beta * beta * (avg_energy_sq - avg_energy * avg_energy);

        // Write the results to the file (use critical section to avoid race conditions)
        #pragma omp critical
        {
            results_file  << k_B_T << "," << avg_magnetization << "," << chi << "," << C_v << "\n";
        }
            
    }

    results_file.close();
    std::cout << "Simulation complete. Results written to curie_temperature_results.csv" << std::endl;

    return 0;
}