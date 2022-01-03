#include <cstdlib>
#include <random>
#include <iostream>
#include <fstream>
#include "contexte.hpp"
#include "individu.hpp"
#include "graphisme/src/SDL2/sdl2.hpp"
#include <chrono>
#include <mpi.h>
#include <omp.h>

void majStatistique(epidemie::Grille &grille, std::vector<epidemie::Individu> const &individus)
{
    #pragma omp parallel for num_threads(2)
    for (auto &statistique : grille.getStatistiques())
    {
        statistique.nombre_contaminant_grippe_et_contamine_par_agent = 0;
        statistique.nombre_contaminant_seulement_contamine_par_agent = 0;
        statistique.nombre_contaminant_seulement_grippe = 0;
    }
    auto [largeur, hauteur] = grille.dimension();
    auto &statistiques = grille.getStatistiques();
    
    #pragma omp parallel for num_threads(2)
    for (auto const &personne : individus)
    {
        auto pos = personne.position();

        std::size_t index = pos.x + pos.y * largeur;
        if (personne.aGrippeContagieuse())
        {
            if (personne.aAgentPathogeneContagieux())
            {
                statistiques[index].nombre_contaminant_grippe_et_contamine_par_agent += 1;
            }
            else
            {
                statistiques[index].nombre_contaminant_seulement_grippe += 1;
            }
        }
        else
        {
            if (personne.aAgentPathogeneContagieux())
            {
                statistiques[index].nombre_contaminant_seulement_contamine_par_agent += 1;
            }
        }
    }
}

void afficheSimulation(sdl2::window &ecran, epidemie::Grille const &grille, std::size_t jour)
{
    auto [largeur_grille, hauteur_grille] = grille.dimension();
    auto [largeur_ecran, hauteur_ecran] = ecran.dimensions();
    auto &statistiques = grille.getStatistiques();
    sdl2::font fonte_texte("./graphisme/src/data/Lato-Thin.ttf", 18);
    ecran.cls({0x00, 0x00, 0x00});
    // Affichage de la grille :
    std::uint16_t stepX = largeur_ecran / largeur_grille;
    unsigned short stepY = (hauteur_ecran - 50) / hauteur_grille;
    double factor = 255. / 15.;

    for (unsigned short i = 0; i < largeur_grille; ++i)
    {

        for (unsigned short j = 0; j < hauteur_grille; ++j)
        {
            auto const &stat = statistiques[i + j * largeur_grille];
            int valueGrippe = stat.nombre_contaminant_grippe_et_contamine_par_agent + stat.nombre_contaminant_seulement_grippe;
            int valueAgent = stat.nombre_contaminant_grippe_et_contamine_par_agent + stat.nombre_contaminant_seulement_contamine_par_agent;
            std::uint16_t origx = i * stepX;
            std::uint16_t origy = j * stepY;
            std::uint8_t red = valueGrippe > 0 ? 127 + std::uint8_t(std::min(128., 0.5 * factor * valueGrippe)) : 0;
            std::uint8_t green = std::uint8_t(std::min(255., factor * valueAgent));
            std::uint8_t blue = std::uint8_t(std::min(255., factor * valueAgent));
            ecran << sdl2::rectangle({origx, origy}, {stepX, stepY}, {red, green, blue}, true);
        }
    }

    ecran << sdl2::texte("Carte population grippee", fonte_texte, ecran, {0xFF, 0xFF, 0xFF, 0xFF}).at(largeur_ecran / 2, hauteur_ecran - 20);
    ecran << sdl2::texte(std::string("Jour : ") + std::to_string(jour), fonte_texte, ecran, {0xFF, 0xFF, 0xFF, 0xFF}).at(0, hauteur_ecran - 20);
    ecran << sdl2::flush;
}

void simulation(bool affiche, int nargs, char *argv[])
{
    MPI_Init(&nargs, &argv);
    MPI_Comm globComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &globComm);
    int nbp;
    MPI_Comm_size(globComm, &nbp);
    int rank;
    MPI_Comm_rank(globComm, &rank);
    MPI_Status Stat;
    MPI_Datatype stat_point;
    MPI_Type_contiguous(3, MPI_INT, &stat_point);
    MPI_Type_commit(&stat_point);
    MPI_Request send_request_quit, rcv_request_quit, send_request1, send_request2;
    // int flag = 0;
    // initialisation des variables selon le mode
    MPI_Comm subComm;
    int colour = rank != 0 ? 0 : 1;
    MPI_Comm_split(globComm, colour, rank, &subComm);

    epidemie::ContexteGlobal contexte;
    // contexte.deplacement_maximal = 1; <= Si on veut moins de brassage
    // contexte.taux_population = 400'000;
    // contexte.taux_population = 1'000;
    contexte.interactions.beta = 60.;
    constexpr const unsigned int largeur_ecran = 1280, hauteur_ecran = 1024;
    epidemie::Grille grille{contexte.taux_population};
    auto [largeur_grille, hauteur_grille] = grille.dimension();

    std::size_t jours_ecoules = 0;

    int quitting = 0;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    auto &rcv_buffer = grille.getStatistiques();
    if (rank == 0)
    {

        sdl2::event_queue queue;
        sdl2::window ecran("Simulation epidemie de grippe", {largeur_ecran, hauteur_ecran});

        while (quitting == 0)
        {
            //#############################################################################################################
            //##    Affichage des resultats pour le temps  actuel
            //#############################################################################################################
            start = std::chrono::system_clock::now();

            auto events = queue.pull_events();
            for (const auto &e : events)
            {
                if (e->kind_of_event() == sdl2::event::quit)
                {
                    quitting = 1;
                }
            }

            MPI_Isend(&quitting, 1, MPI_INT, 1, 100, globComm, &send_request_quit);

            MPI_Recv(rcv_buffer.data(), hauteur_grille * largeur_grille, stat_point, 1, 0, globComm, &Stat);
            MPI_Recv(&jours_ecoules, 1, MPI_INT, 1, 1, globComm, &Stat);

            grille.Set_statistiques(rcv_buffer);
            afficheSimulation(ecran, grille, jours_ecoules);
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds_aff = end - start;
            // std::cout << "temps affichage : " << elapsed_seconds_aff.count() <<std::endl;
            /*std::cout << jours_ecoules << "\t" << grille.nombreTotalContaminesGrippe() << "\t"
                    << grille.nombreTotalContaminesAgentPathogene() << std::endl;*/

        } // Fin boucle temporelle
    }

    if (rank >= 1)
    {
        nbp--;
        unsigned int graine_aleatoire = 1;
        std::uniform_real_distribution<double> porteur_pathogene(0., 1.);
        size_t k = contexte.taux_population / nbp;
        size_t pop_per_proc = rank == nbp - 1 ? contexte.taux_population - (nbp - 1) * k : k;

        float temps = 0;
        std::vector<epidemie::Individu> population;
        population.reserve(pop_per_proc);

        // L'agent pathogene n'evolue pas et reste donc constant...
        epidemie::AgentPathogene agent(graine_aleatoire++);
        // Initialisation de la population initiale :
        graine_aleatoire += (rank - 1) * k;
        for (std::size_t i = 0; i < pop_per_proc; ++i)
        {
            std::default_random_engine motor(100 * (i + 1));
            population.emplace_back(graine_aleatoire++, contexte.esperance_de_vie, contexte.deplacement_maximal);
            population.back().setPosition(largeur_grille, hauteur_grille);
            if (porteur_pathogene(motor) < 0.2)
            {
                population.back().estContamine(agent);
            }
        }

        int flag = 0;
        int jour_apparition_grippe = 0;

        int k2 = (contexte.taux_population * 23) / 100;

        int nombre_immunises_grippe = rank == nbp - 1 ? k2 - (nbp - 1) * (pop_per_proc * 23) / 100 : (pop_per_proc * 23) / 100; // work even if 23*total_pop can't be divided by 100

        epidemie::Grippe grippe(0);
        std::ofstream output("Courbe.dat");
        if (rank == 1)
        {
            std::cout << "Debut boucle epidemie" << std::endl
                      << std::flush;
        }
        while (quitting == 0)
        {
            start = std::chrono::system_clock::now();
            if (jours_ecoules % 365 == 0) // Si le premier Octobre (debut de l'annee pour l'epidemie ;-) )
            {

                grippe = epidemie::Grippe(jours_ecoules / 365);
                jour_apparition_grippe = grippe.dateCalculImportationGrippe();
                grippe.calculNouveauTauxTransmission();
// 23% des gens sont immunises. On prend les 23% premiers
#pragma omp parallel for num_threads(2)
                for (int ipersonne = 0; ipersonne < nombre_immunises_grippe; ++ipersonne)
                {
                    population[ipersonne].devientImmuniseGrippe();
                }
#pragma omp parallel for num_threads(2)
                for (int ipersonne = nombre_immunises_grippe; ipersonne < int(pop_per_proc); ++ipersonne)
                {
                    population[ipersonne].redevientSensibleGrippe();
                }
            }
            int n_contaminations = rank == (nbp - 1) ? 25 - (nbp - 1) * 25 / nbp : 25 / nbp;
            if (jours_ecoules % 365 == std::size_t(jour_apparition_grippe))
            {
#pragma omp parallel for num_threads(2)
                for (int ipersonne = nombre_immunises_grippe; ipersonne < nombre_immunises_grippe + n_contaminations; ++ipersonne)
                {
                    population[ipersonne].estContamine(grippe);
                }
            }
            // Mise a jour des statistiques pour les cases de la grille :
            majStatistique(grille, population);
            auto &buffer = grille.getStatistiques();
            auto rcv_buffer = grille.getStatistiques();

            MPI_Allreduce(buffer.data(), rcv_buffer.data(), largeur_grille * hauteur_grille * 3, MPI_INT, MPI_SUM, subComm);
            grille.Set_statistiques(rcv_buffer);

            // On parcout la population pour voir qui est contamine et qui ne l'est pas, d'abord pour la grippe puis pour l'agent pathogene
            std::size_t compteur_grippe = 0, compteur_agent = 0, mouru = 0;
#pragma omp parallel for num_threads(2)

            for (auto &personne : population)
            {
                if (personne.testContaminationGrippe(grille, contexte.interactions, grippe, agent))
                {
                    compteur_grippe++;
                    personne.estContamine(grippe);
                }
                if (personne.testContaminationAgent(grille, agent))
                {
                    compteur_agent++;
                    personne.estContamine(agent);
                }
                // On verifie si il n'y a pas de personne qui veillissent de veillesse et on genere une nouvelle personne si c'est le cas.
                if (personne.doitMourir())
                {
                    mouru++;
                    unsigned nouvelle_graine = jours_ecoules + personne.position().x * personne.position().y;
                    personne = epidemie::Individu(nouvelle_graine, contexte.esperance_de_vie, contexte.deplacement_maximal);
                    personne.setPosition(largeur_grille, hauteur_grille);
                }
                personne.veillirDUnJour();
                personne.seDeplace(grille);
            }
            jours_ecoules += 1;
            if (rank == 1)
            {

                if (jours_ecoules == 1)
                    output << "# jours_ecoules \t nombreTotalContaminesGrippe \t nombreTotalContaminesAgentPathogene()" << std::endl;
                output << jours_ecoules << "\t" << grille.nombreTotalContaminesGrippe() << "\t"
                       << grille.nombreTotalContaminesAgentPathogene() << std::endl;

                auto &buffer = grille.getStatistiques();

                MPI_Iprobe(0, 100, globComm, &flag, &Stat);

                if (flag == 1)
                {
                    MPI_Irecv(&quitting, 1, MPI_INT, 0, 100, globComm, &rcv_request_quit);
                    MPI_Isend(buffer.data(), largeur_grille * hauteur_grille, stat_point, 0, 0, globComm, &send_request1);
                    MPI_Isend(&jours_ecoules, 1, MPI_INT, 0, 1, globComm, &send_request2);
                    MPI_Wait(&send_request1, &Stat); // obligatoire pour avoir un affichage coherent
                    flag = false;
                }
            }
            MPI_Bcast(&quitting, 1, MPI_INT, 1, globComm);
            end = std::chrono::system_clock::now();
            if (rank == 1)
            {
                std::chrono::duration<double> elapsed_seconds_calc = end - start;
                temps += elapsed_seconds_calc.count();
            }
        }
        if (rank == 1)
            std::cout << "temps_moyen execution : " << temps / jours_ecoules << std::endl;
        output.close();
    }

    MPI_Finalize();
}

int main(int argc, char *argv[])
{
    // parse command-line
    bool affiche = true;
    for (int i = 0; i < argc; i++)
    {
        std::cout << i << " " << argv[i] << "\n";
        if (std::string("-nw") == argv[i])
            affiche = false;
    }

    sdl2::init(argc, argv);
    {
        simulation(affiche, argc, argv);
    }
    sdl2::finalize();
    return EXIT_SUCCESS;
}
