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

void màjStatistique( épidémie::Grille& grille, std::vector<épidémie::Individu> const& individus )
{
    #pragma omp parallel for num_threads(2)
    for (auto &statistique : grille.getStatistiques())
    {
        statistique.nombre_contaminant_grippé_et_contaminé_par_agent = 0;
        statistique.nombre_contaminant_seulement_contaminé_par_agent = 0;
        statistique.nombre_contaminant_seulement_grippé = 0;
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
            if (personne.aAgentPathogèneContagieux())
            {
                statistiques[index].nombre_contaminant_grippé_et_contaminé_par_agent += 1;
            }
            else
            {
                statistiques[index].nombre_contaminant_seulement_grippé += 1;
            }
        }
        else
        {
            if (personne.aAgentPathogèneContagieux())
            {
                statistiques[index].nombre_contaminant_seulement_contaminé_par_agent += 1;
            }
        }
    }
}

void afficheSimulation(sdl2::window& écran, épidémie::Grille const& grille, std::size_t jour)
{
    auto [largeur_écran,hauteur_écran] = écran.dimensions();
    auto [largeur_grille,hauteur_grille] = grille.dimension();
    auto const& statistiques = grille.getStatistiques();
    sdl2::font fonte_texte("./graphisme/src/data/Lato-Thin.ttf", 18);
    écran.cls({0x00,0x00,0x00});
    // Affichage de la grille :
    std::uint16_t stepX = largeur_écran/largeur_grille;
    unsigned short stepY = (hauteur_écran-50)/hauteur_grille;
    double factor = 255./15.;

    for ( unsigned short i = 0; i < largeur_grille; ++i )
    {
        for (unsigned short j = 0; j < hauteur_grille; ++j )
        {
            auto const& stat = statistiques[i+j*largeur_grille];
            int valueGrippe = stat.nombre_contaminant_grippé_et_contaminé_par_agent+stat.nombre_contaminant_seulement_grippé;
            int valueAgent  = stat.nombre_contaminant_grippé_et_contaminé_par_agent+stat.nombre_contaminant_seulement_contaminé_par_agent;
            std::uint16_t origx = i*stepX;
            std::uint16_t origy = j*stepY;
            std::uint8_t red = valueGrippe > 0 ? 127+std::uint8_t(std::min(128., 0.5*factor*valueGrippe)) : 0;
            std::uint8_t green = std::uint8_t(std::min(255., factor*valueAgent));
            std::uint8_t blue= std::uint8_t(std::min(255., factor*valueAgent ));
            écran << sdl2::rectangle({origx,origy}, {stepX,stepY}, {red, green,blue}, true);
        }
    }

    écran << sdl2::texte("Carte population grippée", fonte_texte, écran, {0xFF,0xFF,0xFF,0xFF}).at(largeur_écran/2, hauteur_écran-20);
    écran << sdl2::texte(std::string("Jour : ") + std::to_string(jour), fonte_texte, écran, {0xFF,0xFF,0xFF,0xFF}).at(0,hauteur_écran-20);
    écran << sdl2::flush;
}

void simulation(bool affiche, int nargs, char *argv[])
{
    MPI_Init(&nargs, &argv);
    MPI_Comm globComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &globComm);
    int nbp;
    int rank;
    MPI_Comm_size(globComm, &nbp);
    MPI_Comm_rank(globComm, &rank);
    MPI_Status Stat;
    MPI_Datatype Stat_Type;
    MPI_Type_contiguous(3, MPI_INT, &Stat_Type);
    MPI_Type_commit(&Stat_Type);
    MPI_Request Request1, Request2, Request3, Request4;
    
    
    int flag = 0;

    constexpr const unsigned int largeur_écran = 1280, hauteur_écran = 1024;
    épidémie::ContexteGlobal contexte;
    // contexte.déplacement_maximal = 1; <= Si on veut moins de brassage
    // contexte.taux_population = 400'000;
    //contexte.taux_population = 1'000;
    contexte.interactions.β = 60.;
    épidémie::Grille grille{contexte.taux_population};
    auto [largeur_grille, hauteur_grille] = grille.dimension();

    std::size_t jours_écoulés = 0;

    bool quitting = false;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    auto &rcv_buffer = grille.getStatistiques();
    if (rank == 0)
    {

        sdl2::event_queue queue;
        sdl2::window écran("Simulation épidémie de grippe", {largeur_écran,hauteur_écran});
        float Temps_Global = 0;

        while (!quitting)
        {
            //#############################################################################################################
            //##    Affichage des resultats pour le Temps_Global  actuel
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

            MPI_Isend(&quitting, 1, MPI_INT, 1, 100, globComm, &Request1);

            MPI_Recv(rcv_buffer.data(), hauteur_grille * largeur_grille, Stat_Type, 1, 0, globComm, &Stat);
            MPI_Recv(&jours_écoulés, 1, MPI_INT, 1, 1, globComm, &Stat);

            grille.setStatistiques(rcv_buffer);
            afficheSimulation(écran, grille, jours_écoulés);
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds_aff = end - start;
            Temps_Global += elapsed_seconds_aff.count();
        } 
        std::cout << "Temps affichage moyen : " << Temps_Global / jours_écoulés << std::endl;// Fin boucle temporelle
    }

    if (rank == 1)
    {
        unsigned int graine_aléatoire = 1;
        std::uniform_real_distribution<double> porteur_pathogène(0., 1.);
        float Temps_Global = 0;
        std::vector<épidémie::Individu> population;
        population.reserve(contexte.taux_population);

        // L'agent pathogene n'evolue pas et reste donc constant...
        épidémie::AgentPathogène agent(graine_aléatoire++);
        // Initialisation de la population initiale :

        for (std::size_t i = 0; i <contexte.taux_population; ++i)
        {
            std::default_random_engine motor(100*(i+1));
            population.emplace_back(graine_aléatoire++, contexte.espérance_de_vie, contexte.déplacement_maximal);
            population.back().setPosition(largeur_grille, hauteur_grille);
            if (porteur_pathogène(motor) < 0.2)
            {
                population.back().estContaminé(agent);   
            }
        }

        
        int jour_apparition_grippe = 0;
        int nombre_immunisés_grippe= (contexte.taux_population*23)/100;

        épidémie::Grippe grippe(0);
        std::ofstream output("Courbe.dat");
        if (rank == 1)
        {
            std::cout << "Debut boucle epidemie" << std::endl
                      << std::flush;
        }
        while (!quitting)
        {
            start = std::chrono::system_clock::now();
            if (jours_écoulés % 365 == 0) // Si le premier Octobre (debut de l'annee pour l'epidemie ;-) )
            {

                grippe = épidémie::Grippe(jours_écoulés / 365);
                jour_apparition_grippe = grippe.dateCalculImportationGrippe();
                grippe.calculNouveauTauxTransmission();

                #pragma omp parallel for num_threads(2)
                for (int ipersonne = 0; ipersonne < nombre_immunisés_grippe; ++ipersonne)
                {
                    population[ipersonne].devientImmuniséGrippe();
                }
                #pragma omp parallel for num_threads(2)
                for (int ipersonne = nombre_immunisés_grippe; ipersonne < int(contexte.taux_population); ++ipersonne)
                {
                    population[ipersonne].redevientSensibleGrippe();
                }
            }
            if (jours_écoulés%365 == std::size_t(jour_apparition_grippe))
            {
                #pragma omp parallel for num_threads(2)
                for (int ipersonne = nombre_immunisés_grippe; ipersonne < nombre_immunisés_grippe + 25; ++ipersonne )
                {
                    population[ipersonne].estContaminé(grippe);
                }
            }
            // Mise a jour des statistiques pour les cases de la grille :
            màjStatistique(grille, population);
            auto rcv_buffer = grille.getStatistiques();

            grille.setStatistiques(rcv_buffer);

            // On parcout la population pour voir qui est contamine et qui ne l'est pas, d'abord pour la grippe puis pour l'agent pathogene
            std::size_t compteur_grippe = 0, compteur_agent = 0, mouru = 0;

            #pragma omp parallel for num_threads(2)
            for (auto &personne : population)
            {
                if (personne.testContaminationGrippe(grille, contexte.interactions, grippe, agent))
                {
                    compteur_grippe++;
                    personne.estContaminé(grippe);
                }
                if (personne.testContaminationAgent(grille, agent))
                {
                    compteur_agent++;
                    personne.estContaminé(agent);
                }
                // On verifie si il n'y a pas de personne qui veillissent de veillesse et on genere une nouvelle personne si c'est le cas.
                if (personne.doitMourir())
                {
                    mouru++;
                    unsigned nouvelle_graine = jours_écoulés + personne.position().x * personne.position().y;
                     personne = épidémie::Individu(nouvelle_graine, contexte.espérance_de_vie, contexte.déplacement_maximal);
                    personne.setPosition(largeur_grille, hauteur_grille);
                }
                personne.veillirDUnJour();
                personne.seDéplace(grille);
            }
            jours_écoulés += 1;
            if (rank == 1)
            {

                if (jours_écoulés == 1)
                    output << "# jours_ecoules \t nombreTotalContaminesGrippe \t nombreTotalContaminesAgentPathogene()" << std::endl;
                output << jours_écoulés << "\t" << grille.nombreTotalContaminésGrippe() << "\t"
                       << grille.nombreTotalContaminésAgentPathogène() << std::endl;

                auto &buffer = grille.getStatistiques();

                MPI_Iprobe(0, 100, globComm, &flag, &Stat);

                if (flag == 1)
                {
                    MPI_Irecv(&quitting, 1, MPI_INT, 0, 100, globComm, &Request2);
                    MPI_Isend(buffer.data(), largeur_grille * hauteur_grille, Stat_Type, 0, 0, globComm, &Request3);
                    MPI_Isend(&jours_écoulés, 1, MPI_INT, 0, 1, globComm, &Request4);
                    MPI_Wait(&Request3, &Stat);
                    flag = false;
                }
            }
            end = std::chrono::system_clock::now();
            if (rank == 1)
            {
                std::chrono::duration<double> elapsed_seconds_calc = end - start;
                Temps_Global += elapsed_seconds_calc.count();
            }
        }
        if (rank == 1)
            std::cout << "Temps execution moyen :" << Temps_Global / jours_écoulés << std::endl;
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
