#pragma once
#ifndef heightMap_class_hpp
#define heightMap_class_hpp

#include <cmath>
#include <stdlib.h>
#include <set>
#include <vector>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

    std::random_device rd;  // Will be used to obtain a seed for the random number engine 
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis_01(0, 1);

class Generate_HeightMap{
    public:

        Generate_HeightMap(){
            //initialize the seed
        }
        
        void Steps(double stepHeight){


            double pixelSize_ = 0.02;  //tipycal pixel size is 0.026
            terrainProp_.xSize = 7.0;
            terrainProp_.ySize = 14.0; //ricorda che le dimensioni qui devono essere congruente.
            terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
            terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

            heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

            double stepSize = 0.7;
            double height = stepHeight;

            int xNum = terrainProp_.xSize / stepSize;  //quanti quadratoni ci sono lungo l'asse x
            int yNum = terrainProp_.ySize / stepSize;
            int gridWidth_ = stepSize / pixelSize_;  //quanti pixel ci sono in uno step. 

            Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),  //Eigen::Map prende un vettore (che sara' il primo argomento tra parentesi, e lo rimappa come una matrice di dimensione xsample*ysample)
                                                                terrainProp_.xSamples, //con data stai ritornando un puntatore a height. quindi le modifiche a mapMat tornano su height
                                                                terrainProp_.ySamples);
            mapMat.setZero();
            ///steps
            double h;
            for (size_t i = 0; i < xNum; i++) {
                for (size_t j = 0; j < yNum; j++) {
                    height = 0.65*stepHeight;

                    if( (i >= int(xNum/2)-2 && j >= int(yNum/2)-2) && (i <= int(xNum/2)+2 && j <= int(yNum/2)+2) )
                        height = 0.8*stepHeight;

                    h = dis_01(gen) * height;   
                    if(h > 0.2) 
                        h = 0.2;           
                    mapMat.block(gridWidth_ * i, gridWidth_ * j, gridWidth_, gridWidth_).setConstant(h);   //ogni campione xample, ysample e' un pixel. Quindi tu ti devi muovere di pixel in pixeo    
                }//questo codice dice, "parti da 0,0" e prendimi un blocco di dimensione gridWidth_*gridWidth_, quindi ti sta prendendo un intero step perche' gridWidth_*gridWidth_ e' la risoluzione di uno step
            }//poi si muove lungo le righe e le colonne, perche' si deve spostare di gridWidth_ pixel (cioe' gridWidth_ campioni, verso l'alto e il basso)
            //infine, con setconstant, definisce l'altezza di ogni step.

 
            mapMat.topRows(1).setConstant(0.03);  
            mapMat.leftCols(1).setConstant(0.03); 
            mapMat.bottomRows(1).setConstant(0.03);
            mapMat.rightCols(1).setConstant(0.03);

            distanceFromAnymal_ = terrainProp_.xSize/2;
        } 

        void Hills(double SinusoidalAmplitude){
            double pixelSize_ = 0.15;
            a = 17;
            terrainProp_.xSize = 12.0;
            terrainProp_.ySize = 24.0;
            terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
            terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

            heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

            terrainProp_.fractalOctaves = 12;
            terrainProp_.frequency = 1.1; ///
            terrainProp_.fractalLacunarity = 2.0;
            terrainProp_.fractalGain = 0.25;

            terrainGenerator_.getTerrainProp() = terrainProp_;
            terrainGenerator_.setSeed(10);

            heights_ = terrainGenerator_.generatePerlinFractalTerrain();

            double min_value = std::abs(*std::min_element(heights_.begin(), heights_.end()));

            for (size_t idx = 0; idx < heights_.size(); idx++) {
                heights_[idx] += (0.001) * dis_01(gen); //usando solo questo, il terrenno e' gia' abbastanza abbassato, ma escono fuori solo spuntoni
                heights_[idx] -= min_value; //muove il terreno generato, verso il basso in modo che il punto piu' basso del terreno tocchi terra        
            }
            
            Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                            terrainProp_.xSamples,
                                                            terrainProp_.ySamples);

            //La mappa e' generata, e quanto e' complicata lo puoi settare modificando i parametri di terrain_prop, pero' devi renderla smooth ai bordi.
            //quindi prendo i bordi della height map, e partendoli dal farli smooth, li faccio via via piu' 
            for(int i = 0; i <10 ; i++){ //mi muovo pixel per pixel (campione per campione) che corrispondo alle celle della matrixe mapMa
                mapMat.middleCols(i, 1) *= i*0.1; 
                mapMat.middleRows(i, 1) *= i*0.1; 
                mapMat.middleCols((mapMat.cols()-1)-i, 1) *= i*0.1; //ci sono 800 colonne, che vanno da 0 a 799
                mapMat.middleRows((mapMat.rows()-1)-i, 1) *= i*0.1; 
            }
            mapMat *= SinusoidalAmplitude; //lo uso per abbassare ancora di piu' il terreno. Abbassandolo, le ampiezze delle onde sinusoidali si avvicinano a 0, quindi il terreno diventa pie' regolare. Modulandolo tra 0 e 1 posso aggiornare la heightmap da una superficie piana a delle colline frastagliaet
            //se vale 1, allora il rumore di perlin e' praticamente quello che ho definito sopra. Se invece lo metto a 0.1, le ampiezze delle onde si abbassano (ma sono sempre maggiori di 0), e quindi il terreno diventa piu' flat
            distanceFromAnymal_ = terrainProp_.xSize/2;
        }

        void Slope(double slopes) {

            double pixelSize_ = 0.02;
            terrainProp_.xSize = 8.0;
            terrainProp_.ySize = 16.0;
            terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
            terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

            heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

            double Hills = 0.5;
            double dh = std::tan(Hills) * pixelSize_;

            Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                            terrainProp_.xSamples,
                                                            terrainProp_.ySamples);

            double height = 0;
            int j = 0;

            for (int i = 0; i < mapMat.rows(); i++) {
                if(i < int(mapMat.rows()/3))
                mapMat.middleRows(i,1).setConstant(i*slopes);
                if(i==int(mapMat.rows()/3))
                height = i*slopes;
                if(i >= int(mapMat.rows()/3) && i<int(2*mapMat.rows()/3))
                mapMat.middleRows(i,1).setConstant(height);
                if(i >= int(2*mapMat.rows()/3)){
                mapMat.middleRows(i,1).setConstant(height - j*slopes);
                j++;
                }
            }
            mapMat.rightCols(1).setConstant(0);
            mapMat.leftCols(1).setConstant(0);

            distanceFromAnymal_ = terrainProp_.xSize/2;
        } 

        void PyramidStairs(double pyramidHeight){ 

            double pixelSize_ = 0.02;
            terrainProp_.xSize = 10.0; //in meters
            terrainProp_.ySize = 20.0;
            terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
            terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

            heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

            Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                                terrainProp_.xSamples,  //rows
                                                                terrainProp_.ySamples); //cols
            double step_width = 0.35; //60cm
            double step_width_pixel = step_width/pixelSize_; //lo "converto" da lunghezza in metri a lunghezza in pixel perche' la matrice e' tale che ogni cella corrisponde a un pixel
            //l'idea e' che sulla matrice a blocchi, parto da (0,0) per estrarre il primo blocco, e poi mi muovo lungo x di steo_dith_pixel, e lungo y di step_height_pixel

            double num_step = 10;

            for(int i = 0; i < num_step; i++){
                //Queste 2 variabili sono la lunghezza e larghezza (dimensione lungo x e y) degli step. Sono dei quadrati, e sono enormi, come i blocchi che mettevi prima
                double current_step_dim_x = terrainProp_.xSize - i*2*step_width; //gli devo togliere l'ampiezza degli step precedenti
                double current_step_dim_y = terrainProp_.ySize - i*2*step_width;

                mapMat.block(i*step_width_pixel, i*step_width_pixel, current_step_dim_x/pixelSize_, current_step_dim_y/pixelSize_ ).setConstant(i*pyramidHeight+0.001);  //lo 0.0001 serve per non far coincidere il primo gradino con il terreno, altrimenti escono i punti bianchi che sono brutti

            }
            
            distanceFromAnymal_ = terrainProp_.xSize/2;
        }

        void Flat_with_friction(){

            double pixelSize_ = 0.02;
            terrainProp_.xSize = 16.0;
            terrainProp_.ySize = 16.0;
            terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
            terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

            heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);
            Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                            terrainProp_.xSamples,
                                                            terrainProp_.ySamples);

            mapMat.setConstant(0.08); //lo faccio praticamente un po' a gradino
            distanceFromAnymal_ = terrainProp_.xSize/2;
        }
    
        raisim::TerrainProperties getTerrainProp(){
            return terrainProp_;
        }

        std::vector<double> getHeights(){
            return heights_;
        }


    friend class Environment;    
    double distanceFromAnymal_ = 0; //mi serve a generare la height map vicinissimo la posizione dell'anymal


    private:
        std::vector<double> heights_;
        int a = 10;
        raisim::TerrainProperties terrainProp_;
        raisim::TerrainGenerator terrainGenerator_;  //only for the perlin noise

};


}

#endif