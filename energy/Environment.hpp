//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <cmath>
#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include <vector>
#include <thread>
#include <chrono>
#include <boost/algorithm/string.hpp>

namespace raisim {


enum TerrainType {
  Flat_,
  Steps,
  Hills,
  Slope,
  PyramidStairs
};

  std::random_device rd;  // Will be used to obtain a seed for the random number engine 
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {


    raisim::World::setActivationKey("~/.raisim"); //path of the folder in which i have the activation key
    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    //anymal_ = world_->addArticulatedSystem(resourceDir_+"/anymal/urdf/anymal.urdf");
    anymal_ = world_->addArticulatedSystem("/home/claudio/raisim_ws/raisimlib/rsc/anymal/urdf/anymal.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround(0.0, "ground");  //I have to give the name in order to set later the friction between the ground and the feet


    //Setto la friction
    anymal_->getCollisionBody("LF_FOOT/0").setMaterial(footNames_[0]);  //metto il nome al collision body. Ora questo nome, posso usarlo in setMaterialPairProp per dire che c'è attrito tra il terreno e quel corpo
    anymal_->getCollisionBody("RF_FOOT/0").setMaterial(footNames_[1]);
    anymal_->getCollisionBody("LH_FOOT/0").setMaterial(footNames_[2]);
    anymal_->getCollisionBody("RH_FOOT/0").setMaterial(footNames_[3]);
    for (int i = 0; i < 4; i++) {
      world_->setMaterialPairProp("ground", footNames_[i], 0.9, 0.0, 0.0);  //gli devo passare 2 nomi, il primo e' quello del terreno, il secondo e' quello dei piedi che gli ho passato con .setMaterial
    }

    terrainType_ = TerrainType::Flat_;
 
    READ_YAML(double, curriculumFactor_, cfg["curriculum"]["curriculumFactor_"]);
    READ_YAML(double, curriculumDecayFactor_, cfg["curriculum"]["curriculumDecayFactor_"]);
  
    //using t_value = typename std::iterator_traits<Iter>::value_type;
    //std::vector<t_value> temp(size);
   
    /// get robot data

    READ_YAML(int, num_seq, cfg_["num_seq"]);

    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;
    /// initialize containers
    current_action_.setZero(12);
    previous_action_.setZero(12);
    previous_gv_.setZero(12);

    joint_history_pos_.setZero(num_legs_joint*num_seq);
    joint_history_vel_.setZero(num_legs_joint*num_seq);

    joint_container_.setZero(24);
    joint_history_.setZero(joint_container_.size()*num_seq); //3 timestep to save the joint history.
    q1.setZero(num_seq), q2.setZero(num_seq), q3.setZero(num_seq), q4.setZero(num_seq), q5.setZero(num_seq), q6.setZero(num_seq);
    q7.setZero(num_seq), q8.setZero(num_seq), q9.setZero(num_seq), q10.setZero(num_seq), q11.setZero(num_seq), q12.setZero(num_seq);

    standing_configuration_.setZero(12);
    actual_joint_position_.setZero(12);
    actual_joint_velocities_.setZero(12);
    command_<< 1, 0.2, 0;
    
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    ga_.setZero(nJoints_); ga_init_.setZero(nJoints_);

    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
    standing_configuration_ = gc_init_.tail(12);

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
   

    if(&cfg["action_in_observation_space"]){ //because we don't use the READ_YAML we don't need underscore
      obDim_ = 34 + nJoints_;
      action_in_observation_space_ = true;
      if(&cfg["joint_history_in_observation_space"]){
        obDim_ = obDim_ + joint_history_.size();
        joint_history_in_observation_space_ = true;
      }
    }else
      obDim_ = 34;

    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config. Use the element with _ in READ_YAML otherwise use it without underscore
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);
    //std::cout<<"Reward from yaml: "<<cfg["reward"]<<std::endl;

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK")); //indice 3
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK")); //indice 6
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK")); //indice 9
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK")); //indice 12
    baseIndex_ = anymal_->getBodyIdx("base"); //index of the body
    std::cout<<baseIndex_;

    baseFrameIdx = anymal_->getFrameIdxByLinkName("LF_SHANK");
    //std::cout<<"Anymal position: "<<.e()<<std::endl;
    
    
    footLinkFrame_.insert(anymal_->getFrameIdxByLinkName("LF_FOOT"));  //l'indice del frame e' sempre un numero che ci dice se quel frame e' riferito a qualcosa o no
    footLinkFrame_.insert(anymal_->getFrameIdxByLinkName("RF_FOOT"));
    footLinkFrame_.insert(anymal_->getFrameIdxByLinkName("LH_FOOT"));
    footLinkFrame_.insert(anymal_->getFrameIdxByLinkName("RH_FOOT"));



    /// visualize if it is the first environment
   if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(anymal_);
    }
  }

  void init() final { }

  void reset() final {
    
    anymal_->setState(gc_init_, gv_init_);
    updateObservation(); //when we reset, we don't mind of the previous action

    anymal_->getFramePosition(baseFrameIdx, anymal_pos_);

    std::uniform_int_distribution<> dist4(0,3); // distribution in range [0, 3]
    std::vector<std::string> terrains = {"steps", "Hills", "Slope", "Stairs"};

    /*if(num_episode_ >=500){
      generate_HeightMap(terrains[dist4(gen)], stepHeight_, SinusoidalAmplitude_, slope_, pyramidHeights_);
    }*/

  }

  void command_vel(double v_x, double v_y, double omega_z){ //This function can be called to declare a new command velocity
    command_[0] = v_x;
    command_[1] = v_y;
    command_[2] = omega_z;
    std::cout<<"command_vel: "<<command_<<std::endl;
  }
  
  void fill_vector_of_current_velocities(double vx, double vy, double omega){
    v_x << vx;
   
    v_y << vy;
   
    omega_z << omega;  
  }

  float step(const Eigen::Ref<EigenVec>& action) {
    /// action scaling

    pTarget12_ = action.cast<double>();
    current_action_ = pTarget12_;  //I put the last action in the observation space

    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }
    
    updateJointHistory(); 
    if(action_in_observation_space_)
      updateObservation(current_action_);
    else
      updateObservation();

    hipPenalty();
    footSlipage(); //Check the value of c_F during the action
    error_velocity();

    //rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
    rewards_.record("energy", std::pow(anymal_->getGeneralizedForce().e().dot(gv_.tail(12)), 2)); //in Eigen the dot product between v and w is v.dot(w)
    rewards_.record("forwardVel_x", std::pow(1.8-bodyLinearVel_[0], 2));
    rewards_.record("forwardVel_y", std::pow(bodyLinearVel_[1], 2));
    rewards_.record("angularVel", std::pow(bodyAngularVel_[2], 2)); //must be penalized
    /*rewards_.record("forwardVel_x", forwardVel_x_);]

    rewards_.record("forwardVel_y", forwardVel_y_);
    rewards_.record("angularVel", angularVel_);
    rewards_.record("orthogonal_vel", v_o_exp_);
    //rewards_.record("Joint_velocity", gv_.tail(12).norm());
    //rewards_.record("Smooth_action", (current_action_ - previous_action_).norm()); //penalize big distance between 2 actions*/
    //rewards_.record("number_of_contact", numberOfContact_); weight_update_
    rewards_.record("omega_x", std::pow(bodyAngularVel_[0], 2));
    rewards_.record("omega_y", std::pow(bodyAngularVel_[1], 2));
    rewards_.record("v_z", std::pow(bodyLinearVel_[2], 2));
    //rewards_.record("hip_penalty", hipTerm_); //The more the error is bigger, the more penalize this term
    rewards_.record("slippage", slip_term_);
    rewards_.record("clearence", clearence_);
    //rewards_.record("air_time", sum_air_time(air_time_LF_, air_time_RF_, air_time_LH_, air_time_RH_));

    previous_action_ = current_action_;  //Save the current action to compute the next reward tem
    return rewards_.sum();

  }

  /*float step(const Eigen::Ref<EigenVec>& action) final{
    //std::thread thread_step (step_rl_algorithm, &action);     // spawn new thread that calls foo()
    std::thread thread_phi(raisim::ENVIRONMENT::cacca);  // spawn new thread that calls bar(0)

    // synchronize threads:
    //thread_step.join();                // pauses until first finishes
    thread_phi.join();               // pauses until second finishes

    return 7;
  }*/


  
  void error_velocity(){
    
    double norm_v_xy = std::sqrt(std::pow(bodyLinearVel_[0], 2) + std::pow(bodyLinearVel_[1], 2)); //|v|^2
    
    forwardVel_x_ = std::exp(- std::pow((command_[0] - bodyLinearVel_[0]), 2)/0.25);
    forwardVel_y_ = std::exp(- std::pow((command_[1] - bodyLinearVel_[1]), 2)/0.25);
    angularVel_ = std::exp(- std::pow((command_[2] - bodyAngularVel_[2]),2)/0.25);

    double v_proj_on_vdes = bodyLinearVel_[0]*command_[0] + bodyLinearVel_[1]*command_[1]; //lo scalare che rappresenta v proiettato su v_des
    
    Eigen::VectorXd v_des_norm (2);
    double norm_v_des_xy = std::sqrt(std::pow(command_[0], 2) + std::pow(command_[1], 2)); //|v|^2 //|v_des_xy|^2
    v_des_norm << command_[0]/norm_v_des_xy, command_[1]/norm_v_des_xy;
   
    Eigen::VectorXd v_o (2);
    v_o << (bodyLinearVel_[0] - v_proj_on_vdes*v_des_norm[0]), (bodyLinearVel_[1] - v_proj_on_vdes*v_des_norm[1]);
    double norm_v_o = std::sqrt(std::pow(v_o[0], 2) + std::pow(v_o[1], 2));
    v_o_exp_ = std::exp(-std::pow(norm_v_o, 2)/0.25); //quanto piu' v_o cresce, tanto piu' la ricompensa e' piccola. Dovra' cercare di tenere piccolo sto termine
  }

  void updateObservation(Eigen::VectorXd current_action_) {
    
    if(anymal_)
      anymal_->getState(gc_, gv_);  //Update the value of the joints. 

    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    if(joint_history_in_observation_space_){
      obDouble_ << gc_[2], /// body height
        rot.e().row(2).transpose(), /// body orientation
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity  
        current_action_, //last action
        joint_history_pos_,
        joint_history_vel_,
        command_;  ///Ci mette molto poco ad addestrarsi con questo nuovo termine
    }else
      obDouble_ << gc_[2], /// body height
          rot.e().row(2).transpose(), /// body orientation
          gc_.tail(12), /// joint angles
          bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
          gv_.tail(12), /// joint velocity
          current_action_;

    joint_container_ << gc_.tail(12), gv_.tail(12); //over write the content: 

    actual_joint_position_ = joint_container_(Eigen::seq(0,11));
    actual_joint_velocities_ = joint_container_(Eigen::seq(12,23));

    previous_gv_ = gv_.tail(12);
  }


  void updateObservation() {

    anymal_->getState(gc_, gv_);  //Update the value of the joints. 

    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
  
    obDouble_ << gc_[2], /// body height
          rot.e().row(2).transpose(), /// body orientation
          gc_.tail(12), /// joint angles
          bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
          gv_.tail(12); /// joint velocity
  }



  void updateJointHistory(){
      
    
    Eigen::VectorXd temp_pos (12*num_seq);
    temp_pos << joint_history_pos_; //temp conterrà nelle posizioni 0-11 quello che joints_history conterra' nelle posizioni 12-23
    
    Eigen::VectorXd temp_vel (12*num_seq);
    temp_vel << joint_history_vel_;

    for(int i = 0; i < (num_seq-1); i++){
      joint_history_pos_(Eigen::seq((i+1)*12, (i+2)*12-1)) = temp_pos(Eigen::seq(i*12, (i+1)*12-1)); //overwrite the next sequence

      joint_history_vel_(Eigen::seq(i*12, (i+1)*12-1)) = temp_vel(Eigen::seq((i+1)*12, (i+2)*12-1));
    }

    joint_history_pos_.head(12) = gc_.tail(12);
    joint_history_vel_.head(12) = gv_.tail(12);

    
    for(int i=0; i<num_seq-1; i++){
      q1<<joint_history_pos_[i*12];
      q2<<joint_history_pos_[i*12+1];
      q3<<joint_history_pos_[i*12+2];
      q4<<joint_history_pos_[i*12+3];
      q5<<joint_history_pos_[i*12+4];
      q6<<joint_history_pos_[i*12+5];
      q7<<joint_history_pos_[i*12+6];
      q8<<joint_history_pos_[i*12+7];
      q9<<joint_history_pos_[i*12+8];
      q10<<joint_history_pos_[i*12+9];
      q11<<joint_history_pos_[i*12+10];
      q12<<joint_history_pos_[i*12+11];
    }
  }
  

  void hipPenalty(){
    hipTerm_ = 0.0; //Altrimenti episodio dopo episodio sto termine cresce sempre.
    thighTerm_ = 0.0;

    for(int i=0; i<4; i++){
      hipTerm_ += std::pow(actual_joint_position_(i*3) - standing_configuration_(i*3), 2); //il +1 e' necessario perche' il primo giunto e' quello della base
      thighTerm_ += std::pow(actual_joint_position_(i*3 + 1) - standing_configuration_(i*3 + 1), 2);
    }
    /*Gli indici dei giunti di hip perche' nella cartella raisimExample ho usato il metodo che mi ritorna il nome dei giunti. Il file e' Anymal_numberOfJoints.cpp */
  }

  void generate_command_velocity(){ 
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.1, 0.4);
    command_[0] = dis(gen);
    std::uniform_real_distribution<> dis1(-0.3, 0.3); 
    command_[1] = dis1(gen);
    std::uniform_real_distribution<> dis2(-0.1, 0.1); 
    command_[2] = dis2(gen);

    if(terrainType_ == TerrainType::Slope)
      command_[2] *= 0.1;
  }


 
  void footSlipage(){  
    closedContact();
    int i = 0;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> footVelocity;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> footPosition;
    double foot_height;
    double tan_vel_norm = 0.0;
    slip_term_ = 0.0;

    for(std::set<size_t>::iterator it=footLinkFrame_.begin(); it!=footLinkFrame_.end(); ++it){
      anymal_->getFramePosition(*it, temp_footPosition);
      footPosition.push_back(temp_footPosition.e());
      anymal_->getFrameVelocity(*it, temp_footVelocity);
      footVelocity.push_back(temp_footVelocity.e());
    }

    for(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>::iterator it=footVelocity.begin(); it!=footVelocity.end(); ++it, i++){
      tan_vel_norm = (*it)(Eigen::seq(0,1)).norm(); //(*it) mi da il contenuto di footVelocity, che e' un Eigen vectore, con seq prendo i primi 2 elementi, e poi ne faccio la norma
      if(cF_[footNames_[i]]==1){
        slip_term_ += tan_vel_norm; // is reset when a new episode starts CAMBIAMENTOOOOO
      }
    }

    clearence_ = 0;
    int j= 0;
    for(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>::iterator it=footPosition.begin(); it!=footPosition.end(); ++it, j++){
      if(cF_[footNames_[j]]==0){ ///foot on the air
        foot_height = (*it)[2];//third coordinate of this eigen vector which is the height
        clearence_ += std::pow(foot_height - 0.22, 2)*tan_vel_norm;
      }
    }

  }

                           //Questo funziona perche' si basa sul fatto che gli indici in footIndices_ sono ordinati (LF_FOOT, RF_FOOT, LH_FOOT, RH_FOOT)
  void closedContact(){
    //closed contatct means that the foot is on the ground (stance phase)
    int i = 0;

    for(int j=0; j<4; j++)
      cF_[footNames_[j]]=0; 
    
    /*for(std::set<size_t>::iterator it=footIndices_.begin(); it!=footIndices_.end(); ++it, i++){
      for(auto& contact: anymal_->getContacts()){
        if (contact.skip()) continue;
        if(contact.getlocalBodyIndex() == *it){ //swipe the contact list, if you find a contact then the foot is on the terrain
            cF_[footNames_[i]]=1;  //Closed contact, foot on the terrain
            phi_[i] = 0;
            break;
        }
      } 
    }*/

    std::set<size_t>::iterator it = footIndices_.begin();
    
    int LF = *it;  //vale 3

    std::advance(it, 1);  //increment the iterator of 1 elements
    int RF = *it; //vale 6

    std::advance(it, 1);  //ncrement the iterator of 1 elemento. Attento che ora it punta al secondo elemento, quindi devi incrementarlo di 1. Se lo incrementi di 2, arrivi all'ultimo elemento
    int LH = *it; //vale 9

    std::advance(it, 1);  //increment the iterator of n elements
    int RH = *it; //vale 12

    for(auto& contact: anymal_->getContacts()){
      if (contact.skip()) continue;  //contact.skip() ritorna true se siamo in una self-collision, in quel caso ti ritorna il contatto 2 volte
      if(contact.getlocalBodyIndex() == LF ){
        cF_[footNames_[0]]=1;  //Closed contact, foot on the terrain
        auto end_LF = std::chrono::steady_clock::now();
        //air_time_LF_ = (std::chrono::duration_cast<std::chrono::microseconds>(end_LF - start_LF).count());
      }
      else if(contact.getlocalBodyIndex() == RF ){ //swipe the contact list, until you don't find that your foot is in contact
        cF_[footNames_[1]]=1;  //Closed contact, foot on the terrain
        auto end_RF = std::chrono::steady_clock::now();
        //air_time_RF_ = (std::chrono::duration_cast<std::chrono::microseconds>(end_RF - start_RF).count());
      }
      else if(contact.getlocalBodyIndex() == LH ){ //swipe the contact list, until you don't find that your foot is in contact
        cF_[footNames_[2]]=1;  //Closed contact, foot on the terrain
        auto end_LH = std::chrono::steady_clock::now();
        //air_time_LH_ = (std::chrono::duration_cast<std::chrono::microseconds>(end_LH - start_LH).count());
      }   
      else if(contact.getlocalBodyIndex() == RH ){ //swipe the contact list, until you don't find that your foot is in contact
        cF_[footNames_[3]]=1;  //Closed contact, foot on the terrain
        auto end_RH = std::chrono::steady_clock::now();
        //air_time_RH_ = (std::chrono::duration_cast<std::chrono::microseconds>(end_RH - start_RH).count());
      }         
    } 
    /*if(cF_[footNames_[0]]==0)
      start_LF = std::chrono::steady_clock::now(); //start time in air
    else if(cF_[footNames_[1]]==0)
      start_RF = std::chrono::steady_clock::now();
    else if(cF_[footNames_[2]]==0)
      start_LH = std::chrono::steady_clock::now(); 
    else if(cF_[footNames_[3]]==0)
      start_RH = std::chrono::steady_clock::now();*/
    
  }

  double sum_air_time(double & air_time_LF_, double & air_time_RF_, double & air_time_LH_, double & air_time_RH_){

    return air_time_LF_ + air_time_RF_ + air_time_LH_ + air_time_RH_;
  }


  void footTrajGenerator(){
      std::vector<double> phi = {0,0,0,0};
  }



  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }


    //This doesn't work, but to leave I should cancel out the call in VectorizedEnvironment.hpp
  void vel_vec(Eigen::Ref<EigenVec> vx, Eigen::Ref<EigenVec> vy, Eigen::Ref<EigenVec> omega){  //Gli do 3 vettori da riempire, e lui li riempe. 
    vx = v_x.cast<float>();
    vy = v_y.cast<float>();
    omega = omega_z.cast<float>();
  }



  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);
    /// if the contact body is NOT feet
    for(auto& contact: anymal_->getContacts()){
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end() && terrainType_ == Flat_){ //e' come fare un if(indice contatto != footIndeces_[i]) allora il contatto non e' ai piedi
        //find ritorna un iteratore, se l'elemento non e' stato trovato, ti ritorna un iterator a footIndices_.end()
        return true;
      }else if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end() && terrainType_ == Steps){ //il contatto e' alla base
        number_falls_steps_ ++; 
        return true;  
      }
      else if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end() && terrainType_ == PyramidStairs){ //il contatto e' alla base
        number_falls_stairs_ ++;
        return true;
      }
      else if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end() && terrainType_ == Slope){ //il contatto e' alla base
        number_falls_slope_ ++;   
        return true;
      }
      else if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end() && terrainType_ == Hills){ //il contatto e' alla base
        number_falls_hills_ ++;   
        return true;
      }
    }
    terminalReward = 0.f;
    return false;
  }




  void setFriction(float mu){
    for(int i = 0; i <4; i++){
      world_ ->setMaterialPairProp("ground_heightMap", footNames_[i], mu, 0.0, 0.001); //mu, bouayancy, restitution velocity (and minimum impact velocity to make the object bounce) 
    }
  }


  void generate_HeightMap(std::string terrain_type, double stepHeight, double SinusoidalAmplitude, double slopes, double pyramidHeight){
    
    std::uniform_real_distribution<> dis_01(0, 1);

    if(boost::iequals(terrain_type, "Hills")){ //questo metodo e' simile a type.compare("hills") ma e' case insensitive
      terrainType_ = TerrainType::Hills;
      setFriction(0.5);
    }
    else if(boost::iequals(terrain_type, "steps")){
      terrainType_ = TerrainType::Steps;
      setFriction(0.8);
    }      
    else if(boost::iequals(terrain_type, "Stairs")){
      terrainType_ = TerrainType::PyramidStairs;
      setFriction(0.8);
    }
    else if(boost::iequals(terrain_type, "Slope")){
      terrainType_ = TerrainType::Slope;
      setFriction(0.7);
    }
    if(terrainType_ == Steps){
      double pixelSize_ = 0.02;  //tipycal pixel size is 0.026
      terrainProp_.xSize = 14.0;
      terrainProp_.ySize = 28.0;
      terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
      terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

      heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

      double stepSize = 0.5 + 0.2;
      //double stepHeight = 0.5 + 0.05;   Defined before the if, in order to be updatable

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
          if(i==0 || j==0 || i==xNum -1 || j ==yNum -1) //at the border the steps are lower
              stepHeight = dis_01(gen) * stepHeight/4;

            h = dis_01(gen) * stepHeight;  //in questo modo gli steps esterni saranno piu' bassi di quelli interni
          if(h >0.2) h = 0.2;           
          //mapMat.setConstant(h);
          mapMat.block(gridWidth_ * i, gridWidth_ * j, gridWidth_, gridWidth_).setConstant(h);   //ogni campione xample, ysample e' un pixel. Quindi tu ti devi muovere di pixel in pixeo    
        }//questo codice dice, "parti da 0,0" e prendimi un blocco di dimensione gridWidth_*gridWidth_, quindi ti sta prendendo un intero step perche' gridWidth_*gridWidth_ e' la risoluzione di uno step
      }//poi si muove lungo le righe e le colonne, perche' si deve spostare di gridWidth_ pixel (cioe' gridWidth_ campioni, verso l'alto e il basso)
      //infine, con setconstant, definisce l'altezza di ogni step.
    
      mapMat.topRows(1).setConstant(0);  //il block parte da qui, quindi la mappa si inizia a creare da qui, quindi la prima riga e la prima colonna, saranno gia' riempiti con un valore di altezza maggiore di 0
      mapMat.leftCols(1).setConstant(0); //se io ho 2 pixel adiacenti, e sono tutti e 2 maggiori di 0, la mappa inizia dall'alto. Se invece uno e' 0 e l'altro maggiore da 0, allora l'interpolazione mi fa vedere uno step
      //la griglia si sposta ogni volta di gridWidth pixel, e li riempe con dei valori delle altezze. Se pero' ho una gridwidth=320 e la mia mappa e' di 650 elementi, allora mi riempe con dei valori delle altezze i primi 320, poi i secondi 320, ma gli ultimi 10 li lascia al valore di default (che e' 0)
      //quindi ora vorrei 'eliminare' quei pezzetti che restano fuori

      mapMat.bottomRows(1).setConstant(0);
      mapMat.rightCols(1).setConstant(0);
      //height_risultera' modificato alla fine, perche' alla fine mapMat e' costruito sulla base del vettore height_ che e' stato "matricizzato". Siccome poi gli abbiamo passato height_ per riferimento, le modifiche me le ritrovo anche su height.
    } 
    else if(terrainType_ == Hills){
      double pixelSize_ = 0.15;

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

    } 
    else if (terrainType_ == TerrainType::Slope) {

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

      std::cout<<int(2*mapMat.rows()/3);
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

    } 
    else if(terrainType_ == TerrainType::PyramidStairs){ 
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
    }

    double distanceFromAnymal = terrainProp_.xSize/2;

    if(terrainType_ != TerrainType::Flat_){
      terrain_ = world_->addHeightMap(terrainProp_.xSamples,
                                    terrainProp_.ySamples,
                                    terrainProp_.xSize,
                                    terrainProp_.ySize, anymal_pos_[0] + distanceFromAnymal + 1, 0.0, heights_, "ground_heightMap");  //ground is the name of the material, useful to set the friction
    }
    
    //metto l'ANYmal e il terreno generato a 1 metro di distanza
  }




  void curriculumUpdate() {//USO QUESTA FUNZIONE PERCHE' viene chiamata solo una volta a fine episodio e aggiorna tutto
        
    RSINFO_IF(visualizable_, "commandVel : "<< command_)
    generate_command_velocity();

    // In alternativa allo switch potevi usare il .compare di string "if(terrains[dist4(gen)].compare("steps")==0"

    //Rise gradually these parameters from a number less than 1, to 1
    
    if(terrainType_ != TerrainType::Flat_){
      switch (terrainType_){
        case(TerrainType::Steps):
          RSINFO_IF(visualizable_, "Terrain: Steps with height = "<<stepHeight_)
          if(number_falls_steps_ >= 100)
            stepHeight_ = 0.75*std::pow(10, 1/curriculumDecayFactor_*log10(stepHeight_));
          else{
            if(stepHeight_ < 0.3) //max height
              stepHeight_ = std::pow(stepHeight_, curriculumDecayFactor_);
            else 
              stepHeight_ = 0.3;
          }
          break;  //to exit from the switch

        case(TerrainType::Hills):   
          RSINFO_IF(visualizable_, "Terrain: Hills, with amplitude of the Perlin Noise = "<< SinusoidalAmplitude_)
          if(number_falls_hills_ >= 100)
            SinusoidalAmplitude_ = 0.75*std::pow(10, 1/curriculumDecayFactor_*log10(SinusoidalAmplitude_));
          else{
            if(SinusoidalAmplitude_ < 1)
              SinusoidalAmplitude_ = std::pow(SinusoidalAmplitude_, curriculumDecayFactor_);
            else 
              SinusoidalAmplitude_ = 1;
          }
          break;

        case(TerrainType::Slope):
          RSINFO_IF(visualizable_, "Terrain: ramp with slope = "<<slope_)
          if(number_falls_slope_ >= 100)
            slope_ = 0.75*std::pow(10, 1/curriculumDecayFactor_*log10(slope_));
          else{
            if(slope_ < 0.01)
              slope_ = std::pow(slope_, curriculumDecayFactor_);
            else 
              slope_ = 0.01;
          }        
          break;

        case(TerrainType::PyramidStairs):  
          RSINFO_IF(visualizable_, "Terrain: Pyramid steps with heights = "<< pyramidHeights_)
          if(number_falls_stairs_ >= 100)
            pyramidHeights_ = 0.75*std::pow(10, 1/curriculumDecayFactor_*log10(pyramidHeights_));
          else{
            if(pyramidHeights_ < 0.26)
              pyramidHeights_ = std::pow(pyramidHeights_, curriculumDecayFactor_);
            else 
              pyramidHeights_ = 0.26;
          }
      }
    
      world_->removeObject(terrain_);
      number_falls_steps_ = 0;
      number_falls_hills_ = 0;
      number_falls_slope_ = 0;
      number_falls_stairs_ = 0;
    }

      if(visualizable_)
        std::cout<<"numero episodio: "<<num_episode_<<std::endl;
    

    num_episode_++;

  }




 private:
  std::vector<std::string> footNames_ = {"LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"};
  int numberOfContact_ = 0;
  int num_seq;
  int num_legs_joint = 12;
  int gcDim_, gvDim_, nJoints_;
  int c_F_ = 1; //all'inizio, i piedi sono in contatto con il terreno. 
  int num_steps;
  double width_step, height_step;
  double curriculumFactor_, curriculumDecayFactor_;
  bool visualizable_ = false;
  bool twoTimeStep = false;
  bool nextTimeStep = false;
  raisim::ArticulatedSystem* anymal_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd ga_init_, ga_;
  double terminalRewardCoeff_ = -10.;
  //penalty reward
  double hipTerm_, thighTerm_, gaitTerm_Pos_, gaitTerm_Vel_;
  double timeDerivative = 0.001;

  Eigen::VectorXd actionMean_, actionStd_, obDouble_;

  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_, footLinkFrame_;

  size_t baseIndex_;
  int number_falls_steps_ = 0;
  int number_falls_stairs_ = 0;
  int number_falls_hills_ = 0;
  int number_falls_slope_ = 0;
  std::vector<size_t> steps_indices;
  std::vector<raisim::Box*> boxes;

  bool update_height_step = false;
  
  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;

  bool action_in_observation_space_ = false;
  bool joint_history_in_observation_space_ = false;
  Eigen::VectorXd previous_gv_;
  Eigen::VectorXd current_action_, previous_action_;

  Eigen::VectorXd joint_history_, joint_container_;
  Eigen::VectorXd joint_history_pos_, joint_history_vel_;
  Eigen::VectorXd q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12;

  Eigen::VectorXd standing_configuration_;
  Eigen::VectorXd actual_joint_position_, actual_joint_velocities_;

  //footSlipage variables
  double slip_term_ = 0.0;
  std::vector<double> clearence_foot = {0.0, 0.0, 0.0, 0.0};
  double clearence_ = 0.0;
  double forwardVel_x_ = 0.0;
  double forwardVel_y_ = 0.0;
  double angularVel_ = 0.0;
  double v_o_exp_ = 0.0;
  std::vector<double> phi_ = {1.9, 1.5, 1.1, 0.9};


  //std::vector<size_t> tan_vel_norm_vec;
  raisim::Vec<3> temp_footVelocity, temp_footPosition;


  raisim::Vec<3> anymal_pos_;
  size_t baseFrameIdx;
  //ClosedContact() (used even by footslipage)
  std::map<std::string,int> cF_ = {
    {"LF_FOOT", 1},
    {"RF_FOOT", 1},
    {"LH_FOOT", 1},
    {"RH_FOOT", 1}
  };

  std::vector<std::clock_t> start_time_vector_ = {0,0,0,0};
  std::vector<std::clock_t> end_time_vector_ = {0,0,0,0};
  //std::vector<int> cF_ = {1,1,1,1};
  Eigen::VectorXd v_x, v_y, omega_z;

  int num_episode_=0;
  Eigen::Vector3d command_;


    //Terrain
  std::vector<double> heights_;
  raisim::HeightMap *terrain_, *terrain_2;
  raisim::TerrainProperties terrainProp_;
  raisim::TerrainGenerator terrainGenerator_;  //only for the perlin noise
  raisim::TerrainType terrainType_;            //TerrainType is not a class, but an enumerator

    //UPDATABLE TERRAIN PARAMETERS
  double stepHeight_ = 0.01; //For Steps    //[0,0.3]
  double SinusoidalAmplitude_ = 0.05; //For the hills  [0.05, 1]
  double slope_ = 0.001;  //For UniformSlope    [0.001, 0.01]
  double pyramidHeights_ = 0.01; //For PyramidStairs   //[0,0.22]

  double weight_update_ = 1;

  //AIR TIME
  /*std::chrono::steady_clock::time_point start_LF_ = 0:
  std::chrono::steady_clock::time_point start_RF_ = 0:
  std::chrono::steady_clock::time_point start_LH_ = 0; 
  std::chrono::steady_clock::time_point start_RH_ = 0;*/
  
  double air_time_LF_;
  double air_time_RF_;
  double air_time_LH_;
  double air_time_RH_;
};


thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
}

