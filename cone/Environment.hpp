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
#include "heightMap.hpp"
# define M_PI           3.14159265358979323846
namespace raisim {


enum TerrainType {
  Flat_,
  Steps,
  Hills,
  Slope,
  PyramidStairs,
  Ice
};


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
    anymal_->getCollisionBody("LF_FOOT/0").setMaterial(footNames_[0]);  //prendo il nome al collision body. Ora questo nome, posso usarlo in setMaterialPairProp per dire che c'è attrito tra il terreno e quel corpo
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

    rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
    //rewards_.record("energy", anymal_->getGeneralizedForce().e().dot(gv_.tail(12))); //in Eigen the dot product between v and w is v.dot(w)
    rewards_.record("forwardVel_x", forwardVel_x_);
    rewards_.record("forwardVel_y", forwardVel_y_);
    rewards_.record("angularVel", angularVel_);
    rewards_.record("orthogonal_vel", v_o_exp_);
    rewards_.record("Joint_velocity", gv_.tail(12).norm());
    rewards_.record("Smooth_action", (current_action_ - previous_action_).norm()); //penalize big distance between 2 actions
    rewards_.record("omega_x", std::pow(bodyAngularVel_[0], 2));
    rewards_.record("omega_y", std::pow(bodyAngularVel_[1], 2));
    rewards_.record("v_z", std::pow(bodyLinearVel_[2], 2));
    rewards_.record("hip_penalty", hipTerm_); //The more the error is bigger, the more penalize this term
    rewards_.record("slippage", std::exp(-std::pow(H_, 2)/0.25));
    rewards_.record("clearence", clearence_);
    //rewards_.record("air_time", sum_air_time(air_time_LF_, air_time_RF_, air_time_LH_, air_time_RH_));

    previous_action_ = current_action_;  //Save the current action to compute the next reward tem
    return rewards_.sum();

  }

    
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
    std::uniform_real_distribution<> dis(0.2, 0.6);
    command_[0] = dis(gen);
    std::uniform_real_distribution<> dis1(-0.6, 0.6); 
    command_[1] = dis1(gen);
    std::uniform_real_distribution<> dis2(-0.2, 0.2); 
    command_[2] = dis2(gen);

    if(terrainType_ == TerrainType::Slope)
      command_[2] *= 0.1;
  }


 
  void footSlipage(){  
    FrictionCone();
    int i = 0;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> footVelocity;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> footPosition;
    double foot_height;
    double tan_vel_norm = 0.0;

    for(std::set<size_t>::iterator it=footLinkFrame_.begin(); it!=footLinkFrame_.end(); ++it){
      anymal_->getFramePosition(*it, temp_footPosition);
      footPosition.push_back(temp_footPosition.e());
      anymal_->getFrameVelocity(*it, temp_footVelocity);
      footVelocity.push_back(temp_footVelocity.e());
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
    for(auto& contact: anymal_->getContacts()){
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()){ //e' come fare un if(indice contatto != footIndeces_[i]) allora il contatto non e' ai piedi
        //Se l'if sopra e' soddisfatto, c'e' una collisione con qualcosa che non sono i piedi
        number_falls_ ++;
        return true;
      }
    }
    terminalReward = 0.f;
    return false;
  }


  Eigen::Matrix3d R_z_k(double & pi_j_k){

    Eigen::Matrix3d R_z {
      {std::cos(pi_j_k), -std::sin(pi_j_k), 0},
      {std::sin(pi_j_k), std::cos(pi_j_k), 0},
      {0, 0, 1}
    };

    return R_z;
    
  }


  void FrictionCone(){
    float mu = 0.9;
    int k = 4;
    int lambda = 4; //less than 1 gives some errors
    int i = 0;
    Eigen::Vector3d normal(0,0,1);
    Eigen::Vector3d contactForce;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> fc;

    double pi_j_k;
    double theta = std::atan(mu);
    double alpha = 0;
    

    Eigen::Matrix3d R_y_Theta {
      {std::cos(theta), 0, std::sin(theta)},
      {0, 1, 0}, 
      {-std::sin(theta), 0, std::cos(theta)}
    };

    for(int j=0; j<4; j++)
      cF_[footNames_[j]]=0; 
    
    H_ = 0;
    for(std::set<size_t>::iterator it=footIndices_.begin(); it!=footIndices_.end(); ++it, i++){

      for(auto& contact: anymal_->getContacts()){

        if (contact.skip()) continue;

        if(contact.getlocalBodyIndex() == *it){ //swipe the contact list, if you find a contact then the foot is on the terrain
            cF_[footNames_[i]]=1;  //Closed contact, foot on the terrain
            contactForce = (contact.getContactFrame().e().transpose() * contact.getImpulse().e()) / world_->getTimeStep();
            contactForce.normalize(); //a quanto pare e' stata implementata come void function, quindi non ha un return. D'ora in poi contactForce avra' norma 1
            alpha = std::acos( contact.getNormal().e().dot(contactForce));
            H_ += 1/(lambda*( (theta-alpha) * (theta+alpha)));
            break;
        }

      } 

    }

  }

  void setFriction(float mu, std::string heightMap_Name){
    for(int i = 0; i <4; i++){
      world_ ->setMaterialPairProp(heightMap_Name, footNames_[i], mu, 0.0, 0.001); //mu, bouayancy, restitution velocity (and minimum impact velocity to make the object bounce) 
    }
  }


  void select_heightMap(){
    //Lo faccio stampare a lui il numero episodio perche' e' la prima funzione chiamata nel runner.py
    if(visualizable_)
        std::cout<<"numero episodio: "<<num_episode_<<std::endl;
    
    bool ice = 1;
    if(ice)
      terrainType_ = TerrainType::Ice;
    

    if(terrainType_ != TerrainType::Ice){
      std::random_device rd;  // Will be used to obtain a seed for the random number engine
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dist4(0,3); // distribution in range [0, 3]
      std::vector<std::string> terrains = {"steps", "Hills", "Slope", "Stairs"};

      std::string terraingenerated = terrains[dist4(gen)];

      if(num_episode_ >=0){
        RSINFO_IF(visualizable_, "Terrain spawned : "<< terraingenerated)
        generate_HeightMap(terraingenerated, stepHeight_, SinusoidalAmplitude_, slope_, pyramidHeights_);
      }
    }else{
       if(num_episode_ >=0){
        RSINFO_IF(visualizable_, "Ice skating")
        Ice();
      }
    }

  }

  void generate_HeightMap(std::string terrain_type, double stepHeight, double SinusoidalAmplitude, double slopes, double pyramidHeight){

    if(boost::iequals(terrain_type, "Hills")){ //questo metodo e' simile a type.compare("hills") ma e' case insensitive
      terrainType_ = TerrainType::Hills;
      heightMap_.Hills(SinusoidalAmplitude);
    }
    else if(boost::iequals(terrain_type, "steps")){
      terrainType_ = TerrainType::Steps;
      heightMap_.Steps(stepHeight);
    }      
    else if(boost::iequals(terrain_type, "Stairs")){
      terrainType_ = TerrainType::PyramidStairs;
      heightMap_.PyramidStairs(pyramidHeight);
    }
    else if(boost::iequals(terrain_type, "Slope")){
      terrainType_ = TerrainType::Slope;
      heightMap_.Slope(slopes);
    }

    if(terrainType_ != TerrainType::Flat_){
      terrain_ = world_->addHeightMap(heightMap_.getTerrainProp().xSamples,
                                    heightMap_.getTerrainProp().ySamples,
                                    heightMap_.getTerrainProp().xSize,
                                    heightMap_.getTerrainProp().ySize,
                                    anymal_pos_[0] + heightMap_.distanceFromAnymal_ + 1, 0.0, heightMap_.getHeights(), "ground_heightMap");  //ground is the name of the material, useful to set the friction
    }
    setFriction(0.9, "ground_heightMap");

    if(terrainType_ != TerrainType::Slope)
      terrain_ -> setAppearance("red");
  }

  void Ice(){
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> mu(0.3, 0.8);
    heightMap_.Flat_with_friction();
    terrain_flat_ = world_->addHeightMap(heightMap_.getTerrainProp().xSamples,
                                    heightMap_.getTerrainProp().ySamples,
                                    heightMap_.getTerrainProp().xSize,
                                    heightMap_.getTerrainProp().ySize,
                                    anymal_pos_[0] + heightMap_.distanceFromAnymal_ + 1, 0.0, heightMap_.getHeights(), "Ice");  //ground is the name of the material, useful to set the friction
    terrain_flat_ -> setAppearance("blue");
    setFriction(mu(gen), "Ice");
  }



  void updateTerrainsParameters(int & numberFalls, double & terr_param, std::string typeOfTerrain){
    if(visualizable_)
      std::cout<<"Sul"<<typeOfTerrain<<"Sei caduto "<< numberFalls <<" volte"<<std::endl;
    RSINFO_IF(visualizable_, "Terrain " << typeOfTerrain <<" with height = " <<terr_param)
    if(numberFalls >= 1000){
      terr_param = 0.75*std::pow(10, 1/curriculumDecayFactor_*log10(terr_param));
      if(visualizable_)
        std::cout<<"Fell too many times. I'll make the terrain simpler "<<std::endl;
    }else{
      terr_param = std::pow(terr_param, curriculumDecayFactor_);
    }
  }

  void curriculumUpdate() {//USO QUESTA FUNZIONE PERCHE' viene chiamata solo una volta a fine episodio e aggiorna tutto
        
    RSINFO_IF(visualizable_, "commandVel : "<< command_)
    generate_command_velocity();

    // In alternativa allo switch potevi usare il .compare di string "if(terrains[dist4(gen)].compare("steps")==0"

    //Rise gradually these parameters from a number less than 1, to 1
    if(terrainType_ != TerrainType::Flat_ && terrainType_ != TerrainType::Ice ){
      switch (terrainType_){
        case(TerrainType::Steps):
          updateTerrainsParameters(number_falls_, stepHeight_, "Steps");
          if(stepHeight_ > 0.25)  
            stepHeight_ = 0.25;
          break;  //to exit from the switch

        case(TerrainType::Hills):
          updateTerrainsParameters(number_falls_, SinusoidalAmplitude_, "Hills");
          if(SinusoidalAmplitude_ >= 1)  
            SinusoidalAmplitude_ = 1;
          break;

        case(TerrainType::Slope):
          updateTerrainsParameters(number_falls_, slope_, "Slope");  
          if(slope_ >= 0.01)  
            slope_ = 0.01;
          break;

        case(TerrainType::PyramidStairs):  
          updateTerrainsParameters(number_falls_, pyramidHeights_, "Stairs");  
          if(pyramidHeights_ >= 0.22)
            pyramidHeights_ = 0.22;  
      }
    
      world_->removeObject(terrain_);

      if(visualizable_)
        std::cout<<"Controllo se ha resettato le cadute: "<< number_falls_<<std::endl;

    }



    num_episode_++;
    number_falls_ = 0;
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
  int number_falls_ = 0;
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
  double H_ = 0.0;
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
  raisim::Generate_HeightMap heightMap_;
  raisim::HeightMap *terrain_, *terrain_flat_;
  raisim::TerrainType terrainType_;            //TerrainType is not a class, but an enumerator

    //UPDATABLE TERRAIN PARAMETERS
  double stepHeight_ = 0.02; //For Steps    //[0,0.3]
  double SinusoidalAmplitude_ = 0.05; //For the hills  [0.05, 1]
  double slope_ = 0.001;  //For UniformSlope    [0.001, 0.01]
  double pyramidHeights_ = 0.01; //For PyramidStairs   //[0,0.22]

  double weight_update_ = 1;

  //AIR TIME
  /*std::chrono::steady_clock::time_point start_LF_ = 0:
  std::chrono::steady_clock::time_point start_RF_ = 0:
  std::chrono::steady_clock::time_point start_LH_ = 0; 
  std::chrono::steady_clock::time_point start_RH_ = 0;*/
  
};


thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
}

