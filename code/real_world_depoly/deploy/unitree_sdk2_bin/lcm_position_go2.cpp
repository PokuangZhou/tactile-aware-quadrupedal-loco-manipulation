// lcm related headfile
#include <lcm/lcm-cpp.hpp>
#include "leg_control_data_lcmt.hpp"
#include "state_estimator_lcmt.hpp"
#include "rc_command_lcmt.hpp"
#include "pd_tau_targets_lcmt.hpp"
// standard headfile
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cmath>
// unitree_sdk2 related headfile
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_JOYSTICK "rt/wirelesscontroller"

// Avoid the unitree_sdk2 using-directive to keep name resolution explicit and stable.

constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);


// Unitree-provided motor checksum function; no changes are required.
uint32_t crc32_core(uint32_t* ptr, uint32_t len)
{   
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }

    return CRC32;
}


// Remote-control key union copied from unitree_sdk2; no changes are required.
typedef union
{
  struct
  {
    uint8_t R1 : 1;
    uint8_t L1 : 1;
    uint8_t start : 1;
    uint8_t select : 1;
    uint8_t R2 : 1;
    uint8_t L2 : 1;
    uint8_t F1 : 1;
    uint8_t F2 : 1;
    uint8_t A : 1;
    uint8_t B : 1;
    uint8_t X : 1;
    uint8_t Y : 1;
    uint8_t up : 1;
    uint8_t right : 1;
    uint8_t down : 1;
    uint8_t left : 1;
  } components;
  uint16_t value;
} xKeySwitchUnion;


class Custom
{
public:
    explicit Custom(){}
    ~Custom(){}

    void Init();
    void InitLowCmd();
    void Loop();
    void LowStateMessageHandler(const void* messages);
    void JoystickHandler(const void *message);
    void InitRobotStateClient();
    void activateService(const std::string& serviceName,int activate);
    void lcm_send();
    void lcm_receive();
    void lcm_receive_Handler(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt* msg);
    void LowCmdWrite();
    void SetNominalPose();
    int queryServiceStatus(const std::string& serviceName);

    leg_control_data_lcmt leg_control_lcm_data = {0};
    state_estimator_lcmt body_state_simple = {0};
    pd_tau_targets_lcmt joint_command_simple = {0};
    rc_command_lcmt rc_command = {0};

    unitree_go::msg::dds_::LowState_ low_state{};
    unitree_go::msg::dds_::LowCmd_ low_cmd{};     
    unitree_go::msg::dds_::WirelessController_ joystick{};
    unitree::robot::go2::RobotStateClient rsc;
    
    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_suber;
    lcm::LCM lc;
    
    xKeySwitchUnion key;
    int mode = 0;
    int motiontime = 0;
    float dt = 0.005; // unit [second] (200 Hz low-level loop)
    bool _firstRun;

    /*LowCmd write thread*/
    // Thread pointers for DDS-related low-level command processing.
    unitree::common::ThreadPtr LcmSendThreadPtr;
    unitree::common::ThreadPtr LcmRecevThreadPtr;
    unitree::common::ThreadPtr lowCmdWriteThreadPtr;

};

void Custom::InitRobotStateClient()
{
    rsc.SetTimeout(5.0f); 
    rsc.Init();
}

int Custom::queryServiceStatus(const std::string& serviceName)
{
    std::vector<unitree::robot::go2::ServiceState> serviceStateList;
    int ret,serviceStatus;
    ret = rsc.ServiceList(serviceStateList);
    size_t i, count=serviceStateList.size();
    for (i=0; i<count; i++)
    {
        const unitree::robot::go2::ServiceState& serviceState = serviceStateList[i];
        if(serviceState.name == serviceName)
        {
            if(serviceState.status == 0)
            {
                std::cout << "name: " << serviceState.name <<" is activate"<<std::endl;
                serviceStatus = 1;
            }
            else
            {
                std::cout << "name:" << serviceState.name <<" is deactivate"<<std::endl;
                serviceStatus = 0;
            } 
        }    
    }
    return serviceStatus;
    
}

void Custom::activateService(const std::string& serviceName,int activate)
{
    rsc.ServiceSwitch(serviceName, activate);  
}

void Custom::LowStateMessageHandler(const void* message)
{
    // Low-level state read through SDK2.
    low_state = *(unitree_go::msg::dds_::LowState_*)message;
}

void Custom::JoystickHandler(const void *message)
{
    // Remote-control signal.
    joystick = *(unitree_go::msg::dds_::WirelessController_ *)message;
    key.value = joystick.keys();
}

// -------------------------------------------------------------------------------
// Thread 1: LCM sender.
// Read low_state and joystick signals through unitree_sdk2 and send them to LCM in real time.
void Custom::lcm_send(){
    // leg_control_lcm_data
    for (int i = 0; i < 12; i++)
    {
        leg_control_lcm_data.q[i] = low_state.motor_state()[i].q();
        leg_control_lcm_data.qd[i] = low_state.motor_state()[i].dq();
        leg_control_lcm_data.tau_est[i] = low_state.motor_state()[i].tau_est();
    }
    // Read orientation data from the IMU.
    for(int i = 0; i < 4; i++){
        // Quaternion.
        body_state_simple.quat[i] = low_state.imu_state().quaternion()[i]; 
    }
    for(int i = 0; i < 3; i++){
        // roll pitch yaw
        body_state_simple.rpy[i] = low_state.imu_state().rpy()[i];
        // IMU three-axis acceleration.
        body_state_simple.aBody[i] = low_state.imu_state().accelerometer()[i];
        // IMU three-axis angular velocity.
        body_state_simple.omegaBody[i] = low_state.imu_state().gyroscope()[i];
    }
    for(int i = 0; i < 4; i++){
        // Foot contact force.
        body_state_simple.contact_estimate[i] = low_state.foot_force()[i];
    }
    // Remote-control button and joystick values.
    rc_command.left_stick[0] = joystick.lx();
    rc_command.left_stick[1] = joystick.ly();
    rc_command.right_stick[0] = joystick.rx();
    rc_command.right_stick[1] = joystick.ry();
    rc_command.right_lower_right_switch = key.components.R2;
    rc_command.right_upper_switch = key.components.R1;
    rc_command.left_lower_left_switch = key.components.L2;
    rc_command.left_upper_switch = key.components.L1;

    if(key.components.A > 0){
        mode = 0;
    } else if(key.components.B > 0){
        mode = 1;
    }else if(key.components.X > 0){
        mode = 2;
    }else if(key.components.Y > 0){
        mode = 3;
    }else if(key.components.up > 0){
        mode = 4;
    }else if(key.components.right > 0){
        mode = 5;
    }else if(key.components.down > 0){
        mode = 6;
    }else if(key.components.left > 0){
        mode = 7;
    }

    rc_command.mode = mode;


    lc.publish("leg_control_data", &leg_control_lcm_data);
    lc.publish("state_estimator_data", &body_state_simple);
    lc.publish("rc_command", &rc_command);

    // std::cout << "loop: messsages are sending ......" << std::endl;
}


// -------------------------------------------------------------------------------
// Thread 2: LCM receiver.
// Read desired joint-control signals (q, qd, kp, kd, tau_ff) produced by the
// PyTorch neural network through LCM in real time.
// See go2_gym_deploy/envs/lcm_agent.py: the network outputs only the desired q;
// kp and kd are configurable, while qd and tau_ff are set to zero.
void Custom::lcm_receive_Handler(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt* msg){
    (void) rbuf;
    (void) chan;
    joint_command_simple = *msg; // Receive joint commands from the neural network.
}

// Process incoming LCM messages continuously using the recommended LCM pattern.
void Custom::lcm_receive(){
    while(true){
        lc.handle();
    }
}


// -------------------------------------------------------------------------------
// Thread 3: unitree_sdk2 command writer.
// Initialize low_cmd and apply the neural-network output after the state-machine checks.
void Custom::InitLowCmd()
{
    // The LowCmd head member is the frame header used by the CRC check.
    // Set head, levelFlag, gpio, and related fields to the example defaults.
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    /* LowCmd has 20 motorCmd members, each corresponding to a robot motor.
       Go2 has only 12 motors, so the first 12 entries are active and the
       remaining eight entries are reserved. */
    for(int i=0; i<20; i++)
    {
        /* Set motorCmd mode to 0x01 for servo mode. If a Go2 joint motor
           cannot be controlled during debugging, verify this value is 0x01. */
        low_cmd.motor_cmd()[i].mode() = (0x01);   // motor switch to servo (PMSM) mode
        low_cmd.motor_cmd()[i].q() = (PosStopF);
        low_cmd.motor_cmd()[i].dq() = (VelStopF);
        low_cmd.motor_cmd()[i].kp() = (0);
        low_cmd.motor_cmd()[i].kd() = (0);
        low_cmd.motor_cmd()[i].tau() = (0);
    }
}

void Custom::SetNominalPose(){
    // In addition to initializing communication, initialize the joint angles
    // for the lying-down pose and put each motor in position mode.
    for(int i = 0; i < 12; i++){
        joint_command_simple.qd_des[i] = 0;
        joint_command_simple.tau_ff[i] = 0;
        joint_command_simple.kp[i] = 20; 
        joint_command_simple.kd[i] = 0.5; 
    }

    // Joint angles for the lying-down pose.
    joint_command_simple.q_des[0] = -0.3;
    joint_command_simple.q_des[1] = 1.2;
    joint_command_simple.q_des[2] = -2.721;
    joint_command_simple.q_des[3] = 0.3;
    joint_command_simple.q_des[4] = 1.2;
    joint_command_simple.q_des[5] = -2.721;
    joint_command_simple.q_des[6] = -0.3;
    joint_command_simple.q_des[7] = 1.2;
    joint_command_simple.q_des[8] = -2.721;
    joint_command_simple.q_des[9] = 0.3;
    joint_command_simple.q_des[10] = 1.2;
    joint_command_simple.q_des[11] = -2.721;

    std::cout<<"SET NOMINAL POSE"<<std::endl;
}

void Custom::LowCmdWrite(){
    motiontime++;
    
    if(_firstRun && leg_control_lcm_data.q[0] != 0){
        for(int i = 0; i < 12; i++){
            // On the first pass, use the current joint angles as the targets.
            joint_command_simple.q_des[i] = leg_control_lcm_data.q[i];
            // Initialize L2+B state to prevent accidental damping activation.
            key.components.Y = 0;
            key.components.A = 0;
            key.components.B = 0;
            key.components.L2 = 0;
        }
        _firstRun = false;
    } 

    // Safety fallback for excessive roll or pitch, or an L2+B button press.
    // if (  low_state.imu_state().rpy()[0] > 0.5 || low_state.imu_state().rpy()[1] > 0.5 || ((int)key.components.B==1 && (int)key.components.L2==1))
    if ( std::abs(low_state.imu_state().rpy()[0]) > 0.8 || std::abs(low_state.imu_state().rpy()[1]) > 0.8 || ((int)key.components.B==1 && (int)key.components.L2==1))
    {       
        for (int i = 0; i < 12; i++){
            // Enter damping mode.
            low_cmd.motor_cmd()[i].q() = 0;
            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = 0;
            low_cmd.motor_cmd()[i].kd() = 5;
            low_cmd.motor_cmd()[i].tau() = 0;
        }  
        std::cout << "======= Switched to Damping Mode, and the thread is sleeping ========"<<std::endl;
        sleep(1.5);

        while (true)
        {   
            
            // sleep(0.1);

            if (((int)key.components.B==1 && (int)key.components.L2==1) ) {
                // [L2+B] is pressed again
                std::cout << "======= [L2+B] is pressed again, the script is about to exit========" <<std::endl;
                exit(0);
            } else if (((int)key.components.A==1 && (int)key.components.L2==1) ){
                rsc.ServiceSwitch("sport_mode", 1);
                std::cout << "======= activate sport_mode service and exit========" <<std::endl;
                sleep(0.5);
                exit(0);
            } else{   
                if (((int)key.components.Y==1 && (int)key.components.L2==1) ){
                    std::cout << "=======  Switch to Walk These Ways ========"<<std::endl;
                    std::cout<<"Communicatino is set up successfully" << std::endl;
                    std::cout<<"LCM <<<------------>>> Unitree SDK2" << std::endl;
                    std::cout<<"------------------------------------" << std::endl;
                    std::cout<<"------------------------------------" << std::endl;
                    std::cout<<"Press L2+B if any unexpected error occurs" << std::endl;
                    break;
                    
                }else{
                    std::cout << "======= Press [L2+B] again to exit ========"<<std::endl;
                    std::cout << "======= Press [L2+Y] again to switch to WTW ========"<<std::endl;
                    std::cout << "======= Press [L2+A] again to activate sport_mode service========"<<std::endl;
                    sleep(0.01);
                }
                
            }

        }
        
    } 
    else{
        for (int i = 0; i < 12; i++){
            // Apply the neural-network output only while the safety checks pass.
            low_cmd.motor_cmd()[i].q() = joint_command_simple.q_des[i];
            low_cmd.motor_cmd()[i].dq() = joint_command_simple.qd_des[i];
            low_cmd.motor_cmd()[i].kp() = joint_command_simple.kp[i];
            low_cmd.motor_cmd()[i].kd() = joint_command_simple.kd[i];
            low_cmd.motor_cmd()[i].tau() = joint_command_simple.tau_ff[i];
        }  
    }
    
    /* Compute the CRC, then send the control command to the Go2 robot through
       lowcmd_publisher::Write(). */
    low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_)>>2)-1);
    lowcmd_publisher->Write(low_cmd);
}


//
// End of recurrent-thread function definitions.
//----------------------------------------------------------------------



void Custom::Init(){
    _firstRun = true;
    InitLowCmd();
    SetNominalPose();

    // Configure lc.handle() to pass "pd_plustau_targets" messages to
    // lcm_receive_Handler.
    lc.subscribe("pd_plustau_targets", &Custom::lcm_receive_Handler, this);

    /*create low_cmd publisher*/
    lowcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    lowcmd_publisher->InitChannel();
    /*create low_state dds subscriber*/
    lowstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(std::bind(&Custom::LowStateMessageHandler, this, std::placeholders::_1), 1);
    /*create joystick dds subscriber*/
    joystick_suber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));
    joystick_suber->InitChannel(std::bind(&Custom::JoystickHandler, this, std::placeholders::_1), 1);
}


void Custom::Loop(){
    // Use recurrent threads to run the loop functions.

    // intervalMicrosec: 1 microsecond = 0.000001 seconds.
    // When dt = 0.002 s,
    // ntervalMicrosec = 2000us
    /*lcm send thread*/
    LcmSendThreadPtr = unitree::common::CreateRecurrentThreadEx("lcm_send_thread", UT_CPU_ID_NONE, dt*1e6, &Custom::lcm_send, this);
    /*lcm receive thread*/
    LcmRecevThreadPtr = unitree::common::CreateRecurrentThreadEx("lcm_recev_thread", UT_CPU_ID_NONE, dt*1e6, &Custom::lcm_receive, this);
    /*low command write thread*/
    lowCmdWriteThreadPtr = unitree::common::CreateRecurrentThreadEx("dds_write_thread", UT_CPU_ID_NONE, dt*1e6, &Custom::LowCmdWrite, this);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        exit(-1);
    }

    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is hung up." << std::endl
              << "Caution: The scripts is about to shutdown Unitree sport_mode Service." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]); // Local network interface (PC or Jetson Orin).

    Custom custom;

    custom.InitRobotStateClient();
    if(custom.queryServiceStatus("sport_mode"))
    {
        std::cout<<"Trying to deactivate the service: " << "sport_mode" << std::endl;
        custom.activateService("sport_mode",0);
        sleep(0.5);
        if(!custom.queryServiceStatus("sport_mode")){
            std::cout<<"Trying to deactivate the service: " << "sport_mode" << std::endl;
        }
    } else{
        std::cout <<"sportd_mode is already deactivated now" << std::endl
                  <<"next step is setting up communication" << std::endl
                  << "Press Enter to continue..." << std::endl;
        std::cin.ignore();
    }


    custom.Init();

    std::cout<<"Communicatino is set up successfully" << std::endl;
    std::cout<<"LCM <<<------------>>> Unitree SDK2" << std::endl;
    std::cout<<"------------------------------------" << std::endl;
    std::cout<<"------------------------------------" << std::endl;
    std::cout<<"Press L2+B if any unexpected error occurs" << std::endl;

    custom.Loop();

    // while (true)
    // {
    //     sleep(10);
    // }
    while (true)
        {
            std::cout << "=== 10 second check ===" << std::endl;
            custom.queryServiceStatus("sport_mode");
            sleep(10);
        }

    return 0;
}
