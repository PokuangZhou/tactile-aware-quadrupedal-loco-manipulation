## Easy Setup

Run `install_deploy.sh` to install and build the deploy code.

```bash
cd deploy
./install_deploy.sh
```

The script will:

- download and build LCM under `thirdparty/lcm`
- build the deploy binaries
- install the Python package `go2_gym_deploy` with `pip install -e`
- create local Unitree DDS library symlinks needed at runtime

The Unitree SDK2 static library is already included under `deploy/unitree_sdk2_bin/library/unitree_sdk2`, so the script does not rebuild Unitree SDK2.

Generated binaries:

- `deploy/build/lcm_position_go2`
- `deploy/build/lcm_receive`
- `deploy/build/release_mcf`

## Cable connect to robot
Connect the robot using a cable, and set the ip, for example: 
```text
Address 192.168.123.188 
Netmask 255.255.255.0
```
check
```text
ip link show
ip addr show enx6c6e072d458f
# there should be 
inet 192.168.123.160/24 brd 192.168.123.255
```
## Open 3 terminal to deploy on real robot

🔹T1 lcm connection

👉 need replace the enx6c6e072d458f
```bash
cd deploy/build
sudo ./lcm_position_go2 enx6c6e072d458f
```

🔹T2 shut down local ctrl
```bash
cd deploy/build
sudo ./release_mcf enx6c6e072d458f
```

🔹T3 deploy policy 

go2d1 base task 
```bash
mamba activate dogtac
cd deploy/scripts/
python deploy_policy_cooperate.py --net enx6c6e072d458f
```

## Optional: Wi-Fi Setup

This section describes how to set up a Wi-Fi adapter on the Unitree Go2, transfer the project to the robot, build it on the Go2, and deploy a policy where the local PC performs GPU inference and communicates with the Go2 through ZeroMQ.

Requirements

- A USB Wi-Fi adapter for the Go2  
  Example: **BrosTrend AC1200**
- An Ethernet cable
- A local PC and the Unitree Go2 connected through Ethernet for the initial setup
- SSH and `scp` available on the local PC

---

1. Connect to the Go2 through Ethernet
First, connect the local PC to the Unitree Go2 using an Ethernet cable.
Then SSH into the Go2 Ubuntu system:
```bash
ssh unitree@192.168.123.18
```
Password: 123

2. Copy the project from the local PC to the Go2
Use scp to transfer the local project folder to the Go2:
scp -r <local_project_folder> unitree@192.168.123.18:<target_path_on_go2>

3. Build the program on the Go2
After copying the project, log in to the Go2 and build it there.

4. Set up Wi-Fi on the Go2
After installing and configuring the Wi-Fi adapter, the Go2 should show a wireless interface such as wlan0.
Make sure the Go2 and the local PC are connected to the same Wi-Fi network.

5. SSH into the Go2 through Wi-Fi
Once the Go2 is connected to Wi-Fi, find its Wi-Fi IP address and connect to it from the local PC.
Example:
ssh unitree@192.168.0.68

6. Run the policy on the local PC
The policy runs on the local PC, which uses its GPU for inference.
The local PC then sends control commands to the Go2 through ZeroMQ.
This setup keeps the heavy computation on the local machine while allowing the Go2 to execute the received commands onboard.

7. Deploy on the Go2
On the Go2 side, run the deployment program that receives commands from the local PC and applies them to the robot.

In summary:

The local PC runs the policy with GPU acceleration
The Go2 receives commands through ZeroMQ
The Go2 executes the commands on the robot
